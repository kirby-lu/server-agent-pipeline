"""
orchestrator_with_critic.py
----------------------------
带 Critic Agent + ErrorAwareLLM 的编排器。

实现方式：继承 OrchestratorAgent，仅覆盖 _execute_step_with_retry。
原始 orchestrator.py 零修改。

新增能力（相对于原 Orchestrator）：
  1. ErrorAwareLLMClient：每次 LLM 调用自动携带上一轮的运行时错误 + 生成代码
     机制：_tech_retry 每次调用 agent.execute_step 前，用 wrap_agent_llm
           将 agent.llm 替换为代理版本；执行后用 unwrap_agent_llm 还原。
           执行成功后调用 collect_code_paths_for_step + save_step_generated_code
           将本轮产出的代码文件路径记录到 StateStore，供下一轮注入。
  2. CriticAgent：每步执行成功后自动评审产物
     机制：评审通过 → 继续；REVISE → 生成 RevisionPlan 后重新执行；
           ESCALATE → 人工介入

执行流程（单步）：
    ┌─ Critic 修改循环（最多 max_revisions+1 轮）──────────────────────────┐
    │   ┌─ 技术重试（最多 max_retries 次）──────────────────────────────┐  │
    │   │  wrap_agent_llm → agent.execute_step → unwrap                 │  │
    │   │  成功: save_step_generated_code（记录代码路径）                │  │
    │   │  失败: save_step_error → (下一次重试时自动注入错误+代码)       │  │
    │   └────────────────────────────────────────────────────────────── ┘  │
    │   Critic.review → PASS / REVISE / ESCALATE                           │
    └──────────────────────────────────────────────────────────────────────┘

prompt 注入顺序（仅在重试/修改轮时生效）：
    [原始 user_prompt]
    + [上一轮生成的代码（带文件名标注）]   ← ErrorAwareLLMClient 自动注入
    + [上一轮执行错误]                      ← ErrorAwareLLMClient 自动注入
"""

from __future__ import annotations

import time

from orchestrator import OrchestratorAgent, PipelineConfig, PipelineStatus
from utils.state_store import StepStatus
from utils.logger import setup_logger, LLMClient
from critic.critic_agent import CriticAgent, Decision
from critic.error_aware_llm import (
    wrap_agent_llm,
    unwrap_agent_llm,
    save_step_generated_code,
    collect_code_paths_for_step,
)

logger = setup_logger("orchestrator_critic")


class CriticOrchestratorAgent(OrchestratorAgent):
    """
    继承 OrchestratorAgent，覆盖 _execute_step_with_retry 一个方法，
    插入 ErrorAwareLLM 包装（错误 + 代码双注入）+ Critic 评审循环。
    """

    def __init__(
        self,
        config: PipelineConfig,
        max_revisions: int = 2,
        inject_code: bool = True,
        max_code_length: int = 6000,
        max_error_length: int = 3000,
    ):
        """
        Parameters
        ----------
        config           : Pipeline 配置
        max_revisions    : Critic 最大修改轮数
        inject_code      : 是否在重试时将上一轮生成的代码也注入 prompt（默认 True）
        max_code_length  : 代码注入的最大字符数（所有文件合计）
        max_error_length : 错误信息注入的最大字符数
        """
        super().__init__(config)

        self._inject_code = inject_code
        self._max_code_length = max_code_length
        self._max_error_length = max_error_length

        # Critic Agent 使用独立 LLM 实例，避免被错误信息污染
        _critic_llm = LLMClient(model=config.llm_model)
        self.critic = CriticAgent(
            llm=_critic_llm,
            state=self.state,
            max_revisions=max_revisions,
            use_llm_for_borderline=True,
        )
        logger.info(
            f"  CriticOrchestratorAgent 初始化完成"
            f"（Critic 最大修改轮数={max_revisions}，"
            f"代码注入={'开启' if inject_code else '关闭'}，"
            f"代码最大长度={max_code_length}，"
            f"错误最大长度={max_error_length}）"
        )

    # ── 覆盖核心调度方法 ──────────────────────────

    def _execute_step_with_retry(
        self, step_id: str, phase: int, description: str
    ) -> bool:
        """
        外层：Critic 修改循环（最多 max_revisions+1 轮，含初次执行）
        内层：技术重试（_tech_retry，最多 max_retries 次）
              每次技术重试前用 wrap_agent_llm 注入错误感知 LLM（含代码+错误）
        """
        max_outer = self.critic.max_revisions + 1

        for critic_round in range(1, max_outer + 1):
            round_label = (
                "初次执行" if critic_round == 1
                else f"Critic 修改第 {critic_round - 1} 轮"
            )
            logger.info(f"  [{round_label}] {step_id}")

            # 内层技术重试（含 wrap/unwrap + 代码路径保存）
            tech_success = self._tech_retry(step_id, phase, description)

            if not tech_success:
                logger.error(f"  {step_id} 技术执行失败，已超过最大重试次数")
                return False

            # Critic 评审
            verdict = self.critic.review(step_id)
            self._log_verdict(verdict)

            if verdict.decision == Decision.PASS:
                return True

            elif verdict.decision == Decision.REVISE:
                self.state.set_step_status(step_id, StepStatus.FAILED)
                instructions = (
                    verdict.revision_plan.instructions
                    if verdict.revision_plan else []
                )
                logger.info(
                    f"  [Critic] 修改计划已下发，准备第 {critic_round} 轮修改\n"
                    + "\n".join(f"    • {inst}" for inst in instructions)
                )
                time.sleep(1)
                continue

            else:  # ESCALATE
                logger.warning(
                    f"  [Critic] {step_id} 升级人工介入\n"
                    f"  原因: {verdict.summary}"
                )
                self.state.set(f"critic_escalation_{step_id}", {
                    "summary": verdict.summary,
                    "failed_checks": [
                        {"name": c.name, "detail": c.detail}
                        for c in verdict.checks if not c.passed
                    ],
                    "score": verdict.score,
                })

                if self.config.human_in_the_loop:
                    approved = self._human_checkpoint_escalation(step_id, verdict)
                    if approved == "y":
                        self.state.set_step_status(step_id, StepStatus.SUCCESS)
                        return True
                    elif approved == "r":
                        self.critic.reset_revision_count(step_id)
                        self.state.set_step_status(step_id, StepStatus.FAILED)
                        return self._execute_step_with_retry(step_id, phase, description)
                    else:
                        return False
                else:
                    return False

        return False

    # ── 内层技术重试（wrap/unwrap + 代码路径保存 在这里）────────────

    def _tech_retry(self, step_id: str, phase: int, description: str) -> bool:
        """
        执行单步的技术重试。

        每次尝试前：
          wrap_agent_llm 将 agent.llm 替换为 ErrorAwareLLMClient，
          该代理在每次 LLM 调用前自动读取 StateStore 中该步骤的：
            - generated_code_paths → 上一轮生成的代码文件内容
            - last_error / stderr_tail / stdout_tail → 上一轮运行时错误
          并将两者（代码在前、错误在后）追加到 user_prompt 末尾。

        每次成功后：
          collect_code_paths_for_step 收集本轮产出的代码文件路径，
          save_step_generated_code 将路径写入 StateStore，
          供下一轮（若 Critic 要求修改）注入使用。

        每次尝试后：
          unwrap_agent_llm 还原 agent.llm（无论成功还是失败）。
        """
        agent = self.phase_agents[phase]
        max_retries = self.config.max_retries

        for attempt in range(1, max_retries + 1):
            # ── 安装 ErrorAwareLLMClient（注入代码 + 错误）──────────────────
            original_llm = wrap_agent_llm(
                agent,
                self.state,
                step_id,
                inject_code=self._inject_code,
                max_error_length=self._max_error_length,
                max_code_length=self._max_code_length,
            )

            try:
                self.state.set_step_status(step_id, StepStatus.RUNNING)
                self.state.increment_retry(step_id)

                result = agent.execute_step(step_id)

                self.state.set_step_status(step_id, StepStatus.SUCCESS)
                self.state.save_step_result(step_id, result)

                # ── 保存本轮生成的代码文件路径，供下一轮 Critic 修改时注入 ──
                if self._inject_code:
                    code_paths = collect_code_paths_for_step(self.state, step_id)
                    if code_paths:
                        save_step_generated_code(self.state, step_id, code_paths)
                        logger.debug(
                            f"  ✓ {step_id} 已保存 {len(code_paths)} 个代码文件路径"
                        )

                logger.info(f"  ✓ {step_id} 执行成功（第 {attempt} 次尝试）")
                return True

            except Exception as e:
                err_msg = str(e)
                logger.warning(
                    f"  ✗ {step_id} 第 {attempt} 次执行失败: {err_msg}"
                )
                self.state.set_step_status(step_id, StepStatus.FAILED)
                # 写入 last_error → 下次重试时 ErrorAwareLLMClient 自动注入
                self.state.save_step_error(step_id, err_msg)

                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.info(
                        f"  等待 {wait}s 后重试"
                        f"（下次 LLM 调用将自动携带上一轮代码+错误信息）..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"  {step_id} 已达最大技术重试次数 ({max_retries})"
                    )

            finally:
                # ── 无论成功/失败都还原 agent.llm ──────────────────────
                unwrap_agent_llm(agent, original_llm)

        return False

    # ── Critic 升级后的人工交互 ───────────────────────

    def _human_checkpoint_escalation(self, step_id: str, verdict) -> str:
        self.state.set_pipeline_status(PipelineStatus.PAUSED)

        print("\n" + "═" * 60)
        print(f"  ⚠️  Critic 升级: {step_id}")
        print(f"  评分: {verdict.score:.0f}/100")
        print(f"  原因: {verdict.summary}")
        print("─" * 60)
        print("  未通过检查项:")
        for check in verdict.checks:
            if not check.passed:
                print(f"    ✗ {check.name}")
                if check.detail:
                    print(f"      {check.detail}")
        print("═" * 60)

        while True:
            answer = input(
                "  操作选项:\n"
                "    y — 强制通过（跳过此步骤的 Critic 评审）\n"
                "    r — 重置并重新执行（清除所有修改轮数）\n"
                "    n — 终止 Pipeline\n"
                "  请选择 [y/r/n]: "
            ).strip().lower()
            if answer in ("y", "r", "n"):
                self.state.set_pipeline_status(PipelineStatus.RUNNING)
                return answer
            print("  请输入 y / r / n")

    # ── 日志辅助 ──────────────────────────────────────

    @staticmethod
    def _log_verdict(verdict) -> None:
        icons = {Decision.PASS: "✅", Decision.REVISE: "🔄", Decision.ESCALATE: "🚨"}
        icon = icons.get(verdict.decision, "?")
        logger.info(
            f"  [Critic] {icon} {verdict.step_id} → {verdict.decision.value} "
            f"(score={verdict.score:.0f}, round={verdict.review_round})"
        )
        if verdict.summary:
            logger.info(f"  [Critic] 摘要: {verdict.summary}")


# ─────────────────────────────────────────────
#  CLI 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="ML Service Pipeline（含 Critic Agent + ErrorAware LLM 代码+错误双注入）"
    )
    parser.add_argument("--gitlab-url",    required=True)
    parser.add_argument("--project-name",  required=True)
    parser.add_argument("--work-dir",      default="/tmp/ml_pipeline")
    parser.add_argument("--model",         default="claude-sonnet-4-20250514")
    parser.add_argument("--ip",            default="localhost")
    parser.add_argument("--port",          default=8080, type=int)
    parser.add_argument("--gpu",           action="store_true")
    parser.add_argument("--no-human",      action="store_true")
    parser.add_argument("--docker-image",  default="")
    parser.add_argument(
        "--max-revisions", default=3, type=int,
        help="Critic Agent 最大修改轮数（默认 3）"
    )
    parser.add_argument(
        "--no-inject-code", action="store_true",
        help="禁用代码注入（仅注入错误信息，恢复旧行为）"
    )
    parser.add_argument(
        "--max-code-length", default=60000, type=int,
        help="代码注入最大字符数（所有文件合计，默认 60000）"
    )
    parser.add_argument(
        "--max-error-length", default=30000, type=int,
        help="错误信息注入最大字符数（默认 30000）"
    )
    args = parser.parse_args()

    config = PipelineConfig(
        gitlab_url=args.gitlab_url,
        project_name=args.project_name,
        work_dir=args.work_dir,
        llm_model=args.model,
        server_ip=args.ip,
        server_port=args.port,
        gpu_available=args.gpu,
        human_in_the_loop=not args.no_human,
        docker_image_name=args.docker_image or f"{args.project_name}:latest",
    )
    orchestrator = CriticOrchestratorAgent(
        config,
        max_revisions=args.max_revisions,
        inject_code=not args.no_inject_code,
        max_code_length=args.max_code_length,
        max_error_length=args.max_error_length,
    )
    result = orchestrator.run()
    sys.exit(0 if result.status == PipelineStatus.SUCCESS else 1)
