"""
orchestrator_with_critic.py
----------------------------
带 Critic Agent + ErrorAwareLLM 的编排器。

实现方式：继承 OrchestratorAgent，仅覆盖 _execute_step_with_retry。
原始 orchestrator.py 零修改。

新增能力（相对于原 Orchestrator）：
  1. ErrorAwareLLMClient：每次 LLM 调用自动携带上一轮的运行时错误
     机制：_tech_retry 每次调用 agent.execute_step 前，用 wrap_agent_llm
           将 agent.llm 替换为代理版本；执行后用 unwrap_agent_llm 还原
  2. CriticAgent：每步执行成功后自动评审产物
     机制：评审通过 → 继续；REVISE → 生成 RevisionPlan 后重新执行；
           ESCALATE → 人工介入

执行流程（单步）：
    ┌─ Critic 修改循环（最多 max_revisions+1 轮）──────────────────┐
    │   ┌─ 技术重试（最多 max_retries 次）──────────────────────┐  │
    │   │  wrap_agent_llm → agent.execute_step → unwrap         │  │
    │   │  失败: save_step_error → (下一次重试时自动注入错误)    │  │
    │   └────────────────────────────────────────────────────── ┘  │
    │   Critic.review → PASS / REVISE / ESCALATE                   │
    └──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import time

from orchestrator import OrchestratorAgent, PipelineConfig, PipelineStatus
from utils.state_store import StepStatus
from utils.logger import setup_logger, LLMClient
from critic.critic_agent import CriticAgent, Decision
from critic.error_aware_llm import wrap_agent_llm, unwrap_agent_llm

logger = setup_logger("orchestrator_critic")


class CriticOrchestratorAgent(OrchestratorAgent):
    """
    继承 OrchestratorAgent，覆盖 _execute_step_with_retry 一个方法，
    插入 ErrorAwareLLM 包装 + Critic 评审循环。
    """

    def __init__(self, config: PipelineConfig, max_revisions: int = 2):
        super().__init__(config)

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
            f"（Critic 最大修改轮数={max_revisions}）"
        )

    # ── 覆盖核心调度方法 ──────────────────────────

    def _execute_step_with_retry(
        self, step_id: str, phase: int, description: str
    ) -> bool:
        """
        外层：Critic 修改循环（最多 max_revisions+1 轮，含初次执行）
        内层：技术重试（_tech_retry，最多 max_retries 次）
              每次技术重试前用 wrap_agent_llm 注入错误感知 LLM
        """
        max_outer = self.critic.max_revisions + 1

        for critic_round in range(1, max_outer + 1):
            round_label = (
                "初次执行" if critic_round == 1
                else f"Critic 修改第 {critic_round - 1} 轮"
            )
            logger.info(f"  [{round_label}] {step_id}")

            # 内层技术重试（含 wrap/unwrap）
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

    # ── 内层技术重试（wrap/unwrap 在这里）────────────

    def _tech_retry(self, step_id: str, phase: int, description: str) -> bool:
        """
        执行单步的技术重试。
        每次尝试前：wrap_agent_llm 将 agent.llm 替换为 ErrorAwareLLMClient
        每次尝试后：unwrap_agent_llm 还原 agent.llm（无论成功还是失败）

        ErrorAwareLLMClient 在每次 LLM 调用前自动读取 StateStore 中
        该步骤的 last_error，追加到 user_prompt 末尾，让 LLM 知道
        上一轮出了什么问题并主动修复。
        """
        agent = self.phase_agents[phase]
        max_retries = self.config.max_retries

        for attempt in range(1, max_retries + 1):
            # ── 安装 ErrorAwareLLMClient ──────────────────────────────
            original_llm = wrap_agent_llm(agent, self.state, step_id)

            try:
                self.state.set_step_status(step_id, StepStatus.RUNNING)
                self.state.increment_retry(step_id)

                result = agent.execute_step(step_id)

                self.state.set_step_status(step_id, StepStatus.SUCCESS)
                self.state.save_step_result(step_id, result)
                logger.info(f"  ✓ {step_id} 执行成功（第 {attempt} 次尝试）")
                return True

            except Exception as e:
                err_msg = str(e)
                logger.warning(
                    f"  ✗ {step_id} 第 {attempt} 次执行失败: {err_msg[:300]}"
                )
                self.state.set_step_status(step_id, StepStatus.FAILED)
                # 写入 last_error → 下次重试时 ErrorAwareLLMClient 自动注入
                self.state.save_step_error(step_id, err_msg)

                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.info(
                        f"  等待 {wait}s 后重试"
                        f"（下次 LLM 调用将自动携带此错误信息）..."
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
        description="ML Service Pipeline（含 Critic Agent + ErrorAware LLM）"
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
        "--max-revisions", default=2, type=int,
        help="Critic Agent 最大修改轮数（默认 2）"
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
    orchestrator = CriticOrchestratorAgent(config, max_revisions=args.max_revisions)
    result = orchestrator.run()
    sys.exit(0 if result.status == PipelineStatus.SUCCESS else 1)
