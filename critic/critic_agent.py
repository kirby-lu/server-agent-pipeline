"""
critic/critic_agent.py
-----------------------
Critic Agent — 自动化阶段评审与修改指令生成

设计原则：
- 零侵入：不修改任何现有文件，通过 Orchestrator 的钩子接入
- 职责分离：只做评审+决策，不直接修改任何产出文件
- 结构化输出：评审结果为 CriticVerdict，修改指令为 RevisionPlan

接入方式：
    在 orchestrator.py 中，将 _execute_step_with_retry 的结果传给
    CriticAgent.review(step_id, state)，根据返回的 verdict 决定
    是通过、要求修改还是升级人工。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from utils.logger import setup_logger, LLMClient
from utils.state_store import StateStore

logger = setup_logger("critic_agent")


# ─────────────────────────────────────────────
#  数据结构
# ─────────────────────────────────────────────

class Decision(str, Enum):
    PASS    = "pass"      # 通过，进入下一步
    REVISE  = "revise"    # 需要修改，生成 RevisionPlan 后重试
    ESCALATE = "escalate" # 超过最大修改轮数，升级人工


@dataclass
class CheckItem:
    """单条验收检查项"""
    name: str           # 检查项名称
    passed: bool        # 是否通过
    detail: str = ""    # 说明/失败原因


@dataclass
class CriticVerdict:
    """
    Critic 对单个步骤的评审结论。
    decision=PASS  → Orchestrator 继续执行下一步
    decision=REVISE → Orchestrator 重新执行该步骤，并将 revision_plan 注入 state
    decision=ESCALATE → Orchestrator 暂停并等待人工
    """
    step_id: str
    decision: Decision
    score: float                          # 0~100 综合评分
    checks: list[CheckItem] = field(default_factory=list)
    summary: str = ""                     # 一句话总结
    revision_plan: Optional["RevisionPlan"] = None
    review_round: int = 1                 # 当前是第几轮评审


@dataclass
class RevisionPlan:
    """
    Critic 生成的修改计划，写入 StateStore 后由对应 Phase Agent 读取。
    Phase Agent 在 execute_step 开头调用 state.get('revision_plan') 获取。
    """
    step_id: str
    failed_checks: list[str]              # 未通过的检查项名称列表
    instructions: list[str]               # 具体修改指令（有序）
    context: dict = field(default_factory=dict)  # 附加上下文（如错误日志摘要）
    round_number: int = 1


# ─────────────────────────────────────────────
#  每个步骤的验收标准 (Rubric)
# ─────────────────────────────────────────────

class RubricLibrary:
    """
    存放每个 step_id 对应的验收检查逻辑。
    每个 rubric 方法接收 state: StateStore，返回 list[CheckItem]。
    纯规则检查不调用 LLM，降低延迟和成本。
    """

    @staticmethod
    def for_step(step_id: str) -> Optional["RubricLibrary._RubricFn"]:
        mapping = {
            "step_03": RubricLibrary._rubric_step03,
            "step_04": RubricLibrary._rubric_step04,
            "step_05": RubricLibrary._rubric_step05,
            "step_06": RubricLibrary._rubric_step06,
            "step_07": RubricLibrary._rubric_step07,
            "step_08": RubricLibrary._rubric_step08,
            "step_09": RubricLibrary._rubric_step09,
            "step_11": RubricLibrary._rubric_step11,
            "step_13": RubricLibrary._rubric_step13,
        }
        return mapping.get(step_id)

    # ── step_03: 资源下载 ─────────────────────────

    @staticmethod
    def _rubric_step03(state: StateStore) -> list[CheckItem]:
        result = state.get_step_result("step_03") or {}
        checks = []

        # 检查1：下载过程本身没有报错
        checks.append(CheckItem(
            name="下载流程无异常",
            passed=result.get("resources_downloaded", False) is not False,
            detail="resources_downloaded 标志为 False" if not result.get("resources_downloaded", True) else "",
        ))

        # 检查2：若有失败资源，列出来
        failed = result.get("failed", [])
        checks.append(CheckItem(
            name="无失败下载项",
            passed=len(failed) == 0,
            detail=f"以下资源下载失败: {failed}" if failed else "",
        ))

        return checks

    # ── step_04: 原型验证 ─────────────────────────

    @staticmethod
    def _rubric_step04(state: StateStore) -> list[CheckItem]:
        result = state.get_step_result("step_04") or {}
        checks = []

        checks.append(CheckItem(
            name="原型脚本执行成功",
            passed=result.get("prototype_validated", False),
            detail="" if result.get("prototype_validated") else "prototype_validated 未设置为 True",
        ))

        stdout = result.get("stdout_tail", "")
        checks.append(CheckItem(
            name="无明显 Python 异常输出",
            passed="Traceback" not in stdout and "Error" not in stdout,
            detail=f"输出中包含异常关键词: {stdout[:200]}" if ("Traceback" in stdout or "Error" in stdout) else "",
        ))

        return checks

    # ── step_05: 代码重构 ─────────────────────────

    @staticmethod
    def _rubric_step05(state: StateStore) -> list[CheckItem]:
        result = state.get_step_result("step_05") or {}
        checks = []

        # 检查产物文件存在
        refactor_path = state.get("refactor_py_path", "")
        exists = Path(refactor_path).exists() if refactor_path else False
        checks.append(CheckItem(
            name="single_inference_refactor.py 文件已生成",
            passed=exists,
            detail=f"文件不存在: {refactor_path}" if not exists else "",
        ))

        # 检查四个函数都存在
        if exists:
            code = Path(refactor_path).read_text(encoding="utf-8")
            required_fns = ["def init_model", "def pre_process", "def process", "def post_process"]
            missing = [fn for fn in required_fns if fn not in code]
            checks.append(CheckItem(
                name="四个标准函数均已定义",
                passed=len(missing) == 0,
                detail=f"缺少函数: {missing}" if missing else "",
            ))

            # 检查 post_process 返回 dict（简单启发式）
            checks.append(CheckItem(
                name="post_process 有 return 语句",
                passed="def post_process" in code and "return" in code.split("def post_process")[1][:500],
                detail="post_process 函数体内未找到 return 语句",
            ))

        # 检查重构后脚本可运行
        checks.append(CheckItem(
            name="重构代码运行验证通过",
            passed=result.get("prototype_validated", False),
            detail="" if result.get("prototype_validated") else "运行验证失败，见 stdout_tail",
        ))

        return checks

    # ── step_06: JSON 样例 ────────────────────────

    @staticmethod
    def _rubric_step06(state: StateStore) -> list[CheckItem]:
        checks = []

        for name, key in [("request.json", "request_json_path"), ("response.json", "response_json_path")]:
            path = state.get(key, "")
            exists = Path(path).exists() if path else False
            checks.append(CheckItem(
                name=f"{name} 文件已生成",
                passed=exists,
                detail=f"文件不存在: {path}" if not exists else "",
            ))

            if exists:
                try:
                    data = json.loads(Path(path).read_text(encoding="utf-8"))
                    checks.append(CheckItem(
                        name=f"{name} 为合法 JSON",
                        passed=True,
                    ))
                    # request.json 必须包含 requestId 和 body
                    if name == "request.json":
                        has_fields = "requestId" in data and "body" in data
                        checks.append(CheckItem(
                            name="request.json 包含 requestId 和 body 字段",
                            passed=has_fields,
                            detail=f"缺少字段，实际键: {list(data.keys())}" if not has_fields else "",
                        ))
                    # response.json 必须包含 errorCode
                    if name == "response.json":
                        has_code = "errorCode" in data
                        checks.append(CheckItem(
                            name="response.json 包含 errorCode 字段",
                            passed=has_code,
                            detail=f"缺少 errorCode，实际键: {list(data.keys())}" if not has_code else "",
                        ))
                except json.JSONDecodeError as e:
                    checks.append(CheckItem(
                        name=f"{name} 为合法 JSON",
                        passed=False,
                        detail=f"JSON 解析失败: {e}",
                    ))

        return checks

    # ── step_07: server_refactor.py ──────────────

    @staticmethod
    def _rubric_step07(state: StateStore) -> list[CheckItem]:
        checks = []
        server_path = state.get("server_refactor_path", "")
        exists = Path(server_path).exists() if server_path else False

        checks.append(CheckItem(
            name="server_refactor.py 文件已生成",
            passed=exists,
            detail=f"文件不存在: {server_path}" if not exists else "",
        ))

        if exists:
            code = Path(server_path).read_text(encoding="utf-8")

            checks.append(CheckItem(
                name="使用 FastAPI 框架",
                passed="fastapi" in code.lower() or "FastAPI" in code,
                detail="未找到 FastAPI 导入",
            ))

            checks.append(CheckItem(
                name="包含推理接口路由 /infer",
                passed="/infer" in code,
                detail="未找到 /infer 路由定义",
            ))

            checks.append(CheckItem(
                name="包含 init_model 调用",
                passed="init_model" in code,
                detail="未找到 init_model 调用",
            ))

            # 语法检查
            import ast
            try:
                ast.parse(code)
                checks.append(CheckItem(name="Python 语法合法", passed=True))
            except SyntaxError as e:
                checks.append(CheckItem(
                    name="Python 语法合法",
                    passed=False,
                    detail=f"语法错误: {e}",
                ))

        return checks

    # ── step_08: 冒烟测试 ─────────────────────────

    @staticmethod
    def _rubric_step08(state: StateStore) -> list[CheckItem]:
        result = state.get_step_result("step_08") or {}
        checks = []

        checks.append(CheckItem(
            name="冒烟测试通过",
            passed=result.get("smoke_test_passed", False),
            detail="" if result.get("smoke_test_passed") else "smoke_test_passed 未设置为 True",
        ))

        output = result.get("smoke_test_output", "")
        if output:
            try:
                resp = json.loads(output) if output.strip().startswith("{") else {}
                error_code = resp.get("errorCode", -1)
                checks.append(CheckItem(
                    name="响应 errorCode 为 200",
                    passed=error_code == 200,
                    detail=f"实际 errorCode: {error_code}" if error_code != 200 else "",
                ))
            except Exception:
                checks.append(CheckItem(
                    name="响应为合法 JSON",
                    passed=False,
                    detail=f"响应非 JSON 格式: {output[:200]}",
                ))

        return checks

    # ── step_09: 精度测试重构 ─────────────────────

    @staticmethod
    def _rubric_step09(state: StateStore) -> list[CheckItem]:
        result = state.get_step_result("step_09") or {}
        checks = []

        skipped = result.get("precision_test_skipped", False)
        if skipped:
            checks.append(CheckItem(
                name="精度测试（跳过：无 val_precision.py）",
                passed=True,
                detail="val_precision.py 不存在，跳过精度测试，视为通过",
            ))
            return checks

        precision_info = result.get("server_precision", [])
        checks.append(CheckItem(
            name="精度信息成功提取",
            passed=len(precision_info) > 0,
            detail="precision_info 为空，LLM 未能提取精度指标" if not precision_info else f"提取到 {len(precision_info)} 条精度指标",
        ))

        project_dir = Path(state.get("project_dir", ""))
        refactor_script = project_dir / "val_precision_refactor.py"
        checks.append(CheckItem(
            name="val_precision_refactor.py 已生成",
            passed=refactor_script.exists(),
            detail=f"文件不存在: {refactor_script}" if not refactor_script.exists() else "",
        ))

        return checks

    # ── step_11: Docker 脚本生成 ──────────────────

    @staticmethod
    def _rubric_step11(state: StateStore) -> list[CheckItem]:
        result = state.get_step_result("step_11") or {}
        checks = []

        docker_scripts = result.get("docker_scripts", {})
        expected = ["run_load_image.sh", "run_create_image.sh", "run_start_server.sh", "run_stop_server.sh"]
        for script_name in expected:
            path = docker_scripts.get(script_name, "")
            exists = Path(path).exists() if path else False
            checks.append(CheckItem(
                name=f"{script_name} 已生成",
                passed=exists,
                detail=f"文件不存在: {path}" if not exists else "",
            ))
            if exists:
                content = Path(path).read_text(encoding="utf-8")
                checks.append(CheckItem(
                    name=f"{script_name} 包含 #!/bin/bash",
                    passed=content.startswith("#!/bin/bash") or "#!/bin/bash" in content[:50],
                    detail="脚本缺少 shebang 行",
                ))

        return checks

    # ── step_13: 接口文档 ─────────────────────────

    @staticmethod
    def _rubric_step13(state: StateStore) -> list[CheckItem]:
        result = state.get_step_result("step_13") or {}
        checks = []

        doc_path = result.get("api_doc_path", state.get("api_doc_path", ""))
        exists = Path(doc_path).exists() if doc_path else False
        checks.append(CheckItem(
            name="接口文档文件已生成",
            passed=exists,
            detail=f"文件不存在: {doc_path}" if not exists else "",
        ))

        if exists:
            content = Path(doc_path).read_text(encoding="utf-8")
            min_length = 200
            checks.append(CheckItem(
                name=f"文档内容不少于 {min_length} 字符",
                passed=len(content) >= min_length,
                detail=f"文档过短: {len(content)} 字符",
            ))

            for keyword in ["requestId", "errorCode"]:
                checks.append(CheckItem(
                    name=f"文档包含字段说明「{keyword}」",
                    passed=keyword in content,
                    detail=f"文档中未找到 {keyword}",
                ))

        return checks


# ─────────────────────────────────────────────
#  Critic Agent 主体
# ─────────────────────────────────────────────

class CriticAgent:
    """
    Critic Agent：对每个步骤的执行结果进行自动评审。

    使用方式（在 Orchestrator 中）：
        critic = CriticAgent(llm, state, max_revisions=2)
        verdict = critic.review(step_id)
        if verdict.decision == Decision.PASS:
            ...
        elif verdict.decision == Decision.REVISE:
            # 将 revision_plan 写入 state，然后重新执行该步骤
            ...
        else:  # ESCALATE
            # 暂停，等待人工
            ...
    """

    def __init__(
        self,
        llm: LLMClient,
        state: StateStore,
        max_revisions: int = 2,
        use_llm_for_borderline: bool = True,
    ):
        """
        Parameters
        ----------
        llm                   : LLMClient 实例（与其他 Agent 共用）
        state                 : StateStore 实例（共用）
        max_revisions         : 单步最大 Critic 修改轮数（超出则 ESCALATE）
        use_llm_for_borderline: 规则检查不确定时，是否再用 LLM 深度分析
        """
        self.llm = llm
        self.state = state
        self.max_revisions = max_revisions
        self.use_llm_for_borderline = use_llm_for_borderline
        self._revision_counts: dict[str, int] = {}  # step_id -> 已修改轮数

    # ── 主入口 ────────────────────────────────────

    def review(self, step_id: str) -> CriticVerdict:
        """
        对 step_id 的执行结果进行评审，返回 CriticVerdict。
        若该步骤没有配置 Rubric，则直接返回 PASS（向后兼容）。
        """
        rubric_fn = RubricLibrary.for_step(step_id)
        if rubric_fn is None:
            logger.debug(f"  [Critic] {step_id} 无评审 Rubric，自动通过")
            return CriticVerdict(
                step_id=step_id,
                decision=Decision.PASS,
                score=100.0,
                summary="无 Rubric 配置，自动通过",
            )

        # 执行规则检查
        logger.info(f"  [Critic] 开始评审 {step_id}")
        checks = rubric_fn(self.state)
        passed_count = sum(1 for c in checks if c.passed)
        total_count = len(checks)
        score = (passed_count / total_count * 100) if total_count > 0 else 100.0

        failed_checks = [c for c in checks if not c.passed]
        round_number = self._revision_counts.get(step_id, 0) + 1

        # 打印评审结果
        self._log_checks(step_id, checks, score)

        # 所有检查通过
        if not failed_checks:
            logger.info(f"  [Critic] {step_id} 评审通过 (score={score:.0f})")
            return CriticVerdict(
                step_id=step_id,
                decision=Decision.PASS,
                score=score,
                checks=checks,
                summary=f"全部 {total_count} 项检查通过",
                review_round=round_number,
            )

        # 存在失败项：判断是修改还是升级
        # current_revisions 表示「已经执行过的修改轮数」
        # 首次执行 = 0，第一轮修改后 = 1，...
        # 当已修改次数 >= max_revisions 时，不再继续修改，升级人工
        current_revisions = self._revision_counts.get(step_id, 0)
        if current_revisions >= self.max_revisions:
            logger.warning(
                f"  [Critic] {step_id} 已达最大修改轮数 ({self.max_revisions})，升级人工介入"
            )
            return CriticVerdict(
                step_id=step_id,
                decision=Decision.ESCALATE,
                score=score,
                checks=checks,
                summary=f"经过 {current_revisions} 轮修改仍有 {len(failed_checks)} 项未通过，请人工介入",
                review_round=round_number,
            )

        # 生成修改计划，并递增已修改轮数
        revision_plan = self._generate_revision_plan(step_id, failed_checks, round_number)
        self._revision_counts[step_id] = current_revisions + 1  # 先递增再返回

        # 将修改计划写入 StateStore，Phase Agent 重新执行时读取
        self.state.set(f"revision_plan_{step_id}", asdict(revision_plan))

        logger.info(
            f"  [Critic] {step_id} 需要修改 "
            f"(round={round_number}, failed={len(failed_checks)}/{total_count})"
        )

        return CriticVerdict(
            step_id=step_id,
            decision=Decision.REVISE,
            score=score,
            checks=checks,
            summary=f"{len(failed_checks)}/{total_count} 项未通过，已生成修改计划",
            revision_plan=revision_plan,
            review_round=round_number,
        )

    def get_revision_plan(self, step_id: str) -> Optional[RevisionPlan]:
        """Phase Agent 调用：获取针对本步骤的最新修改计划"""
        data = self.state.get(f"revision_plan_{step_id}")
        if not data:
            return None
        return RevisionPlan(
            step_id=data["step_id"],
            failed_checks=data["failed_checks"],
            instructions=data["instructions"],
            context=data.get("context", {}),
            round_number=data.get("round_number", 1),
        )

    def reset_revision_count(self, step_id: str) -> None:
        """Orchestrator 重置步骤时调用（如人工干预后断点恢复）"""
        self._revision_counts.pop(step_id, None)
        self.state.set(f"revision_plan_{step_id}", None)

    # ── 修改计划生成 ──────────────────────────────

    def _generate_revision_plan(
        self, step_id: str, failed_checks: list[CheckItem], round_number: int
    ) -> RevisionPlan:
        """
        根据失败的检查项生成修改计划。
        优先用规则生成指令，可选用 LLM 补充更详细的修改建议。
        """
        instructions = []
        context = {}

        for check in failed_checks:
            # 根据检查项名称映射到具体修改指令
            instruction = self._check_to_instruction(step_id, check)
            if instruction:
                instructions.append(instruction)
            if check.detail:
                context[check.name] = check.detail

        # 若开启 LLM 深度分析且有失败项，调用 LLM 补充修改建议
        if self.use_llm_for_borderline and failed_checks and self.llm._client:
            llm_instructions = self._llm_revision_advice(step_id, failed_checks)
            # 合并（LLM 建议作为补充，不替换规则生成的指令）
            for inst in llm_instructions:
                if inst not in instructions:
                    instructions.append(inst)

        if not instructions:
            instructions = [f"请检查并修复以下问题: {[c.name for c in failed_checks]}"]

        return RevisionPlan(
            step_id=step_id,
            failed_checks=[c.name for c in failed_checks],
            instructions=instructions,
            context=context,
            round_number=round_number,
        )

    @staticmethod
    def _check_to_instruction(step_id: str, check: CheckItem) -> str:
        """将检查项失败转换为具体修改指令（规则映射）"""
        name = check.name
        detail = check.detail

        # step_05 相关
        if "四个标准函数" in name:
            return f"重新生成 single_inference_refactor.py，确保包含 init_model / pre_process / process / post_process 四个函数定义。当前问题: {detail}"
        if "post_process 有 return" in name:
            return "检查 post_process 函数，确保其有明确的 return dict 语句。"
        if "重构代码运行验证" in name:
            return f"修复 single_inference_refactor.py 中导致运行失败的问题。当前错误: {detail}"

        # step_06 相关
        if "合法 JSON" in name:
            return f"重新生成 {name.split()[0]}，确保输出为合法 JSON 格式。当前问题: {detail}"
        if "requestId 和 body" in name:
            return "重新生成 request.json，确保顶层包含 requestId（字符串）和 body（对象）两个字段。"
        if "errorCode" in name and "response" in name.lower():
            return "重新生成 response.json，确保顶层包含 errorCode 字段（正常为 200）。"

        # step_07 相关
        if "FastAPI 框架" in name:
            return "重新生成 server_refactor.py，必须使用 FastAPI 框架（from fastapi import FastAPI）。"
        if "/infer 路由" in name:
            return "在 server_refactor.py 中添加 POST /infer 路由。"
        if "语法合法" in name:
            return f"修复 server_refactor.py 中的语法错误: {detail}"

        # step_08 相关
        if "冒烟测试通过" in name:
            return f"检查 server_refactor.py 的服务逻辑，修复导致冒烟测试失败的问题。"
        if "errorCode 为 200" in name:
            return f"修复服务响应，确保正常请求返回 errorCode=200。当前响应: {detail}"

        # step_11 相关
        if "已生成" in name and ".sh" in name:
            return f"重新生成 {name.split()[0]} 脚本，确保文件被正确写入磁盘。"
        if "#!/bin/bash" in name:
            return f"在 {name.split()[0]} 脚本第一行添加 #!/bin/bash。"

        # step_13 相关
        if "文档内容不少于" in name:
            return "重新生成接口文档，确保内容完整，包含字段说明、请求/响应示例和部署说明。"
        if "字段说明" in name:
            keyword = name.split("「")[1].rstrip("」") if "「" in name else ""
            return f"在接口文档中补充字段 {keyword} 的说明。"

        # 兜底
        return f"修复问题「{name}」: {detail}"

    def _llm_revision_advice(
        self, step_id: str, failed_checks: list[CheckItem]
    ) -> list[str]:
        """
        调用 LLM 对失败检查项给出更详细的修改建议。
        仅在规则映射不足时补充，控制 LLM 调用次数。
        """
        try:
            failed_summary = "\n".join(
                f"- {c.name}: {c.detail}" for c in failed_checks
            )
            system = (
                "你是一位 MLOps 代码审查专家。根据失败的检查项，"
                "给出简洁、可操作的修改建议（每条一行，不超过5条）。"
                "只输出建议列表，每行一条，不要编号，不要解释。"
            )
            user = (
                f"步骤 {step_id} 执行后以下检查项未通过：\n\n"
                f"{failed_summary}\n\n"
                "请给出修改建议："
            )
            raw = self.llm.complete(system, user, max_tokens=512)
            return [line.strip() for line in raw.strip().splitlines() if line.strip()]
        except Exception as e:
            logger.debug(f"  [Critic] LLM 修改建议生成失败（忽略）: {e}")
            return []

    # ── 日志输出 ──────────────────────────────────

    @staticmethod
    def _log_checks(step_id: str, checks: list[CheckItem], score: float) -> None:
        passed = sum(1 for c in checks if c.passed)
        total = len(checks)
        logger.info(f"  [Critic] {step_id} 评审结果: {passed}/{total} 通过，得分 {score:.0f}")
        for check in checks:
            icon = "✓" if check.passed else "✗"
            detail = f" — {check.detail}" if check.detail and not check.passed else ""
            logger.info(f"    {icon} {check.name}{detail}")
