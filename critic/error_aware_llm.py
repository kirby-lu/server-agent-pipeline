"""
critic/error_aware_llm.py
--------------------------
ErrorAwareLLMClient — 透明包装 LLMClient，自动将上一轮错误信息和生成的代码注入 prompt。

设计原则：
- 零侵入：不修改任何现有文件
- 透明代理：所有方法签名与 LLMClient 完全一致，Phase Agent 无感知
- 错误注入：每次 LLM 调用前，自动在 user_prompt 末尾追加上一轮的
             运行时错误（来自 StateStore steps[step_id].last_error）
- 代码注入：每次 LLM 调用前，同时将上一轮生成的代码内容追加到 prompt 中，
             让 LLM 能基于已有代码精准修复，而非从头生成

工作机制：
    Orchestrator._tech_retry 调用 agent.execute_step(step_id) 之前，
    先调用 wrap_agent_llm(agent, state, step_id)，
    将 agent.llm 替换为 ErrorAwareLLMClient；
    execute_step 内部所有 self.llm.xxx() 调用自动命中代理方法；
    执行结束后 unwrap_agent_llm 还原原始 llm。

    代码保存时机：
    _tech_retry 在 agent.execute_step 成功后调用 save_step_generated_code，
    将本轮生成的代码文件路径列表记录到 StateStore，供下一轮注入使用。

注入顺序（追加到 user_prompt 末尾）：
    1. [上一轮生成的代码]  ← 本次新增
    2. [上一轮执行错误]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils.logger import setup_logger, LLMClient
from utils.state_store import StateStore

logger = setup_logger("error_aware_llm")


# ── step_id → 代码文件路径在 StateStore 中的键名映射 ────────────────────────────
# 当某个 step 执行完毕后，其产出的代码文件路径以这些键名写入 StateStore。
# ErrorAwareLLMClient 根据 step_id 查找对应的键，读取文件内容后注入到下一轮 prompt。
#
# 值为 list[str]，支持单步输出多个代码文件（如同时生成 server.py + refactor.py）。
# 若某步骤无代码产出（如仅下载资源），留空列表即可，注入逻辑自动跳过。
_STEP_CODE_STATE_KEYS: dict[str, list[str]] = {
    "step_03": [],                                           # 资源下载，无代码产出
    "step_04": ["refactor_py_path"],                        # 原型验证（重构后脚本）
    "step_05": ["refactor_py_path"],                        # 代码重构
    "step_06": ["request_json_path", "response_json_path"], # JSON 样例
    "step_07": ["server_refactor_path"],                    # server_refactor.py
    "step_08": ["server_refactor_path"],                    # 冒烟测试（代码同上步）
    "step_09": [],                                          # 精度测试（路径在 result 中）
    "step_11": [],                                          # Docker 脚本（多文件由 result 给出）
    "step_13": ["api_doc_path"],                            # 接口文档
}


class ErrorAwareLLMClient:
    """
    LLMClient 的透明代理。

    构造时接收原始 llm 和 state；每次调用 LLM 方法时：
    1. 从 StateStore 取出当前 step_id 的上一轮生成代码（若有）
    2. 从 StateStore 取出当前 step_id 的 last_error（若有）
    3. 将代码和错误信息（按此顺序）追加到 user_prompt 末尾
    4. 其余参数原样传给原始 llm

    属性代理：所有未显式定义的属性都透明转发给 _inner_llm，
    确保 Phase Agent 中 self.llm.model / self.llm._client 等访问正常。
    """

    # ── 注入模板 ─────────────────────────────────────────────────────────────

    _CODE_INJECTION_TEMPLATE = """
        ===== 上一轮生成的代码（请基于此代码进行修复，勿从头重写）=====
        以下是上一轮执行时生成的代码文件内容。请仔细阅读后，
        针对下方「执行错误」部分描述的问题进行精准修复，保留已经正确的部分：

        {code_blocks}
        ====================================="""

    _ERROR_INJECTION_TEMPLATE = """
        ===== 上一轮执行错误（必须修复）=====
        上一轮生成的代码/内容在运行时出现了以下错误，
        请仔细分析根本原因，在本次生成中直接修复，不要添加"待修复"注释：

        {error_info}
        ====================================="""

    _CODE_BLOCK_TEMPLATE = """
        【文件: {file_path}】
            ```{lang}
            {content}
            ```
        """

    def __init__(
        self,
        inner_llm: LLMClient,
        state: StateStore,
        step_id: str,
        max_error_length: int = 3000,
        max_code_length: int = 6000,
        inject_code: bool = True,
    ):
        """
        Parameters
        ----------
        inner_llm        : 被代理的原始 LLMClient
        state            : StateStore 实例（共用）
        step_id          : 当前执行的步骤 ID
        max_error_length : 错误信息最大截断长度（字符数）
        max_code_length  : 代码注入最大截断长度（所有文件合计，字符数）
        inject_code      : 是否启用代码注入（可通过构造参数关闭）
        """
        # 使用 object.__setattr__ 绕过 __setattr__ 代理，避免死递归
        object.__setattr__(self, "_inner_llm", inner_llm)
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_step_id", step_id)
        object.__setattr__(self, "_max_error_len", max_error_length)
        object.__setattr__(self, "_max_code_len", max_code_length)
        object.__setattr__(self, "_inject_code", inject_code)

    # ── 属性透明代理 ──────────────────────────────────

    def __getattr__(self, name: str):
        """未显式定义的属性一律转发给 inner_llm"""
        return getattr(object.__getattribute__(self, "_inner_llm"), name)

    def __setattr__(self, name: str, value):
        """属性写入也转发给 inner_llm"""
        inner = object.__getattribute__(self, "_inner_llm")
        setattr(inner, name, value)

    # ── 三个核心 LLM 方法的代理 ───────────────────────

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.1,
    ) -> str:
        enriched = self._inject_context(user_prompt)
        return object.__getattribute__(self, "_inner_llm").complete(
            system_prompt, enriched, max_tokens, temperature
        )

    def generate_python_code(
        self,
        system_prompt: str,
        user_prompt: str,
        output_path: Optional[Path] = None,
        max_self_correct: int = 3,
    ) -> str:
        enriched = self._inject_context(user_prompt)
        return object.__getattribute__(self, "_inner_llm").generate_python_code(
            system_prompt, enriched, output_path, max_self_correct
        )

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[dict] = None,
        output_path: Optional[Path] = None,
    ) -> dict:
        enriched = self._inject_context(user_prompt)
        return object.__getattribute__(self, "_inner_llm").generate_json(
            system_prompt, enriched, schema, output_path
        )

    # ── 私有：上下文注入（代码 + 错误）────────────────

    def _inject_context(self, user_prompt: str) -> str:
        """
        将上一轮生成的代码和运行错误按序追加到 user_prompt 末尾。

        注入顺序：
          1. 代码块（让 LLM 先了解上一轮写了什么）
          2. 错误信息（再告知哪里出了问题）

        若两者均不存在（首次执行或无错误），原样返回 user_prompt。
        """
        step_id = object.__getattribute__(self, "_step_id")
        inject_code = object.__getattribute__(self, "_inject_code")

        code_section = ""
        error_section = ""

        # 注入上一轮代码
        if inject_code:
            code_content = self._get_last_generated_code()
            if code_content:
                logger.info(f"  [ErrorAwareLLM] 向 {step_id} 注入上一轮生成的代码")
                code_section = self._CODE_INJECTION_TEMPLATE.format(
                    code_blocks=code_content
                )

        # 注入上一轮错误
        error_info = self._get_last_error()
        if error_info:
            logger.info(f"  [ErrorAwareLLM] 向 {step_id} 注入上一轮错误信息")
            error_section = self._ERROR_INJECTION_TEMPLATE.format(
                error_info=error_info
            )

        if not code_section and not error_section:
            return user_prompt

        return user_prompt + code_section + error_section

    def _get_last_generated_code(self) -> str:
        """
        从 StateStore 读取上一轮该步骤生成的代码文件内容。

        查找策略（按优先级）：
          1. state.steps[step_id].generated_code_paths —— 由 _tech_retry 在成功后写入，
             包含本轮产出的所有代码文件路径列表
          2. _STEP_CODE_STATE_KEYS[step_id] 中定义的 StateStore 顶层键 ——
             Phase Agent 执行完将路径写入 state 顶层（如 refactor_py_path）
          3. step 的 result 字典中的路径型字段（兜底）

        读取后将所有文件内容拼接成带文件名标注的代码块字符串，并按长度截断。
        """
        state   = object.__getattribute__(self, "_state")
        step_id = object.__getattribute__(self, "_step_id")
        max_len = object.__getattribute__(self, "_max_code_len")

        file_paths: list[str] = []

        # 优先级1：由 orchestrator 主动记录的路径列表
        step_data = state.get("steps", {}).get(step_id, {})
        recorded_paths = step_data.get("generated_code_paths", [])
        if recorded_paths:
            file_paths = list(recorded_paths)

        # 优先级2：从 _STEP_CODE_STATE_KEYS 映射中查找 state 顶层键
        if not file_paths:
            state_keys = _STEP_CODE_STATE_KEYS.get(step_id, [])
            for key in state_keys:
                path_val = state.get(key, "")
                if path_val and isinstance(path_val, str):
                    file_paths.append(path_val)

        # 优先级3：从 step result 中扫描路径型字段（兜底）
        if not file_paths:
            result = step_data.get("result") or {}
            if isinstance(result, dict):
                for val in result.values():
                    if isinstance(val, str) and val.endswith((".py", ".sh", ".json", ".md")):
                        file_paths.append(val)

        if not file_paths:
            return ""

        # 读取文件内容并拼接
        blocks: list[str] = []
        total_len = 0

        for file_path in file_paths:
            p = Path(file_path)
            if not p.exists():
                logger.debug(f"  [ErrorAwareLLM] 代码文件不存在，跳过: {file_path}")
                continue
            try:
                content = p.read_text(encoding="utf-8")
            except Exception as e:
                logger.debug(f"  [ErrorAwareLLM] 读取代码文件失败 {file_path}: {e}")
                continue

            # 超出总长度限制时截断当前文件
            remaining = max_len - total_len
            if remaining <= 0:
                logger.debug(
                    f"  [ErrorAwareLLM] 已达代码注入长度上限 ({max_len})，"
                    f"跳过剩余文件: {file_path}"
                )
                break

            truncated = False
            if len(content) > remaining:
                content = content[:remaining] + f"\n...(已截断，原始长度 {len(content)} 字符)"
                truncated = True

            lang = _infer_lang(p.suffix)
            block = self._CODE_BLOCK_TEMPLATE.format(
                file_path=file_path,
                lang=lang,
                content=content,
            )
            blocks.append(block)
            total_len += len(content)

            if truncated:
                break

        return "\n\n".join(blocks) if blocks else ""

    def _get_last_error(self) -> str:
        """
        从 StateStore 组合错误信息：
          1. steps[step_id].last_error  — 运行时异常堆栈
          2. steps[step_id].result.stderr_tail — stderr 输出
          3. steps[step_id].result.stdout_tail — stdout 中含 Traceback 时
        """
        state   = object.__getattribute__(self, "_state")
        step_id = object.__getattribute__(self, "_step_id")
        max_len = object.__getattribute__(self, "_max_error_len")

        step_data = state.get("steps", {}).get(step_id, {})
        parts: list[str] = []

        last_error = step_data.get("last_error", "")
        if last_error:
            parts.append(f"[运行时异常]\n{last_error}")

        result = step_data.get("result") or {}
        if isinstance(result, dict):
            if result.get("stderr_tail"):
                parts.append(f"[stderr]\n{result['stderr_tail']}")
            stdout = result.get("stdout_tail", "")
            if stdout and ("Traceback" in stdout or "Error" in stdout):
                parts.append(f"[stdout 异常]\n{stdout}")

        if not parts:
            return ""

        combined = "\n\n".join(parts)
        if len(combined) > max_len:
            combined = (
                combined[:max_len]
                + f"\n...(已截断，原始长度 {len(combined)} 字符)"
            )
        return combined


# ── 辅助函数 ─────────────────────────────────────────────────────────────────

def _infer_lang(suffix: str) -> str:
    """根据文件扩展名推断 Markdown 代码块的语言标识"""
    return {
        ".py":   "python",
        ".sh":   "bash",
        ".json": "json",
        ".md":   "markdown",
        ".yaml": "yaml",
        ".yml":  "yaml",
        ".txt":  "text",
    }.get(suffix.lower(), "text")


# ─────────────────────────────────────────────
#  工厂函数：供 Orchestrator._tech_retry 调用
# ─────────────────────────────────────────────

def wrap_agent_llm(
    agent,
    state: StateStore,
    step_id: str,
    inject_code: bool = True,
    max_error_length: int = 3000,
    max_code_length: int = 6000,
) -> LLMClient:
    """
    将 agent.llm 替换为 ErrorAwareLLMClient，返回原始 llm 供事后还原。

    Parameters
    ----------
    agent            : Phase Agent 实例（拥有 .llm 属性）
    state            : StateStore 实例
    step_id          : 当前步骤 ID
    inject_code      : 是否启用代码注入（默认 True）
    max_error_length : 错误信息最大截断长度
    max_code_length  : 代码注入最大截断长度（所有文件合计）

    典型用法（在 _tech_retry 内）：
        original_llm = wrap_agent_llm(agent, self.state, step_id)
        try:
            result = agent.execute_step(step_id)
        finally:
            unwrap_agent_llm(agent, original_llm)
    """
    original_llm = agent.llm
    agent.llm = ErrorAwareLLMClient(
        inner_llm=original_llm,
        state=state,
        step_id=step_id,
        max_error_length=max_error_length,
        max_code_length=max_code_length,
        inject_code=inject_code,
    )
    logger.debug(
        f"  [ErrorAwareLLM] 已为 {type(agent).__name__} 安装"
        f" ErrorAwareLLMClient (step={step_id}, inject_code={inject_code})"
    )
    return original_llm


def unwrap_agent_llm(agent, original_llm: LLMClient) -> None:
    """还原 agent.llm 为原始 LLMClient 实例"""
    agent.llm = original_llm
    logger.debug(
        f"  [ErrorAwareLLM] 已还原 {type(agent).__name__} 的原始 LLM"
    )


def save_step_generated_code(
    state: StateStore,
    step_id: str,
    code_file_paths: list[str],
) -> None:
    """
    将本轮生成的代码文件路径列表写入 StateStore，
    供下一轮 ErrorAwareLLMClient 注入到 prompt 中。

    应在 _tech_retry 执行成功后立即调用：
        result = agent.execute_step(step_id)
        save_step_generated_code(state, step_id, [...])

    Parameters
    ----------
    state            : StateStore 实例
    step_id          : 步骤 ID
    code_file_paths  : 本轮生成的代码文件路径列表（绝对路径字符串）
    """
    # 过滤掉不存在的路径，避免注入无效内容
    valid_paths = [p for p in code_file_paths if Path(p).exists()]
    if not valid_paths:
        return

    # 写入 steps[step_id].generated_code_paths
    steps = state.get("steps", {})
    if step_id not in steps:
        steps[step_id] = {}
    steps[step_id]["generated_code_paths"] = valid_paths
    state.set("steps", steps)

    logger.debug(
        f"  [ErrorAwareLLM] {step_id} 已记录 {len(valid_paths)} 个代码文件路径: "
        + ", ".join(valid_paths)
    )


def collect_code_paths_for_step(state: StateStore, step_id: str) -> list[str]:
    """
    根据 _STEP_CODE_STATE_KEYS 映射，从 StateStore 中收集本步骤产出的代码文件路径。

    供 orchestrator_with_critic._tech_retry 在执行成功后调用，
    自动汇总路径后交给 save_step_generated_code 保存。

    兜底策略：同时扫描 step result 字典中的路径型字段。
    """
    paths: list[str] = []

    # 从顶层 state 键收集
    state_keys = _STEP_CODE_STATE_KEYS.get(step_id, [])
    for key in state_keys:
        val = state.get(key, "")
        if val and isinstance(val, str) and Path(val).exists():
            paths.append(val)

    # 从 step result 兜底收集
    result = state.get_step_result(step_id) or {}
    if isinstance(result, dict):
        for val in result.values():
            if (
                isinstance(val, str)
                and val.endswith((".py", ".sh", ".json", ".md"))
                and Path(val).exists()
                and val not in paths
            ):
                paths.append(val)
        # docker_scripts 是 dict[name, path]
        docker_scripts = result.get("docker_scripts", {})
        if isinstance(docker_scripts, dict):
            for path_val in docker_scripts.values():
                if (
                    isinstance(path_val, str)
                    and Path(path_val).exists()
                    and path_val not in paths
                ):
                    paths.append(path_val)

    return paths
