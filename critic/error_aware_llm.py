"""
critic/error_aware_llm.py
--------------------------
ErrorAwareLLMClient — 透明包装 LLMClient，自动将上一轮错误注入 prompt。

设计原则：
- 零侵入：不修改任何现有文件
- 透明代理：所有方法签名与 LLMClient 完全一致，Phase Agent 无感知
- 错误注入：每次 LLM 调用前，自动在 user_prompt 末尾追加上一轮的
             运行时错误（来自 StateStore steps[step_id].last_error）

工作机制：
    Orchestrator._tech_retry 调用 agent.execute_step(step_id) 之前，
    先调用 wrap_agent_llm(agent, state, step_id)，
    将 agent.llm 替换为 ErrorAwareLLMClient；
    execute_step 内部所有 self.llm.xxx() 调用自动命中代理方法；
    执行结束后 unwrap_agent_llm 还原原始 llm。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils.logger import setup_logger, LLMClient
from utils.state_store import StateStore

logger = setup_logger("error_aware_llm")


class ErrorAwareLLMClient:
    """
    LLMClient 的透明代理。

    构造时接收原始 llm 和 state；每次调用 LLM 方法时：
    1. 从 StateStore 取出当前 step_id 的 last_error
    2. 若存在错误，将其格式化追加到 user_prompt 末尾
    3. 其余参数原样传给原始 llm

    属性代理：所有未显式定义的属性都透明转发给 _inner_llm，
    确保 Phase Agent 中 self.llm.model / self.llm._client 等访问正常。
    """

    _ERROR_INJECTION_TEMPLATE = """\


===== 上一轮执行错误（必须修复）=====
上一轮生成的代码/内容在运行时出现了以下错误，
请仔细分析根本原因，在本次生成中直接修复，不要添加"待修复"注释：

{error_info}
====================================="""

    def __init__(
        self,
        inner_llm: LLMClient,
        state: StateStore,
        step_id: str,
        max_error_length: int = 3000,
    ):
        # 使用 object.__setattr__ 绕过 __setattr__ 代理，避免死递归
        object.__setattr__(self, "_inner_llm", inner_llm)
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_step_id", step_id)
        object.__setattr__(self, "_max_error_len", max_error_length)

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
        enriched = self._inject_error(user_prompt)
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
        enriched = self._inject_error(user_prompt)
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
        enriched = self._inject_error(user_prompt)
        return object.__getattribute__(self, "_inner_llm").generate_json(
            system_prompt, enriched, schema, output_path
        )

    # ── 私有：错误提取与注入 ──────────────────────────

    def _inject_error(self, user_prompt: str) -> str:
        """读取上轮错误并追加到 prompt；无错误时原样返回"""
        error_info = self._get_last_error()
        if not error_info:
            return user_prompt
        step_id = object.__getattribute__(self, "_step_id")
        logger.info(f"  [ErrorAwareLLM] 向 {step_id} 注入上一轮错误信息")
        return user_prompt + self._ERROR_INJECTION_TEMPLATE.format(
            error_info=error_info
        )

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


# ─────────────────────────────────────────────
#  工厂函数：供 Orchestrator._tech_retry 调用
# ─────────────────────────────────────────────

def wrap_agent_llm(agent, state: StateStore, step_id: str) -> LLMClient:
    """
    将 agent.llm 替换为 ErrorAwareLLMClient，返回原始 llm 供事后还原。

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
    )
    logger.debug(
        f"  [ErrorAwareLLM] 已为 {type(agent).__name__} 安装"
        f" ErrorAwareLLMClient (step={step_id})"
    )
    return original_llm


def unwrap_agent_llm(agent, original_llm: LLMClient) -> None:
    """还原 agent.llm 为原始 LLMClient 实例"""
    agent.llm = original_llm
    logger.debug(
        f"  [ErrorAwareLLM] 已还原 {type(agent).__name__} 的原始 LLM"
    )
