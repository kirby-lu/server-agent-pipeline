"""工具模块：日志配置 + LLM 客户端封装"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


# ─────────────────────────────────────────────
#  日志
# ─────────────────────────────────────────────

# ANSI 颜色代码
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
# 前景色
COLOR_BLACK = "\033[30m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_MAGENTA = "\033[35m"
COLOR_CYAN = "\033[36m"
COLOR_WHITE = "\033[37m"
# 亮色
COLOR_BRIGHT_RED = "\033[91m"
COLOR_BRIGHT_GREEN = "\033[92m"
COLOR_BRIGHT_YELLOW = "\033[93m"
COLOR_BRIGHT_BLUE = "\033[94m"
COLOR_BRIGHT_MAGENTA = "\033[95m"
COLOR_BRIGHT_CYAN = "\033[96m"
COLOR_BRIGHT_WHITE = "\033[97m"
COLOR_256_TEAL = "\033[38;5;30m"      # 蓝绿/鸭绿
COLOR_256_ORANGE = "\033[38;5;208m"   # 橙色
COLOR_256_PURPLE = "\033[38;5;129m"   # 紫色
COLOR_256_GREY = "\033[38;5;244m"
COLOR_256_LAVENDER = "\033[38;5;146m"

# Logger 名称到颜色的映射
LOG_COLORS = {
    # Phase agents
    "phase1_env": COLOR_256_TEAL,   # 蓝绿/鸭绿
    "phase2_service": COLOR_BRIGHT_GREEN, # 绿色
    "phase3_eval": COLOR_CYAN,            # 青色（暗）
    "phase4_docker": COLOR_256_ORANGE,   # 橙色
    # LLM 相关
    "llm_client": COLOR_BLUE,        # 蓝色
    "error_aware_llm": COLOR_BRIGHT_YELLOW, # 黄色
    # 协调器
    "orchestrator": COLOR_BRIGHT_BLUE,      # 蓝色 
    "orchestrator_critic": COLOR_MAGENTA,   # 洋红
    "critic_agent": COLOR_256_PURPLE,      # 紫色
    # 工具类
    "shell_executor": COLOR_256_GREY,       # 暗草色
    "service_utils": COLOR_256_LAVENDER,    # 浅紫灰
    # 默认颜色
    "default": COLOR_WHITE,
}

# 日志级别颜色
LEVEL_COLORS = {
    logging.DEBUG: COLOR_BLUE,
    logging.INFO: COLOR_GREEN,
    logging.WARNING: COLOR_YELLOW,
    logging.ERROR: COLOR_RED,
    logging.CRITICAL: COLOR_BRIGHT_RED,
}


class ColorFormatter(logging.Formatter):
    """带颜色的控制台日志格式化器"""

    def format(self, record):
        # 获取 logger 名称对应的颜色
        name_color = LOG_COLORS.get(record.name, LOG_COLORS["default"])
        # 获取日志级别颜色
        level_color = LEVEL_COLORS.get(record.levelno, COLOR_WHITE)

        # 创建带颜色的格式字符串
        fmt = (
            f"{COLOR_BOLD}{COLOR_WHITE}%(asctime)s{COLOR_RESET}  "
            f"{name_color}%(name)-20s{COLOR_RESET}  "
            f"{level_color}%(levelname)-8s{COLOR_RESET}  "
            f"{COLOR_WHITE}%(message)s{COLOR_RESET}"
        )

        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    # 控制台 handler，根据 NO_COLOR 环境变量决定是否带颜色
    console_handler = logging.StreamHandler(sys.stdout)
    if os.environ.get("NO_COLOR"):
        # 用户明确要求无颜色
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S"
        ))
    else:
        # 默认使用颜色
        console_handler.setFormatter(ColorFormatter())
    logger.addHandler(console_handler)

    # 同时写文件（无颜色）
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(name)s  %(levelname)s  %(message)s"
    ))
    logger.addHandler(file_handler)
    return logger


# ─────────────────────────────────────────────
#  LLM 客户端（封装 Anthropic API）
# ─────────────────────────────────────────────

class LLMClient:
    """
    对 Anthropic Claude API 的轻量封装。
    支持：普通补全 / 代码生成（带语法验证） / JSON 生成（带 Schema 验证）
    """

    def __init__(self, model: str = "deepseek-chat", max_retries: int = 3):
        self.model = model
        self.max_retries = max_retries
        self.logger = setup_logger("llm_client")
        self._client = self._init_client()

    def _init_client(self):
        try:
            import anthropic
            return anthropic.Anthropic()
        except ImportError:
            self.logger.warning("anthropic 包未安装，LLM 调用将使用 Mock 模式")
            return None

    # ── 通用补全 ──────────────────────────────────

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.1,
    ) -> str:
        """调用 LLM 返回纯文本"""
        if self._client is None:
            return self._mock_response(user_prompt)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return response.content[0].text
            except Exception as e:
                self.logger.warning(f"LLM 请求失败 (第{attempt}次): {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                else:
                    raise

    # ── 代码生成（带自我修正） ─────────────────────

    def generate_python_code(
        self,
        system_prompt: str,
        user_prompt: str,
        output_path: Optional[Path] = None,
        max_self_correct: int = 3,
    ) -> str:
        """
        生成 Python 代码，并用 py_compile 验证语法。
        如果语法错误则将错误反馈给 LLM 自我修正（最多 max_self_correct 轮）。
        """
        messages = [{"role": "user", "content": user_prompt}]
        error_context = ""

        for round_ in range(1, max_self_correct + 1):
            if error_context:
                messages.append({
                    "role": "user",
                    "content": (
                        f"上次生成的代码有语法错误，请修正：\n\n```\n{error_context}\n```\n"
                        "请只输出修正后的完整 Python 代码，不要有任何解释文字。"
                    )
                })

            raw = self._client.messages.create(
                model=self.model,
                max_tokens=8192,
                temperature=0.1,
                system=system_prompt,
                messages=messages,
            ).content[0].text if self._client else self._mock_python(user_prompt)

            code = self._extract_code_block(raw, lang="python")

            # 语法验证
            syntax_error = self._check_python_syntax(code)
            if syntax_error is None:
                self.logger.info(f"  代码生成成功（第{round_}轮）")
                if output_path:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(code, encoding="utf-8")
                return code
            else:
                self.logger.warning(f"  语法错误（第{round_}轮）: {syntax_error}")
                error_context = syntax_error
                messages.append({"role": "assistant", "content": raw})

        raise RuntimeError(f"LLM 代码生成失败，{max_self_correct} 轮后仍有语法错误")

    # ── JSON 生成（带 Schema 验证） ────────────────

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[dict] = None,
        output_path: Optional[Path] = None,
    ) -> dict:
        """
        生成 JSON，自动提取代码块并验证。
        若提供 schema 则用 jsonschema 验证结构。
        """
        raw = self.complete(
            system_prompt + "\n\n重要：只输出纯 JSON，不要 Markdown 代码块，不要任何解释。",
            user_prompt,
        )

        # 尝试提取 JSON
        data = self._extract_json(raw)

        if schema is not None:
            try:
                import jsonschema
                jsonschema.validate(data, schema)
            except ImportError:
                pass  # jsonschema 未安装时跳过验证
            except Exception as e:
                raise ValueError(f"JSON Schema 验证失败: {e}\n原始内容: {raw[:500]}")

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        return data

    # ── 内部工具 ──────────────────────────────────

    @staticmethod
    def _extract_code_block(text: str, lang: str = "python") -> str:
        """从 LLM 输出中提取代码块，若无则返回整段文本"""
        pattern = rf"```{lang}\s*([\s\S]*?)```"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # 兜底：去除所有 markdown
        return re.sub(r"```[^\n]*\n?", "", text).strip()

    @staticmethod
    def _extract_json(text: str) -> dict:
        """从文本中提取 JSON 对象"""
        # 先尝试直接解析
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        # 提取 ```json ... ``` 块
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        # 找第一个 { ... }
        match = re.search(r"(\{[\s\S]*\})", text)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"无法从 LLM 输出中提取 JSON: {text[:300]}")

    @staticmethod
    def _check_python_syntax(code: str) -> Optional[str]:
        """返回语法错误字符串，无错误返回 None"""
        import ast
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return str(e)

    @staticmethod
    def _mock_response(prompt: str) -> str:
        return f"[MOCK] 响应: {prompt[:80]}..."

    @staticmethod
    def _mock_python(prompt: str) -> str:
        return "# Mock Python code\nprint('hello world')\n"
