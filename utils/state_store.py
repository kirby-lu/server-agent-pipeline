"""
StateStore — 持久化状态存储
每步执行状态、产出物路径、错误日志都写入磁盘 JSON 文件，支持断点续跑。
"""

from __future__ import annotations

import json
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class StepStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    SUCCESS  = "success"
    FAILED   = "failed"
    SKIPPED  = "skipped"


class StateStore:
    """线程安全的 JSON 持久化状态存储"""

    def __init__(self, state_file: Path):
        self._file = state_file
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._load()

    # ── 读写接口 ──────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._persist()

    def update(self, mapping: dict[str, Any]) -> None:
        with self._lock:
            self._data.update(mapping)
            self._persist()

    # ── 步骤状态 ──────────────────────────────────

    def get_step_status(self, step_id: str) -> StepStatus:
        steps = self._data.get("steps", {})
        return StepStatus(steps.get(step_id, {}).get("status", StepStatus.PENDING))

    def set_step_status(self, step_id: str, status: StepStatus) -> None:
        with self._lock:
            if "steps" not in self._data:
                self._data["steps"] = {}
            if step_id not in self._data["steps"]:
                self._data["steps"][step_id] = {}
            self._data["steps"][step_id]["status"] = status.value
            if status == StepStatus.RUNNING:
                import time
                self._data["steps"][step_id]["started_at"] = time.time()
            elif status in (StepStatus.SUCCESS, StepStatus.FAILED):
                import time
                self._data["steps"][step_id]["finished_at"] = time.time()
            self._persist()

    def increment_retry(self, step_id: str) -> int:
        with self._lock:
            if "steps" not in self._data:
                self._data["steps"] = {}
            if step_id not in self._data["steps"]:
                self._data["steps"][step_id] = {}
            count = self._data["steps"][step_id].get("retries", 0) + 1
            self._data["steps"][step_id]["retries"] = count
            self._persist()
            return count

    def save_step_result(self, step_id: str, result: dict) -> None:
        with self._lock:
            if "steps" not in self._data:
                self._data["steps"] = {}
            if step_id not in self._data["steps"]:
                self._data["steps"][step_id] = {}
            self._data["steps"][step_id]["result"] = result
            # 将结果中的键也提升到顶层，方便跨步骤访问
            if isinstance(result, dict):
                self._data.update(result)
            self._persist()

    def save_step_error(self, step_id: str, error: str) -> None:
        with self._lock:
            if "steps" not in self._data:
                self._data["steps"] = {}
            if step_id not in self._data["steps"]:
                self._data["steps"][step_id] = {}
            self._data["steps"][step_id]["last_error"] = error
            self._persist()

    def get_step_result(self, step_id: str) -> Optional[dict]:
        steps = self._data.get("steps", {})
        return steps.get(step_id, {}).get("result")

    # ── Pipeline 状态 ─────────────────────────────

    def set_pipeline_status(self, status) -> None:
        self.set("pipeline_status", status.value if hasattr(status, "value") else status)

    def get_pipeline_status(self) -> str:
        return self._data.get("pipeline_status", "pending")

    # ── 快捷方法（项目上下文） ─────────────────────

    def set_project_dir(self, path: str) -> None:
        self.set("project_dir", path)

    def get_project_dir(self) -> str:
        return self._data.get("project_dir", "")

    def set_venv_python(self, path: str) -> None:
        self.set("venv_python", path)

    def get_venv_python(self) -> str:
        return self._data.get("venv_python", "python")

    # ── 内部 ──────────────────────────────────────

    def _load(self) -> dict:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _persist(self) -> None:
        """调用前必须持有 self._lock"""
        self._file.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def dump(self) -> dict:
        with self._lock:
            return dict(self._data)
