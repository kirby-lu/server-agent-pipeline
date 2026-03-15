"""
Phase 2 Sub-Agent — 服务生成
步骤 5-8：代码重构 → 接口样例 → FastAPI 服务 → 冒烟测试
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tools.shell_executor import ShellExecutor
from utils.logger import setup_logger, LLMClient
from utils.state_store import StateStore

logger = setup_logger("phase2_service")


# ─────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────

REFACTOR_SYSTEM_PROMPT = """你是一位资深 MLOps 工程师，专注于将机器学习推理代码标准化为生产可用的模块。
你的任务是将 single_inference.py 中的逻辑严格重构为四个标准函数，保持原有逻辑不变，只做结构拆分。
输出完整的 Python 文件，不要有任何额外解释。"""

REFACTOR_USER_TEMPLATE = """将以下 single_inference.py 重构为包含四个标准函数的 single_inference_refactor.py。

四个函数规范：
1. `init_model() -> Any`
   - 加载模型、初始化权重、设置设备（CPU/GPU）
   - 只执行一次，返回 model 对象或 model 相关的所有对象

2. `pre_process(raw_input: dict) -> Any`
   - 接收原始 HTTP 请求 dict，执行数据预处理（resize/normalize/tokenize 等）
   - 返回处理后的 tensor/array

3. `process(model: Any, processed_input: Any) -> Any`
   - 执行模型推理
   - 返回原始推理结果

4. `post_process(raw_output: Any) -> dict`
   - 将推理结果转换为可序列化的 dict（用于 JSON 响应）
   - 必须返回 dict 类型

原始代码：
```python
{original_code}
```

重构要求：
- 保留所有 import 语句
- 在文件顶部添加全局 model 变量
- 如有必要添加类型注解
- 函数签名必须与规范完全一致
- 输出完整可运行的 Python 文件"""

JSON_SYSTEM_PROMPT = """你是 API 设计专家。
根据推理服务代码，生成符合实际数据类型的请求和响应 JSON 样例。
只输出纯 JSON，不要 Markdown 代码块，不要任何解释。"""

SERVER_SYSTEM_PROMPT = """你是 FastAPI 专家，将推理函数封装为生产级 HTTP 服务。
输出完整的 server_refactor.py 文件，不要有任何额外解释。"""

SERVER_USER_TEMPLATE = """将 single_inference_refactor.py 的四个函数封装为 FastAPI 服务。

single_inference_refactor.py 内容：
```python
{refactor_code}
```

server.py 模板（参考结构）：
```python
{server_template}
```

生成 server_refactor.py 要求：
1. 导入 single_inference_refactor 的四个函数
2. 应用启动时调用 init_model()，结果存在全局变量
3. POST /predict 接口：
   - 接收 JSON body（参考 request_sample）
   - 调用 pre_process → process → post_process
   - 返回 JSON response
4. GET /health 健康检查接口
5. GET /metrics 返回简单的请求统计（调用次数、平均延迟）
6. 完整的异常处理和日志记录
7. 使用 uvicorn 启动，端口从环境变量 SERVER_PORT 读取（默认 {port}）

request 样例：
```json
{request_json}
```

输出完整的 server_refactor.py 文件。"""

SMOKE_TEST_SYSTEM_PROMPT = """你是 QA 工程师，专门编写 HTTP 服务的冒烟测试脚本。
输出完整的 Python 测试脚本，不要有任何额外解释。"""

SMOKE_TEST_USER_TEMPLATE = """基于以下信息，生成冒烟测试脚本 smoke_test.py。

服务地址: {server_url}
request.json:
```json
{request_json}
```
response.json（预期格式）:
```json
{response_json}
```

测试脚本要求：
1. 测试 GET /health — 期望 200 且有 status 字段
2. 测试 POST /predict — 发送 request.json 内容，验证：
   - HTTP 状态码 200
   - 响应是合法 JSON
   - 响应包含 response.json 中的所有顶层 key
3. 打印详细的测试结果（PASS/FAIL + 响应内容）
4. 全部通过时 sys.exit(0)，任一失败时 sys.exit(1)
5. 支持命令行参数 --url 覆盖服务地址

输出完整的 smoke_test.py 脚本。"""


# ─────────────────────────────────────────────
#  Phase 2 Agent
# ─────────────────────────────────────────────

class Phase2ServiceAgent:

    def __init__(self, config, state: StateStore):
        self.config = config
        self.state = state
        self.llm = LLMClient(model=config.llm_model)
        self._server_proc = None  # 保存服务进程引用

    def execute_step(self, step_id: str) -> dict[str, Any]:
        dispatch = {
            "step_05": self._step05_refactor_code,
            "step_06": self._step06_generate_json_samples,
            "step_07": self._step07_build_server,
            "step_08": self._step08_smoke_test,
        }
        handler = dispatch.get(step_id)
        if handler is None:
            raise ValueError(f"Phase2 未知步骤: {step_id}")
        return handler()

    # ── 步骤5：代码重构 ──────────────────────────

    def _step05_refactor_code(self) -> dict:
        """LLM 将 single_inference.py 重构为四函数标准结构"""
        project_dir = Path(self.state.get_project_dir())
        original_code = (project_dir / "single_inference.py").read_text(encoding="utf-8")

        logger.info("  [Act] 调用 LLM 重构代码")
        output_path = project_dir / "single_inference_refactor.py"

        code = self.llm.generate_python_code(
            system_prompt=REFACTOR_SYSTEM_PROMPT,
            user_prompt=REFACTOR_USER_TEMPLATE.format(original_code=original_code),
            output_path=output_path,
        )

        # Observe：验证四个函数都存在
        required_functions = ["init_model", "pre_process", "process", "post_process"]
        missing = [fn for fn in required_functions if f"def {fn}" not in code]
        if missing:
            raise ValueError(f"重构代码缺少函数: {missing}")

        logger.info(f"  [Observe] 重构成功: {output_path}")
        return {"refactor_py_path": str(output_path)}

    # ── 步骤6：生成接口样例 ──────────────────────

    def _step06_generate_json_samples(self) -> dict:
        """LLM 根据重构代码生成 request.json 和 response.json"""
        project_dir = Path(self.state.get_project_dir())
        refactor_code = (project_dir / "single_inference_refactor.py").read_text(encoding="utf-8")

        # 生成 request.json
        logger.info("  [Act] 生成 request.json")
        req_path = project_dir / "request.json"
        self.llm.generate_json(
            system_prompt=JSON_SYSTEM_PROMPT,
            user_prompt=(
                f"根据 pre_process 函数的参数类型，生成真实的 request JSON 样例。\n\n"
                f"代码：\n```python\n{refactor_code}\n```\n\n"
                "只输出一个可被 pre_process(raw_input) 直接使用的 JSON 对象。"
            ),
            output_path=req_path,
        )

        # 生成 response.json
        logger.info("  [Act] 生成 response.json")
        resp_path = project_dir / "response.json"
        req_content = req_path.read_text(encoding="utf-8")
        self.llm.generate_json(
            system_prompt=JSON_SYSTEM_PROMPT,
            user_prompt=(
                f"根据 post_process 函数的返回类型，生成真实的 response JSON 样例。\n\n"
                f"代码：\n```python\n{refactor_code}\n```\n\n"
                f"对应的 request 样例：\n{req_content}\n\n"
                "只输出 post_process 返回的 JSON 对象。"
            ),
            output_path=resp_path,
        )

        logger.info(f"  [Observe] ✓ request.json: {req_path}")
        logger.info(f"  [Observe] ✓ response.json: {resp_path}")
        return {
            "request_json_path": str(req_path),
            "response_json_path": str(resp_path),
        }

    # ── 步骤7：生成 FastAPI 服务 ─────────────────

    def _step07_build_server(self) -> dict:
        """LLM 将重构代码融合为 FastAPI 服务"""
        project_dir = Path(self.state.get_project_dir())
        refactor_code = (project_dir / "single_inference_refactor.py").read_text(encoding="utf-8")
        request_json = (project_dir / "request.json").read_text(encoding="utf-8")

        # 读取 server.py 模板（若存在）
        server_template_path = project_dir / "server.py"
        server_template = (
            server_template_path.read_text(encoding="utf-8")
            if server_template_path.exists()
            else DEFAULT_SERVER_TEMPLATE
        )

        logger.info("  [Act] 调用 LLM 生成 server_refactor.py")
        output_path = project_dir / "server_refactor.py"

        self.llm.generate_python_code(
            system_prompt=SERVER_SYSTEM_PROMPT,
            user_prompt=SERVER_USER_TEMPLATE.format(
                refactor_code=refactor_code,
                server_template=server_template,
                request_json=request_json,
                port=self.config.server_port,
            ),
            output_path=output_path,
        )

        logger.info(f"  [Observe] ✓ server_refactor.py: {output_path}")
        return {"server_refactor_path": str(output_path)}

    # ── 步骤8：冒烟测试 ──────────────────────────

    def _step08_smoke_test(self) -> dict:
        """启动服务 → LLM 生成冒烟测试脚本 → 执行测试"""
        project_dir = Path(self.state.get_project_dir())
        venv_python = self.state.get_venv_python()
        port = self.config.server_port
        server_url = f"http://localhost:{port}"

        # ── 8a: 生成冒烟测试脚本 ──
        request_json = (project_dir / "request.json").read_text(encoding="utf-8")
        response_json = (project_dir / "response.json").read_text(encoding="utf-8")

        logger.info("  [Act] 调用 LLM 生成冒烟测试脚本")
        smoke_path = project_dir / "smoke_test.py"
        self.llm.generate_python_code(
            system_prompt=SMOKE_TEST_SYSTEM_PROMPT,
            user_prompt=SMOKE_TEST_USER_TEMPLATE.format(
                server_url=server_url,
                request_json=request_json,
                response_json=response_json,
            ),
            output_path=smoke_path,
        )

        # ── 8b: 启动服务 ──
        logger.info(f"  [Act] 启动服务 (port={port})")
        server_script = project_dir / "server_refactor.py"
        self._server_proc = subprocess.Popen(
            [venv_python, str(server_script)],
            cwd=str(project_dir),
            env={**__import__("os").environ, "SERVER_PORT": str(port)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # 等待服务就绪（最多60秒）
        logger.info("  等待服务启动...")
        if not self._wait_for_service(port, timeout=60):
            stdout = self._server_proc.stdout.read(2000).decode(errors="ignore")
            stderr = self._server_proc.stderr.read(2000).decode(errors="ignore")
            self._server_proc.kill()
            raise RuntimeError(
                f"服务启动超时\nstdout: {stdout}\nstderr: {stderr}"
            )

        logger.info(f"  [Observe] 服务已启动: {server_url}")

        # ── 8c: 执行冒烟测试 ──
        try:
            logger.info("  [Act] 执行冒烟测试")
            executor = ShellExecutor(cwd=project_dir, venv_python=venv_python)
            result = executor.run_python(
                smoke_path,
                args=f"--url {server_url}",
                timeout=120,
            )

            if not result.success:
                raise RuntimeError(
                    f"冒烟测试失败\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )

            logger.info("  [Observe] ✓ 冒烟测试全部通过")
            return {
                "smoke_test_passed": True,
                "server_url": server_url,
                "smoke_test_output": result.stdout[-1000:],
            }

        finally:
            # 测试完成后停止服务（阶段三会重新启动）
            if self._server_proc and self._server_proc.poll() is None:
                self._server_proc.terminate()
                logger.info("  服务已停止（供后续阶段重启）")

    @staticmethod
    def _wait_for_service(port: int, timeout: int = 60) -> bool:
        """轮询直到端口可连接"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    return True
            except (ConnectionRefusedError, OSError):
                time.sleep(1)
        return False


# ─────────────────────────────────────────────
#  默认 Server 模板（当项目中没有 server.py 时使用）
# ─────────────────────────────────────────────

DEFAULT_SERVER_TEMPLATE = '''"""FastAPI inference service template"""
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
_model = None
_request_count = 0
_total_latency = 0.0


class PredictRequest(BaseModel):
    # Override this with actual fields
    data: Any


class PredictResponse(BaseModel):
    result: Any
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    logger.info("Initializing model...")
    _model = init_model()
    logger.info("Model ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="ML Inference Service", lifespan=lifespan)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    global _request_count, _total_latency
    start = time.time()
    try:
        processed = pre_process(request.dict())
        output = process(_model, processed)
        result = post_process(output)
        latency = (time.time() - start) * 1000
        _request_count += 1
        _total_latency += latency
        return PredictResponse(result=result, latency_ms=round(latency, 2))
    except Exception as e:
        logger.error(f"Predict error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": _model is not None}


@app.get("/metrics")
async def metrics():
    avg = (_total_latency / _request_count) if _request_count > 0 else 0
    return {
        "request_count": _request_count,
        "avg_latency_ms": round(avg, 2),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("SERVER_PORT", "8080")))
'''
