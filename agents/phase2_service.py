"""
Phase 2 Sub-Agent — 服务生成
步骤 5-8：代码重构 → 接口样例 → FastAPI 服务 → 冒烟测试
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.shell_executor import ShellExecutor
from utils.logger import setup_logger, LLMClient
from utils.state_store import StateStore
from utils.service_utils import ServiceManager
from prompts.phase2_prompts import (
    REFACTOR_SYSTEM, REFACTOR_USER,
    JSON_SYSTEM, REQUEST_JSON_USER, RESPONSE_JSON_USER,
    REQUEST_TEMPLATE, RESPONSE_TEMPLATE,
    SERVER_SYSTEM, SERVER_USER,
    SMOKE_TEST_SYSTEM, SMOKE_TEST_USER
)

logger = setup_logger("phase2_service")


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
        """LLM 将 single_inference.py 重构为四个标准函数结构"""
        project_dir = Path(self.state.get_project_dir())
        original_code = (project_dir / "single_inference.py").read_text(encoding="utf-8")

        logger.info("  [Act] 调用 LLM 重构代码")
        output_path = project_dir / "single_inference_refactor.py"

        code = self.llm.generate_python_code(
            system_prompt=REFACTOR_SYSTEM,
            user_prompt=REFACTOR_USER.format(original_code=original_code),
            output_path=output_path,
        )

        # Observe：验证四个函数都存在
        required_functions = ["init_model", "pre_process", "process", "post_process"]
        missing = [fn for fn in required_functions if f"def {fn}" not in code]
        if missing:
            raise ValueError(f"重构代码缺少函数: {missing}")
        
        # 新增验证函数是否能够正常运行
        project_dir = Path(self.state.get_project_dir())
        venv_python = self.state.get_venv_python()
        script = project_dir / "single_inference_refactor.py"
        
        executor = ShellExecutor(cwd=project_dir, venv_python=venv_python)
        result = executor.run_python(script, timeout=300)

        if not result.success:
            # 将错误输出上报，供 Orchestrator 决策
            raise RuntimeError(
                f"single_inference_refactor.py 执行失败 (code={result.returncode})\n"
                f"stderr: {result.stderr[-2000:]}\n"
                f"stdout: {result.stdout[-1000:]}"
            )

        logger.info("  [Observe] ✓ 重构为四个标准函数结构通过")
        return {
            "prototype_validated": True,
            "stdout_tail": result.stdout[-500:],    # 打印的是single_inference_refactor.py中print的东西
            "refactor_py_path": str(output_path),
            "elapsed": result.elapsed,
        }

    # ── 步骤6：生成接口样例 ──────────────────────

    def _step06_generate_json_samples(self) -> dict:
        """LLM 根据重构代码生成 request.json 和 response.json"""
        project_dir = Path(self.state.get_project_dir())
        refactor_code = (project_dir / "single_inference_refactor.py").read_text(encoding="utf-8")

        # 生成 request.json
        logger.info("  [Act] 生成 request.json")
        # single_inference_refactor = project_dir / "single_inference_refactor.py"
        original_code = (project_dir / "single_inference_refactor.py").read_text(encoding="utf-8")
        req_path = project_dir / "request.json"
        req_result = self.llm.generate_json(
            system_prompt=JSON_SYSTEM,
            user_prompt=REQUEST_JSON_USER.format(
                request_template=REQUEST_TEMPLATE,
                original_code=original_code,
            ),
            output_path=req_path,
        )

        # 生成 response.json
        logger.info("  [Act] 生成 response.json")
        resp_path = project_dir / "response.json"
        req_content = req_path.read_text(encoding="utf-8")
        resp_result = self.llm.generate_json(
            system_prompt=JSON_SYSTEM,
            user_prompt=RESPONSE_JSON_USER.format(
                response_template=RESPONSE_TEMPLATE,
                original_code=original_code,
                req_content=req_content,
            ),
            output_path=resp_path,
        )

        logger.info(f"  [Observe] ✓ request.json: {req_path}, request : {req_result}")
        logger.info(f"  [Observe] ✓ response.json: {resp_path}, response : {resp_result}")
        return {
            "request_json_path": str(req_path),
            "request_json_data": req_result,
            "response_json_path": str(resp_path),
            "response_json_data": resp_result,
        }

    # ── 步骤7：生成 FastAPI 服务 ─────────────────

    def _step07_build_server(self) -> dict:
        """LLM 将重构代码融合为 FastAPI 服务"""
        project_dir = Path(self.state.get_project_dir())
        logger.info("  [Act] 调用 LLM 生成 server_refactor.py")
        output_path = project_dir / "server_refactor.py"

        req_path = project_dir / "request.json"
        resp_path = project_dir / "response.json"
        req_content = req_path.read_text(encoding="utf-8")
        resp_content = resp_path.read_text(encoding="utf-8")
        inference_code = (project_dir / "single_inference_refactor.py").read_text(encoding="utf-8")
        server_code = Path("./templates/server.py").read_text(encoding="utf-8")
        
        self.llm.generate_python_code(
            system_prompt=SERVER_SYSTEM,
            user_prompt=SERVER_USER.format(
                ip="localhost",
                port=self.config.server_port,
                server_interface="/infer",
                request=req_content,
                response=resp_content,
                single_inference_refactor=inference_code,
                server=server_code,
            ),
            output_path=output_path,
        )

        logger.info(f"  [Observe] ✓ server_refactor.py: {output_path}")
        return {"server_refactor_path": str(output_path)}

    # ── 步骤8：冒烟测试 ──────────────────────────

    def _step08_smoke_test(self) -> dict:
        """启动服务 → 执行测试"""
        project_dir = Path(self.state.get_project_dir())
        venv_python = self.state.get_venv_python()
        ip = self.config.server_ip
        port = self.config.server_port
        server_url = f"http://{ip}:{port}/infer"
        request_json = (project_dir / "request.json").read_text(encoding="utf-8")

        # 使用 ServiceManager 管理服务
        service_mgr = ServiceManager(project_dir, venv_python, port)
        service_mgr.start()

        try:
            logger.info("  [Act] 执行冒烟测试")
            executor = ShellExecutor(cwd=project_dir)
            result = executor.run(
                SMOKE_TEST_USER.format(
                    server_url=server_url,
                    request_data=json.dumps(
                        json.loads(request_json) if isinstance(request_json, str) else request_json,
                        ensure_ascii=False,
                    ),
                ),
                timeout=600,
            )

            def is_valid_json(json_str):
                try:
                    json.loads(json_str)
                    return True
                except json.JSONDecodeError:
                    return False

            if not result.success or \
                    (is_valid_json(result.stdout) and json.loads(result.stdout)["errorCode"]) != 200:
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
            service_mgr.stop()
