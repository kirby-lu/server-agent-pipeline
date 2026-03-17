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
import textwrap

logger = setup_logger("phase2_service")


# ─────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────

REFACTOR_SYSTEM_PROMPT = """你是一位资深 MLOps 工程师，专注于将机器学习推理代码标准化为生产可用的模块。
你的任务是将 single_inference.py 中的逻辑严格重构为四个标准函数，保持原有逻辑不变，只做结构拆分。
输出完整的 Python 文件，不要有任何额外解释。"""

REFACTOR_USER_TEMPLATE = """
将以下 single_inference.py 重构为包含四个标准函数的 single_inference_refactor.py。
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
- 输出完整可运行的 Python 文件
- 删除掉任何不影响最终结果的文件落盘的功能
- 在if __name__ == "__main__":下依次调用上面四个函数,使其能够正常处理
"""

JSON_SYSTEM_PROMPT = """你是 API 设计专家。
根据推理服务代码，生成符合实际数据类型的请求和响应 JSON 样例。
只输出纯 JSON，不要 Markdown 代码块，不要任何解释。"""

def get_request_json_user_prompt(original_code):
    template = {
        "requestId": "123456",  
        "body": {
            "resourceUrl": "image1.png" 
        }
    }
    template = json.dumps(template, indent=4, ensure_ascii=False)
    REQUEST_JSON_USER_PROMPT = textwrap.dedent(f"""
        指令：优化推理服务输入参数设计
        当前上下文： 我正在将推理逻辑封装为微服务，需要定义original_code中pre_process函数接收的参数结构。 
        核心要求：
            - 不可变量：保持外层的 requestId（用于日志追踪）和内层的 body（业务负荷）不动。
            - 增量设计：分析original_code中的pre_process 、process和post_process的输入需求。
                    如果除了resourceUrl(表示输入资源的字段名称,其值请填充original_code中真实的数据),还需要其他控制变量(device_id除外、模型名称),
                    请在 body 下方进行增量定义,请确保不同变量名的含义不要出现重复，
            - 逻辑闭环：设计的参数不应涵盖原始输入数据的标准化、格式转换等操作的信息 
            - 请求模板：{template},requestId表示每个请求唯一的uuid字符传
            - original_code为:{original_code}
    """)
    return REQUEST_JSON_USER_PROMPT

def get_response_json_user_prompt(original_code, req_content):
    # 先定义干净的 JSON 模板（不带注释和多余逗号）
    template = {
        "requestId": "123456",
        "body": {
            "result": "",
            "status": "",
            "latency": {
                "pre_process": 0,
                "process": 0,
                "post_process": 0
            }
        },
        "errorCode": 200,
        "version": "v1.0.0.0"
    }
    
    # 转换为格式化的 JSON 字符串
    template = json.dumps(template, indent=4, ensure_ascii=False)
    
    RESPONSE_JSON_USER_PROMPT = textwrap.dedent(f"""
        指令：优化推理服务响应参数设计
        当前上下文： 我正在将推理逻辑封装为微服务，需要定义original_code中post_process 接收的参数结构。 核心要求：
        - 不可变量：
            (1) 响应的requestId要和req_content中的requestId保持一致
            (2) body.result: 填充original_code中post_process 实际返回的推理数据
            (3) body.status: 若推理过程无异常，固定返回 "success"；若捕获到异常，返回具体的错误描述
            (4) body.latency: 精确记录并填充original_code中pre_process、process、post_process 三个阶段的耗时（单位：毫秒）
        - 增量设计: 分析original_code中post_process的输入需求。如果还需要其他非常必要的变量，请在 body 下方进行增量定义。
        - 逻辑闭环:设计的参数不应涵盖原始输入数据的标准化、格式转换等操作的信息
        - 输出响应模板：{template}
        - original_code为:{original_code}
        - req_content为: {req_content}
        注：模板中的 "requestId" 需与请求中的 requestId 保持一致
    """)
    
    return RESPONSE_JSON_USER_PROMPT

SERVER_SYSTEM_PROMPT = """
你是 FastAPI 专家，将推理函数封装为生产级 HTTP 服务。
输出完整的 server_refactor.py 文件，不要有任何额外解释。"""

def get_server_user_prompt(request,response,single_inference_refactor, server, 
                            ip, port, server_interface):
    
    SERVER_USER_TEMPLATE = f"""
        请作为一名资深 Python 开发工程师，协助我完成 AI 推理服务代码的重构与代码融合。
        1. 任务目标
        请参考提供的 请求request和响应response以及single_inference_refactor,对模板server文件进行以下重构：
        - 数据模型对齐：修改 InferenceRequest、RequestBodyData（如需比较可以新增或删除python类）
            确保对请求request字段进行对齐 ，修改InferenceResponse、ResponseBodyData 和 LatencyData 的字段定义（如需比较可以新增或删除python类），
            确保其与 响应response示例中的层级及字段名严格一一对应。
        - 核心逻辑重写：根据新的数据结构，重新实现single_inference_refactor中的init_model()、pre_process()、process() 和 post_process() 函数。
        - 接口规范优化：优化 infer() 函数内部的调用链路，确保预处理、推理和后处理的返回值类型一致，且最终生成的响应体符合 请求request和 响应response规范。
        - 服务接口修改：请将服务的ip改为{ip}, 端口号改为{port},服务接口改为{server_interface}
        2. 融合要求
        - 无缝集成：将 single_inference_refactor中的init_model()、pre_process()、process() 和 post_process()融入到server.py中。
        - 性能与健壮性：在post_process中需准确计算并填充 LatencyData；在各环节加入必要的异常处理。
        - 代码风格：保持类型注解（Type Hinting）的一致性，确保代码简洁且符合 PEP8 规范。
        3. 补充：
        - request内容为: {request}
        - response内容为: {response}
        - single_inference_refactor内容为: {single_inference_refactor}
        - 原始server文件为: {server}
        """
    return SERVER_USER_TEMPLATE

SMOKE_TEST_SYSTEM_PROMPT = """你是 QA 工程师，专门编写 HTTP 服务的冒烟测试脚本。
输出完整的 Python 测试脚本，不要有任何额外解释。"""

def get_smoke_test_user_template(request_json, server_url):
    # 如果 request_json 是字符串，先解析
    if isinstance(request_json, str):
        request_data = json.loads(request_json)
    else:
        request_data = request_json
    
    # 生成新的 curl 命令
    curl_cmd = f"""curl -X POST "{server_url}" \\
    -H "Content-Type: application/json" \\
    -d '{json.dumps(request_data, ensure_ascii=False)}'"""
    
    return curl_cmd
    
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
            system_prompt=REFACTOR_SYSTEM_PROMPT,
            user_prompt=REFACTOR_USER_TEMPLATE.format(original_code=original_code),
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
            system_prompt=JSON_SYSTEM_PROMPT,
            user_prompt=get_request_json_user_prompt(original_code),
            output_path=req_path,
        )

        # 生成 response.json
        logger.info("  [Act] 生成 response.json")
        resp_path = project_dir / "response.json"
        req_content = req_path.read_text(encoding="utf-8")
        resp_result = self.llm.generate_json(
            system_prompt=JSON_SYSTEM_PROMPT,
            user_prompt=get_response_json_user_prompt(original_code,req_content),
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
            system_prompt=SERVER_SYSTEM_PROMPT,
            user_prompt=get_server_user_prompt(
                            req_content,
                            resp_content,
                            inference_code, 
                            server_code,
                            ip = "localhost",
                            port = self.config.server_port,
                            server_interface = "/infer"), 
            output_path=output_path,
        )

        logger.info(f"  [Observe] ✓ server_refactor.py: {output_path}")
        return {"server_refactor_path": str(output_path)}

    # ── 步骤8：冒烟测试 ──────────────────────────

    def _step08_smoke_test(self) -> dict:
        """启动服务 → 启动服务 → 执行测试"""
        project_dir = Path(self.state.get_project_dir())
        venv_python = self.state.get_venv_python()
        ip = self.config.server_ip
        port = self.config.server_port
        server_url = f"http://{ip}:{port}/infer"
        request_json = (project_dir / "request.json").read_text(encoding="utf-8")

        # ── 8a: 启动服务 ──
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

        # ── 8b: 执行冒烟测试 ──
        try:
            logger.info("  [Act] 执行冒烟测试")
            executor = ShellExecutor(cwd=project_dir)
            logger.info(f"  request_json is \n{request_json}")
            result = executor.run(
                get_smoke_test_user_template(request_json=request_json,
                                             server_url=server_url),
                timeout=600,
            )

            def is_valid_json(json_str):
                """判断字符串是否为有效的 JSON"""
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

