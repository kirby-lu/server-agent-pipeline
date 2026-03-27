"""
Phase 4 Sub-Agent — Docker 镜像构建
步骤 11-13：生成 Shell 脚本 → 容器启动验证 → 接口文档生成
"""

from __future__ import annotations

import json
import re
import socket
import time
from pathlib import Path
from typing import Any
from string import Template
import os

from tools.shell_executor import ShellExecutor
from utils.logger import setup_logger, LLMClient
from utils.state_store import StateStore
from prompts.phase4_prompts import API_DOC_SYSTEM, API_DOC_USER

logger = setup_logger("phase4_docker")

class Phase4DockerAgent:

    def __init__(self, config, state: StateStore):
        self.config = config
        self.state = state
        self.llm = LLMClient(model=config.llm_model)
        self.container_name = f"{config.project_name}_service"
        self.image_name = config.docker_image_name or f"{config.project_name}:latest"

    def execute_step(self, step_id: str) -> dict[str, Any]:
        dispatch = {
            "step_11": self._step11_generate_docker_scripts,
            "step_12": self._step12_start_container,
            "step_13": self._step13_generate_api_doc,
        }
        handler = dispatch.get(step_id)
        if handler is None:
            raise ValueError(f"Phase4 未知步骤: {step_id}")
        return handler()

    # ── 步骤11：生成 Docker Shell 脚本 ───────────

    def _step11_generate_docker_scripts(self) -> dict:
        project_dir = Path(self.state.get_project_dir())
        server_ip = self.config.server_ip
        server_port = self.config.server_port
        host_port = self.config.host_port
        project_name = self.config.project_name
        
        def save_shell(shell_path, content):
            # 写入新文件
            with open(str(shell_path), 'w') as f:
                f.write(content)
            # 添加可执行权限
            os.chmod(str(shell_path), 0o755)
            
        # 读取加载镜像
        load_image_shell_path =  project_dir / "../" / "run_load_image.sh"
        with open('templates/run_load_image.sh', 'r') as f:
            load_image_template = f.read()
        save_shell(load_image_shell_path, load_image_template)

        # 读取shell模板，然后填充变量
        create_docker_shell_path =  project_dir / "../" / "run_create_image.sh"
        with open('templates/run_create_image.sh', 'r') as f:
            template = f.read()
            
        template = Template(template)
        create_docker_template = template.substitute(
            HOST_PORT=host_port,
            SERVER_PORT=server_port,
            TASK_NAME=project_name
        )
        save_shell(create_docker_shell_path, create_docker_template)
    
        
        # 读取启动服务模板，然后填充变量
        start_server_shell_path = project_dir / "../" / "run_start_server.sh"
        with open('templates/run_start_server.sh', 'r') as f:
            template = f.read()
        template = Template(template)
        run_start_server = template.substitute(
            TASK_NAME=project_name
        )
        save_shell(start_server_shell_path, run_start_server)
        
        # 读取停止服务模板，然后填充变量
        stop_server_shell_path = project_dir / "../" / "run_stop_server.sh"
        with open('templates/run_stop_server.sh', 'r') as f:
            template = f.read()
            
        template = Template(template)
        run_stop_server = template.substitute(
            TASK_NAME=project_name
        )
        save_shell(stop_server_shell_path, run_stop_server)
        
        # shell_info = {"run_load_image.sh": load_image_template,
        #             "run_create_image.sh": create_docker_template,
        #             "run_start_server.sh": run_start_server,
        #             "run_stop_server.sh":run_stop_server}
        
        shell_path = {"run_load_image.sh": str(load_image_shell_path.resolve()),
                    "run_create_image.sh": str(create_docker_shell_path.resolve()),
                    "run_start_server.sh": str(start_server_shell_path.resolve()),
                    "run_stop_server.sh":str(stop_server_shell_path.resolve())}
        
        logger.info(f"  [Observe] ✓ {shell_path}")

        return {"docker_scripts":shell_path}

    # ── 步骤12：执行容器启动并验证 ────────────────

    def _step12_start_container(self) -> dict:
        project_dir = Path(self.state.get_project_dir())
        executor = ShellExecutor(cwd=project_dir)
        scripts = self.state.get("docker_scripts", {})
        server_ip = self.config.server_ip

        # 1. 构建镜像（如果没有 tar 包，直接 docker build）
        dockerfile = project_dir / "Dockerfile"
        if not dockerfile.exists():
            logger.info("  [Act] 生成 Dockerfile")
            self._generate_dockerfile(project_dir, executor)

        logger.info("  [Act] docker build 构建镜像")
        result = executor.run(
            f"docker build -t {self.image_name} .",
            timeout=600,
        )
        result.raise_if_failed("docker build 失败")

        # 2. 移除旧容器（如果存在）
        executor.run(f"docker rm -f {self.container_name} 2>/dev/null || true")

        # 3. 运行创建脚本
        create_script = scripts.get("run_create_docker.sh", "")
        if create_script:
            logger.info("  [Act] 执行 run_create_docker.sh")
            result = executor.run(f"bash {create_script}", timeout=60)
            result.raise_if_failed("容器创建失败")
        else:
            # 兜底：直接 docker run
            gpu_flag = "--gpus all" if self.config.gpu_available else ""
            executor.run(
                f"docker run -d --name {self.container_name} "
                f"-p {self.config.server_port}:{self.config.server_port} "
                f"{gpu_flag} {self.image_name}",
                timeout=60,
            ).raise_if_failed("docker run 失败")

        # 4. 等待服务健康
        logger.info("  等待容器服务就绪...")
        port = self.config.server_port
        if not self._wait_for_port(port, timeout=90, host=server_ip):
            logs = executor.run(f"docker logs {self.container_name} --tail 50")
            raise RuntimeError(f"容器服务启动超时\n日志:\n{logs.stdout}")

        # 5. 验证 /health 接口
        import requests as req
        try:
            resp = req.get(f"http://{server_ip}:{port}/health", timeout=10)
            if resp.status_code != 200:
                raise RuntimeError(f"/health 返回 {resp.status_code}: {resp.text}")
            logger.info(f"  [Observe] ✓ 容器服务健康: {resp.json()}")
        except Exception as e:
            raise RuntimeError(f"健康检查失败: {e}")

        return {
            "container_name": self.container_name,
            "image_name": self.image_name,
            "container_verified": True,
        }

    # ── 步骤13：生成接口文档 ──────────────────────

    def _step13_generate_api_doc(self) -> dict:
        project_dir = Path(self.state.get_project_dir())
        project_name = self.config.project_name
        request_json = (project_dir / "request.json").read_text(encoding="utf-8")
        response_json = (project_dir / "response.json").read_text(encoding="utf-8")

        # Docker 脚本内容
        run_load_image = (project_dir / "../" / "run_load_image.sh").read_text(encoding="utf-8")
        run_create_image = (project_dir / "../" / "run_create_image.sh").read_text(encoding="utf-8")
        run_start_server = (project_dir / "../" / "run_start_server.sh").read_text(encoding="utf-8")
        run_stop_server = (project_dir / "../" / "run_stop_server.sh").read_text(encoding="utf-8")

        # ── 从 StateStore 读取性能报告和数据集分析（step10 写入）────────
        perf_report = self.state.get("perf_report", {})
        dataset_analysis = self.state.get("dataset_analysis", {})

        # 若 step10 未执行（被跳过），尝试从 perf_report.json 文件兜底读取
        if not perf_report:
            perf_report_path = project_dir / "perf_report.json"
            if perf_report_path.exists():
                try:
                    perf_report = json.loads(perf_report_path.read_text(encoding="utf-8"))
                    logger.info("  [Observe] 从 perf_report.json 文件读取性能数据")
                except Exception:
                    pass

        if perf_report:
            logger.info("  [Observe] 已获取性能测试数据，将生成性能测试章节")
        else:
            logger.warning("  [Observe] 未获取到性能测试数据，性能章节将标注为待补充")

        if dataset_analysis:
            logger.info(f"  [Observe] 已获取数据集分析：{dataset_analysis.get('dataset_type', '未知')}")
        else:
            logger.warning("  [Observe] 未获取到数据集分析，数据集章节将标注为待补充")

        with open("./templates/原型服务接口文档模板.md", 'r', encoding='utf-8') as f:
            doc_template = f.read()

        # 在模板末尾追加性能测试章节占位（LLM 将根据数据填充）
        doc_template += """

# 3. 性能测试
## 3.1 数据集说明
$【TODO：根据 dataset_analysis 填写数据集类型、文件数量、大小、分辨率/时长等特征描述】

## 3.2 性能指标
$【TODO：将 perf_report 中的 QPS、P50/P95/P99 延迟、平均延迟、错误率、CPU/内存/GPU 使用率整理为表格，并说明每项指标含义】
"""

        logger.info("  [Act] 调用 LLM 生成接口文档（含性能测试章节）")
        doc_content = self.llm.complete(
            system_prompt=API_DOC_SYSTEM,
            user_prompt=API_DOC_USER.format(
                project_name=project_name,
                request_json=request_json,
                response_json=response_json,
                doc_template=doc_template,
                run_load_image=run_load_image,
                run_create_image=run_create_image,
                run_start_server=run_start_server,
                run_stop_server=run_stop_server,
                dataset_analysis=json.dumps(dataset_analysis, ensure_ascii=False, indent=2),
                perf_report=json.dumps(perf_report, ensure_ascii=False, indent=2),
            ),
            max_tokens=6000,
        )

        doc_path = project_dir / "原型服务接口文档.md"
        doc_path.write_text(doc_content, encoding="utf-8")
        logger.info(f"  [Observe] ✓ 接口文档: {doc_path}")

        return {"api_doc_path": str(doc_path)}

    # ── 内部工具 ──────────────────────────────────

    def _generate_dockerfile(self, project_dir: Path, executor: ShellExecutor) -> None:
        """当项目中无 Dockerfile 时自动生成基础版本"""
        venv_python = self.state.get_venv_python()
        py_version = executor.run(
            f'"{venv_python}" --version', stream_output=False
        ).stdout.strip()
        py_minor = re.search(r"3\.(\d+)", py_version)
        py_tag = f"3.{py_minor.group(1)}" if py_minor else "3.10"

        dockerfile = f"""FROM python:{py_tag}-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir uv && \\
    uv pip install --system -r requirements.txt

COPY . .

ENV SERVER_PORT={self.config.server_port}
EXPOSE {self.config.server_port}

CMD ["python", "server_refactor.py"]
"""
        (project_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        logger.info("  自动生成 Dockerfile")

    def _wait_for_port(self, port: int, timeout: int = 90, host: str = None) -> bool:
        if host is None:
            host = self.config.server_ip
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1):
                    return True
            except (ConnectionRefusedError, OSError):
                time.sleep(2)
        return False

    @staticmethod
    def _parse_multifile_output(text: str) -> dict[str, str]:
        """解析 LLM 多文件输出，格式：===FILE: xxx.sh===\n内容"""
        files = {}
        pattern = r"===FILE:\s*([^\n=]+)===\s*\n([\s\S]*?)(?====FILE:|$)"
        for match in re.finditer(pattern, text):
            name = match.group(1).strip()
            content = match.group(2).strip()
            # 去除可能的 markdown 代码块
            content = re.sub(r"^```[^\n]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
            files[name] = content.strip()
        return files

    @staticmethod
    def _default_script(name: str) -> str:
        """兜底脚本（LLM 生成失败时使用）"""
        defaults = {
            "run_load_image.sh": "#!/bin/bash\nset -e\necho 'Loading image...'\ndocker load -i \"${1:-image.tar}\"\n",
            "run_create_docker.sh": "#!/bin/bash\nset -e\necho 'Creating container...'\ndocker create --name ml_service -p 8080:8080 ml_service:latest\n",
            "run_start_server.sh": "#!/bin/bash\nset -e\ndocker start ml_service\necho 'Service started'\n",
            "run_stop_server.sh": "#!/bin/bash\ndocker stop ml_service && echo 'Stopped'\n",
        }
        return defaults.get(name, f"#!/bin/bash\necho '{name} placeholder'\n")
