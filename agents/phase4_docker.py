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

logger = setup_logger("phase4_docker")


DOCKER_SCRIPTS_SYSTEM = """你是 Docker 和 DevOps 专家，专门为 ML 推理服务生成 Shell 运维脚本。
输出完整可执行的 Shell 脚本，包含完善的错误处理。"""

DOCKER_SCRIPTS_USER = """基于以下信息，生成四个 Docker 运维脚本。

项目信息：
- 项目名称: {project_name}
- Docker 镜像名: {image_name}
- 容器名: {container_name}
- 服务端口: {port}
- GPU 支持: {gpu_flag}

server_refactor.py 的主要依赖（从 requirements.txt 提取）：
{requirements}

模板参考：
{template_content}

请生成以下四个脚本（以 ===FILE: 文件名=== 分隔）：

===FILE: run_load_image.sh===
功能：加载 Docker tar 镜像文件
- 接受参数 IMAGE_TAR（默认 {image_name}.tar）
- 验证文件存在
- docker load 并显示进度
- 加载后列出镜像

===FILE: run_create_docker.sh===
功能：创建并配置容器
- 挂载 weights 和 data 目录（只读）
- 设置环境变量 SERVER_PORT={port}
- 映射端口 {port}:{port}
- {gpu_create_flag}
- 容器名 {container_name}

===FILE: run_start_server.sh===
功能：启动容器内服务
- 检查容器是否存在
- docker start 容器
- 等待服务就绪（轮询 /health）
- 显示服务状态

===FILE: run_stop_server.sh===
功能：停止并清理
- docker stop 容器
- 可选 docker rm（通过 --remove 参数）

每个脚本都要有：set -e, 彩色输出（echo -e），详细注释。"""

API_DOC_SYSTEM = """你是技术文档写作专家，生成清晰规范的 API 接口文档。
输出 Markdown 格式文档。"""

API_DOC_USER = """
Role：你是一位资深的 AI 部署工程师，精通 Linux x86 环境下算法模型的高性能 Python 微服务封装，且擅长编写标准化、规范化的原型服务接口文档。
 Task：请阅读我提供的代码逻辑，以及 `request.json` / `response.json` 样例文件，按照以下具体要求填充《原型交互接口文档模板》中所有 `$【TODO】` 占位内容（全文字体无需添加背景色）：

具体任务要求
1. 简介与模块定义
   - 将 `$【TODO:请生成任务类型】` 填充为：{project_name}的中文名。
   
2. 接口规范完善（核心重点）
   - 字段映射：
     ① 读取指定的 `request.json`（输入字段来源）和 `response.json`（输出字段来源）文件；
     ② 提取两个文件中的全部字段，整理至结构化表格中；
     ③ 字段整理规则：
        - 严格保留所有字段的原始名称，**禁止修改任何字段名称**；
        - 为每个字段标记“输入/输出”属性：
          - 若字段同时出现在 `request.json` 和 `response.json` 中，属性标记为“输入/输出”；
          - 若字段仅出现在 `request.json` 中，属性标记为“输入”；
          - 若字段仅出现在 `response.json` 中，属性标记为“输出”；
        - 表格字段排列顺序：优先展示仅“输入”的字段，其次展示仅“输出”的字段，最后展示“输入/输出”的字段；若需按原始文件字段出现顺序排列，需保证“输入在前、输出在后”的核心逻辑；
     ④ 表格需至少包含“字段名称”“输入/输出属性”两列，确保结构清晰、字段无遗漏。
   - 样例构造：将request.json和response.json复制到制定位置即可
   
3. 部署与镜像规范
   -按照文档要求，将指定脚本内容复制在对应处即可
4. 已知信息
    - 模板信息为：{doc_template}
    - request.json的内容为{request_json}
    - response.json的内容为{response_json}
    - run_load_image.sh内容为：{run_load_image}
    - run_create_image.sh内容为：{run_create_image}
    - run_start_server.sh内容为：{run_start_server}
    - run_stop_server.sh内容为：{run_stop_server}
总结
1. 核心角色定位：资深 AI 部署工程师，聚焦 Linux x86 环境下 Python 微服务封装与标准化接口文档编写；
2. 核心任务：填充接口文档模板，重点完成字段映射表格（严格保名字段、按规则标记属性）、真实场景 JSON 样例、标准化 Docker 部署命令；
3. 关键约束：字段名称不可修改、Docker 命令需包含端口映射和资源限制、JSON 样例需贴合真实检测场景。

"""

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
        if not self._wait_for_port(port, timeout=90):
            logs = executor.run(f"docker logs {self.container_name} --tail 50")
            raise RuntimeError(f"容器服务启动超时\n日志:\n{logs.stdout}")

        # 5. 验证 /health 接口
        import requests as req
        try:
            resp = req.get(f"http://localhost:{port}/health", timeout=10)
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
        run_create_image = (project_dir /  "../" / "run_create_image.sh").read_text(encoding="utf-8")
        run_start_server = (project_dir /  "../" / "run_start_server.sh").read_text(encoding="utf-8")
        run_stop_server = (project_dir /  "../" / "run_stop_server.sh").read_text(encoding="utf-8")

        with open("./templates/原型服务接口文档模板.md", 'r', encoding='utf-8') as f:
            doc_template = f.read()

        logger.info("  [Act] 调用 LLM 生成接口文档")
        doc_content = self.llm.complete(
            system_prompt=API_DOC_SYSTEM,
            user_prompt=API_DOC_USER.format(
                project_name = project_name,
                request_json = request_json,
                response_json = response_json,
                doc_template = doc_template,
                run_load_image = run_load_image,
                run_create_image = run_create_image,
                run_start_server = run_start_server,
                run_stop_server = run_stop_server
            ),
            max_tokens=4096,
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

    @staticmethod
    def _wait_for_port(port: int, timeout: int = 90) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection(("localhost", port), timeout=1):
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
