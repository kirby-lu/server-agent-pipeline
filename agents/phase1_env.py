"""
Phase 1 Sub-Agent — 环境准备
步骤 1-4：克隆仓库 → 安装依赖 → 下载资源 → 验证原型
内部使用 ReAct 循环处理每个步骤
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from tools.shell_executor import ShellExecutor, UVManager
from utils.logger import setup_logger
from utils.logger import LLMClient
from utils.state_store import StateStore

logger = setup_logger("phase1_env")


class Phase1EnvAgent:
    """阶段一：环境准备 Sub-Agent"""

    def __init__(self, config, state: StateStore):
        self.config = config
        self.state = state
        self.work_dir = Path(config.work_dir) / config.project_name
        self.llm = LLMClient(model=config.llm_model)

    def execute_step(self, step_id: str) -> dict[str, Any]:
        """ReAct 分发：根据 step_id 路由到对应方法"""
        dispatch = {
            "step_01": self._step01_clone_repo,
            "step_02": self._step02_create_venv,
            "step_03": self._step03_download_resources,
            "step_04": self._step04_validate_prototype,
        }
        handler = dispatch.get(step_id)
        if handler is None:
            raise ValueError(f"Phase1 未知步骤: {step_id}")
        return handler()

    # ── 步骤1：克隆仓库 ──────────────────────────

    def _step01_clone_repo(self) -> dict:
        """
        Reason: 需要将远程仓库下载到本地工作目录
        Act:    执行 git clone
        Observe: 验证关键文件是否存在
        """
        logger.info("  [Reason] 准备克隆 GitLab 仓库")

        project_dir = self.work_dir / "source"
        executor = ShellExecutor(cwd=self.work_dir)

        # 若已存在则先清空（重试时避免 clone 失败）
        if project_dir.exists():
            logger.info("  [Observe] 目标目录已存在，先删除")
            shutil.rmtree(project_dir)

        logger.info(f"  [Act] git clone {self.config.gitlab_url}")
        result = executor.run(
            f"git clone {self.config.gitlab_url} {project_dir}",
            timeout=300,
        )
        result.raise_if_failed("git clone 失败")

        # Observe：验证关键文件
        required_files = ["single_inference.py", "requirements.txt"]
        missing = [f for f in required_files if not (project_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"仓库缺少关键文件: {missing}")

        logger.info(f"  [Observe] 克隆成功，项目目录: {project_dir}")

        # 写入 State Store
        self.state.set_project_dir(str(project_dir))
        return {"project_dir": str(project_dir)}

    # ── 步骤2：创建虚拟环境 ──────────────────────

    def _step02_create_venv(self) -> dict:
        """
        Reason: 需要隔离的 Python 环境安装项目依赖
        Act:    uv venv + uv pip install
        Observe: 验证 python 可执行文件存在
        """
        project_dir = Path(self.state.get_project_dir())
        logger.info(f"  [Reason] 在 {project_dir} 创建 uv 虚拟环境")

        req_file = project_dir / "requirements.txt"
        if not req_file.exists():
            raise FileNotFoundError(f"requirements.txt 不存在: {req_file}")

        uv = UVManager(project_dir)

        # Act：创建 venv
        logger.info("  [Act] uv venv 创建虚拟环境")
        uv.create_venv()

        # Act：安装依赖
        logger.info("  [Act] uv pip install -r requirements.txt")
        uv.install_requirements(req_file)

        # Observe：验证
        if not uv.is_venv_ready():
            raise RuntimeError(f"虚拟环境创建失败，python 不存在: {uv.python_path}")

        logger.info(f"  [Observe] 虚拟环境就绪: {uv.python_path}")

        self.state.set_venv_python(uv.python_path)
        return {"venv_python": uv.python_path}

    # ── 步骤3：下载权重和数据集 ──────────────────

    def _step03_download_resources(self) -> dict:
        """
        Reason: README.md 中包含权重/数据集的下载地址，需要 LLM 解析
        Act:    LLM 解析 README → 提取下载命令 → 执行下载
        Observe: 验证文件是否下载到指定位置
        """
        project_dir = Path(self.state.get_project_dir())
        readme_path = project_dir / "README.md"

        if not readme_path.exists():
            # 尝试大写变体
            for name in ["Readme.md", "readme.md", "README.MD"]:
                if (project_dir / name).exists():
                    readme_path = project_dir / name
                    break
            else:
                logger.warning("  [Observe] README.md 不存在，跳过资源下载")
                return {"resources_downloaded": False, "reason": "no README"}

        readme_content = readme_path.read_text(encoding="utf-8", errors="ignore")
        logger.info("  [Act] 调用 LLM 解析 README，提取资源下载信息")

        system_prompt = """你是一个专业的 MLOps 工程师。
                            你的任务是分析 README.md，提取所有需要下载或需要拷贝的资源（模型权重、数据集、预训练文件等）。
                            输出严格的 JSON 格式，不要有任何额外文字。"""
        user_prompt = f"""分析以下 README.md，提取所有需要下载的资源信息。
                        README 内容：
                        ```
                        {readme_content[:8000]}
                        ```

                        输出 JSON 格式：
                        {{
                        "resources": [
                            {{
                            "name": "资源名称",
                            "url": "下载地址（http/https/wget/curl/cp等命令）",
                            "local_path": "README 中指定的本地保存路径（相对于项目根目录）",
                            "type": "weights|dataset|config|other",
                            "download_command": "完整的下载命令（使用 wget 或 curl）"
                            }}
                        ],
                        "notes": "其他注意事项"
                        }}

                        如果没有需要下载的资源，返回 {{"resources": [], "notes": "no downloads required"}}"""

        try:
            resources_info = self.llm.generate_json(system_prompt, user_prompt)
        except Exception as e:
            logger.warning(f"  LLM 解析 README 失败: {e}，跳过资源下载")
            raise RuntimeError(f"资源下载失败，具体原因为:{str(e)}")    # 通过抛出异常来停止后续任务
            # return {"resources_downloaded": False, "reason": str(e)}        
        resources = resources_info.get("resources", [])
        logger.info(f"  [Observe] LLM 识别到 {len(resources)} 个资源")

        venv_python = self.state.get_venv_python()
        executor = ShellExecutor(cwd=project_dir, venv_python=venv_python)

        downloaded = []
        failed = []
        for res in resources:
            url = res.get("url", "")
            local_path = res.get("local_path", "")
            name = res.get("name", url)
            cmd = res.get("download_command", "")

            if not url or not cmd:
                continue

            # 确保目标目录存在
            if local_path:
                target = project_dir / local_path
                target.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"  [Act] 下载: {name}")
            result = executor.run(cmd, timeout=1800, stream_output=True)

            if result.success:
                downloaded.append(name)
                logger.info(f"  [Observe] ✓ 下载成功: {name}")
            else:
                failed.append(name)
                logger.warning(f"  [Observe] ✗ 下载失败: {name}\n{result.stderr[:500]}")

        return {
            "resources_downloaded": True,
            "downloaded": downloaded,
            "failed": failed,
        }

    # ── 步骤4：验证原型代码 ──────────────────────

    def _step04_validate_prototype(self) -> dict:
        """
        Reason: 确认 single_inference.py 在当前环境可正常执行
        Act:    在虚拟环境中运行脚本（设置超时防止卡死）
        Observe: 检查返回码和输出
        """
        project_dir = Path(self.state.get_project_dir())
        venv_python = self.state.get_venv_python()
        script = project_dir / "single_inference.py"

        if not script.exists():
            raise FileNotFoundError(f"single_inference.py 不存在: {script}")

        logger.info(f"  [Act] 运行 {script}")
        executor = ShellExecutor(cwd=project_dir, venv_python=venv_python)
        result = executor.run_python(script, timeout=300)

        if not result.success:
            # 将错误输出上报，供 Orchestrator 决策
            raise RuntimeError(
                f"single_inference.py 执行失败 (code={result.returncode})\n"
                f"stderr: {result.stderr[-2000:]}\n"
                f"stdout: {result.stdout[-1000:]}"
            )

        logger.info("  [Observe] ✓ 原型验证通过")
        return {
            "prototype_validated": True,
            "stdout_tail": result.stdout,    # 打印的是single_inference.py中print的东西
            "elapsed": result.elapsed,
        }
