"""
Shell 执行器工具
封装 subprocess，支持：虚拟环境感知 / 超时 / 流式日志 / 返回结构化结果
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger("shell_executor")


@dataclass
class ShellResult:
    returncode: int
    stdout: str
    stderr: str
    command: str
    elapsed: float

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def raise_if_failed(self, msg: str = "") -> None:
        if not self.success:
            error_msg = msg or f"命令失败 (code={self.returncode})"
            raise RuntimeError(
                f"{error_msg}\n"
                f"命令: {self.command}\n"
                f"stderr: {self.stderr[-2000:]}\n"
                f"stdout: {self.stdout[-1000:]}"
            )


class ShellExecutor:
    """在指定工作目录执行 shell 命令，可切换虚拟环境"""

    def __init__(
        self,
        cwd: Optional[Path] = None,
        venv_python: Optional[str] = None,
        timeout: int = 600,
    ):
        self.cwd = cwd or Path.cwd()
        self.venv_python = venv_python  # 虚拟环境的 python 路径
        self.timeout = timeout

    def run(
        self,
        command: str,
        timeout: Optional[int] = None,
        env_extra: Optional[dict] = None,
        stream_output: bool = True,
    ) -> ShellResult:
        """执行 shell 命令"""
        timeout = timeout or self.timeout
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)

        # 注入虚拟环境路径
        if self.venv_python:
            venv_bin = str(Path(self.venv_python).parent)
            env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
            env["VIRTUAL_ENV"] = str(Path(self.venv_python).parent.parent)

        logger.debug(f"执行: {command} (cwd={self.cwd})")
        start = time.time()

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=str(self.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            stdout_lines = []
            stderr_lines = []

            if stream_output:
                # 流式读取，实时显示日志
                import select
                import threading

                def reader(stream, lines, prefix):
                    for line in stream:
                        line = line.rstrip()
                        lines.append(line)
                        logger.debug(f"  {prefix} {line}")

                t_out = threading.Thread(target=reader, args=(proc.stdout, stdout_lines, "│"))
                t_err = threading.Thread(target=reader, args=(proc.stderr, stderr_lines, "│"))
                t_out.start()
                t_err.start()

                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    raise TimeoutError(f"命令超时 ({timeout}s): {command}")

                t_out.join()
                t_err.join()
            else:
                try:
                    out, err = proc.communicate(timeout=timeout)
                    stdout_lines = out.splitlines()
                    stderr_lines = err.splitlines()
                except subprocess.TimeoutExpired:
                    proc.kill()
                    raise TimeoutError(f"命令超时 ({timeout}s): {command}")

            elapsed = time.time() - start
            result = ShellResult(
                returncode=proc.returncode,
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
                command=command,
                elapsed=elapsed,
            )

            status = "✓" if result.success else "✗"
            logger.debug(f"  {status} 耗时 {elapsed:.1f}s, code={proc.returncode}")
            return result

        except TimeoutError:
            raise
        except Exception as e:
            raise RuntimeError(f"命令执行异常: {e}\n命令: {command}")

    def run_python(self, script_path: Path, args: str = "", **kwargs) -> ShellResult:
        """在虚拟环境 Python 中运行脚本"""
        python = self.venv_python or sys.executable
        return self.run(f'"{python}" "{script_path}" {args}', **kwargs)

    def run_python_module(self, module: str, args: str = "", **kwargs) -> ShellResult:
        """运行 python -m 模块"""
        python = self.venv_python or sys.executable
        return self.run(f'"{python}" -m {module} {args}', **kwargs)


class UVManager:
    """使用 uv 管理虚拟环境"""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.venv_dir = project_dir / ".venv"
        self.executor = ShellExecutor(cwd=project_dir)

    @property
    def python_path(self) -> str:
        if sys.platform == "win32":
            return str(self.venv_dir / "Scripts" / "python.exe")
        return str(self.venv_dir / "bin" / "python")

    def create_venv(self) -> None:
        """创建虚拟环境"""
        logger.info(f"  创建虚拟环境: {self.venv_dir}")
        result = self.executor.run(f"uv venv {self.venv_dir} --clear --python 3.10")    # TODO: python版本需要从外部传入
        result.raise_if_failed("uv venv 创建失败")

    def install_requirements(self, requirements_file: Path) -> None:
        """安装 requirements.txt"""
        logger.info(f"  追加微服务依赖: {requirements_file}")
        # 追加依赖
        result = self.executor.run(
            f"echo 'Flask==3.1.3' >> {requirements_file}",
            f"echo 'fastapi==0.135.1' >> {requirements_file}",
            timeout=1200,  # 依赖安装可能很慢
        )
        result.raise_if_failed("追加微服务以来失败")
        
        logger.info(f"  安装依赖: {requirements_file}")
        result = self.executor.run(
            f"uv pip install -r {requirements_file} --python {self.python_path}",
            timeout=1200,  # 依赖安装可能很慢
        )
        result.raise_if_failed("依赖安装失败")

    def is_venv_ready(self) -> bool:
        return Path(self.python_path).exists()
