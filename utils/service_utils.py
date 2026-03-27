"""通用工具函数模块 - 服务管理和等待功能"""

import socket
import subprocess
import time
from pathlib import Path
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger("service_utils")


def wait_for_service(port: int, timeout: int = 60, host: str = "localhost") -> bool:
    """轮询直到端口可连接

    Args:
        port: 端口号
        timeout: 超时时间（秒）
        host: 主机地址

    Returns:
        bool: 服务是否就绪
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(1)
    return False


class ServiceManager:
    """服务进程管理器"""

    def __init__(self, project_dir: Path, venv_python: str, port: int, host: str = "localhost"):
        self.project_dir = project_dir
        self.venv_python = venv_python
        self.port = port
        self.host = host
        self._server_proc: Optional[subprocess.Popen] = None

    def start(self, script_name: str = "server_refactor.py", wait_ready: bool = True) -> None:
        """启动服务

        Args:
            script_name: 服务脚本名称
            wait_ready: 是否等待服务就绪
        """
        logger.info(f"  [Act] 启动服务 (port={self.port})")
        server_script = self.project_dir / script_name

        self._server_proc = subprocess.Popen(
            [self.venv_python, str(server_script)],
            cwd=str(self.project_dir),
            env={**__import__("os").environ, "SERVER_PORT": str(self.port)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if wait_ready:
            logger.info("  等待服务启动...")
            if not wait_for_service(self.port, timeout=60, host=self.host):
                stdout = self._server_proc.stdout.read(2000).decode(errors="ignore")
                stderr = self._server_proc.stderr.read(2000).decode(errors="ignore")
                self._server_proc.kill()
                raise RuntimeError(
                    f"服务启动超时\nstdout: {stdout}\nstderr: {stderr}"
                )
            logger.info(f"  [Observe] 服务已启动: http://{self.host}:{self.port}")

    def stop(self) -> None:
        """停止服务"""
        if self._server_proc and self._server_proc.poll() is None:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
            logger.info("  服务已停止")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
