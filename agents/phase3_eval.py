"""
Phase 3 Sub-Agent — 性能指标评估
步骤 9-10：精度测试重构 → 效率测试（QPS/延迟/资源）
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List

import requests

from tools.shell_executor import ShellExecutor
from utils.logger import setup_logger, LLMClient
from utils.state_store import StateStore

logger = setup_logger("phase3_eval")


PRECISION_REFACTOR_SYSTEM = """你是 MLOps 工程师，专注于将本地精度测试迁移为服务测试。
输出完整的 Python 测试脚本，不要任何额外解释。"""

PRECISION_REFACTOR_USER = """将 test_precision.py 中的模型加载、预处理、推理、后处理替换为 HTTP 调用。

原始 test_precision.py：
```python
{original_code}
```

single_inference_refactor.py（了解数据格式）：
```python
{refactor_code}
```

request.json 样例：
```json
{request_json}
```

服务地址：{server_url}

改造要求：
1. 删除 init_model() 调用
2. 将 pre_process + process + post_process 调用链替换为：
   - 构造 HTTP POST {server_url}/predict 的请求体
   - 发送请求并解析响应
3. 保留原有的精度计算逻辑（accuracy/mAP/F1 等指标计算不变）
4. 添加命令行参数 --url 覆盖服务地址，--timeout 设置请求超时
5. 输出精度报告到 accuracy_report.json

生成完整的 test_precision_refactor.py 文件。"""


@dataclass
class PerformanceReport:
    """性能测试报告"""
    timestamp: str
    server_url: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float
    qps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_min_ms: float
    latency_max_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_usage_percent: float
    gpu_memory_mb: float
    error_rate: float


class Phase3EvalAgent:

    def __init__(self, config, state: StateStore):
        self.config = config
        self.state = state
        self.llm = LLMClient(model=config.llm_model)
        self._server_proc = None

    def execute_step(self, step_id: str) -> dict[str, Any]:
        dispatch = {
            "step_09": self._step09_refactor_precision_test,
            "step_10": self._step10_efficiency_test,
        }
        handler = dispatch.get(step_id)
        if handler is None:
            raise ValueError(f"Phase3 未知步骤: {step_id}")
        return handler()

    # ── 步骤9：精度测试重构 ──────────────────────

    def _step09_refactor_precision_test(self) -> dict:
        project_dir = Path(self.state.get_project_dir())
        precision_test = project_dir / "test_precision.py"

        if not precision_test.exists():
            logger.warning("  test_precision.py 不存在，跳过精度测试重构")
            return {"precision_test_skipped": True}

        refactor_code = (project_dir / "single_inference_refactor.py").read_text(encoding="utf-8")
        original_code = precision_test.read_text(encoding="utf-8")
        request_json = (project_dir / "request.json").read_text(encoding="utf-8")
        server_url = f"http://localhost:{self.config.server_port}"

        logger.info("  [Act] 调用 LLM 改造精度测试脚本")
        output_path = project_dir / "test_precision_refactor.py"

        self.llm.generate_python_code(
            system_prompt=PRECISION_REFACTOR_SYSTEM,
            user_prompt=PRECISION_REFACTOR_USER.format(
                original_code=original_code,
                refactor_code=refactor_code,
                request_json=request_json,
                server_url=server_url,
            ),
            output_path=output_path,
        )

        logger.info(f"  [Observe] ✓ test_precision_refactor.py: {output_path}")
        return {"precision_test_refactor_path": str(output_path)}

    # ── 步骤10：效率测试 ──────────────────────────

    def _step10_efficiency_test(self) -> dict:
        """
        启动服务 → 并发压测 → 采集资源监控 → 生成报告
        """
        project_dir = Path(self.state.get_project_dir())
        venv_python = self.state.get_venv_python()
        port = self.config.server_port
        server_url = f"http://localhost:{port}"

        # 启动服务
        self._start_server(project_dir, venv_python, port)

        try:
            request_data = json.loads((project_dir / "request.json").read_text())

            # 并行：压测 + 资源监控
            logger.info("  [Act] 开始并发压测 + 资源监控")
            report = self._run_load_test(
                server_url=server_url,
                request_data=request_data,
                concurrent_users=10,
                duration_seconds=30,
            )

            # 写报告
            report_path = project_dir / "perf_report.json"
            report_path.write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            # 打印摘要
            self._print_report(report)
            logger.info(f"  [Observe] ✓ 性能报告: {report_path}")

            return {
                "perf_report_path": str(report_path),
                "qps": report.qps,
                "p50_ms": report.latency_p50_ms,
                "p95_ms": report.latency_p95_ms,
                "p99_ms": report.latency_p99_ms,
            }

        finally:
            self._stop_server()

    def _start_server(self, project_dir: Path, venv_python: str, port: int) -> None:
        logger.info(f"  [Act] 启动推理服务 (port={port})")
        self._server_proc = subprocess.Popen(
            [venv_python, str(project_dir / "server_refactor.py")],
            cwd=str(project_dir),
            env={**os.environ, "SERVER_PORT": str(port)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # 等待就绪
        import socket
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    logger.info("  服务就绪")
                    return
            except (ConnectionRefusedError, OSError):
                time.sleep(1)
        raise RuntimeError("服务启动超时（60s）")

    def _stop_server(self) -> None:
        if self._server_proc and self._server_proc.poll() is None:
            self._server_proc.terminate()
            self._server_proc.wait(timeout=10)
            logger.info("  服务已停止")

    def _run_load_test(
        self,
        server_url: str,
        request_data: dict,
        concurrent_users: int = 10,
        duration_seconds: int = 30,
    ) -> PerformanceReport:
        """
        多线程并发压测：
        - N 个工作线程持续发 POST /predict
        - 1 个监控线程采集 CPU/GPU 资源
        """
        latencies: List[float] = []
        errors: List[str] = []
        lock = threading.Lock()
        stop_event = threading.Event()

        # ── 压测工作线程 ──
        def worker():
            session = requests.Session()
            while not stop_event.is_set():
                start = time.time()
                try:
                    resp = session.post(
                        f"{server_url}/predict",
                        json=request_data,
                        timeout=30,
                    )
                    latency = (time.time() - start) * 1000
                    if resp.status_code == 200:
                        with lock:
                            latencies.append(latency)
                    else:
                        with lock:
                            errors.append(f"HTTP {resp.status_code}")
                except Exception as e:
                    with lock:
                        errors.append(str(e))

        # ── 资源监控线程 ──
        cpu_samples: List[float] = []
        mem_samples: List[float] = []
        gpu_samples: List[float] = []
        gpu_mem_samples: List[float] = []

        def monitor():
            try:
                import psutil
                process = psutil.Process()
                while not stop_event.is_set():
                    cpu_samples.append(psutil.cpu_percent(interval=None))
                    mem_samples.append(process.memory_info().rss / 1024 / 1024)
                    # GPU 监控（可选）
                    if self.config.gpu_available:
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            gpu_samples.append(util.gpu)
                            gpu_mem_samples.append(mem_info.used / 1024 / 1024)
                        except Exception:
                            pass
                    time.sleep(0.5)
            except ImportError:
                logger.warning("  psutil 未安装，资源监控不可用")

        # 启动线程
        workers = [threading.Thread(target=worker, daemon=True) for _ in range(concurrent_users)]
        monitor_thread = threading.Thread(target=monitor, daemon=True)

        test_start = time.time()
        for t in workers:
            t.start()
        monitor_thread.start()

        time.sleep(duration_seconds)
        stop_event.set()

        for t in workers:
            t.join(timeout=5)
        monitor_thread.join(timeout=2)

        elapsed = time.time() - test_start

        # 计算统计
        import statistics
        total_ok = len(latencies)
        total_err = len(errors)
        total = total_ok + total_err

        if latencies:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            p50 = sorted_lat[int(n * 0.50)]
            p95 = sorted_lat[int(n * 0.95)]
            p99 = sorted_lat[int(n * 0.99)]
            mean = statistics.mean(latencies)
            min_lat = min(latencies)
            max_lat = max(latencies)
        else:
            p50 = p95 = p99 = mean = min_lat = max_lat = 0.0

        return PerformanceReport(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            server_url=server_url,
            total_requests=total,
            successful_requests=total_ok,
            failed_requests=total_err,
            duration_seconds=round(elapsed, 2),
            qps=round(total_ok / elapsed, 2) if elapsed > 0 else 0,
            latency_p50_ms=round(p50, 2),
            latency_p95_ms=round(p95, 2),
            latency_p99_ms=round(p99, 2),
            latency_mean_ms=round(mean, 2),
            latency_min_ms=round(min_lat, 2),
            latency_max_ms=round(max_lat, 2),
            cpu_usage_percent=round(sum(cpu_samples) / len(cpu_samples), 1) if cpu_samples else 0,
            memory_usage_mb=round(sum(mem_samples) / len(mem_samples), 1) if mem_samples else 0,
            gpu_usage_percent=round(sum(gpu_samples) / len(gpu_samples), 1) if gpu_samples else 0,
            gpu_memory_mb=round(sum(gpu_mem_samples) / len(gpu_mem_samples), 1) if gpu_mem_samples else 0,
            error_rate=round(total_err / total * 100, 2) if total > 0 else 0,
        )

    @staticmethod
    def _print_report(r: PerformanceReport) -> None:
        print("\n  ┌─────────────────────────────────────────┐")
        print(f"  │  性能测试报告                              │")
        print("  ├─────────────────────────────────────────┤")
        print(f"  │  QPS:          {r.qps:>8.1f} req/s           │")
        print(f"  │  延迟 P50:      {r.latency_p50_ms:>8.1f} ms             │")
        print(f"  │  延迟 P95:      {r.latency_p95_ms:>8.1f} ms             │")
        print(f"  │  延迟 P99:      {r.latency_p99_ms:>8.1f} ms             │")
        print(f"  │  错误率:        {r.error_rate:>8.2f} %              │")
        print(f"  │  CPU 使用率:    {r.cpu_usage_percent:>8.1f} %              │")
        print(f"  │  内存:          {r.memory_usage_mb:>8.1f} MB             │")
        if r.gpu_usage_percent > 0:
            print(f"  │  GPU 使用率:    {r.gpu_usage_percent:>8.1f} %              │")
            print(f"  │  GPU 显存:      {r.gpu_memory_mb:>8.1f} MB             │")
        print("  └─────────────────────────────────────────┘\n")
