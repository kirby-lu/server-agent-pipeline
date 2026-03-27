"""
Phase 3 Sub-Agent — 性能指标评估
步骤 9-10：精度测试重构 → 效率测试（QPS/延迟/资源）
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import socket
import threading
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List

import requests

from tools.shell_executor import ShellExecutor
from utils.logger import setup_logger, LLMClient
from utils.state_store import StateStore
from utils.service_utils import ServiceManager
from prompts.phase3_prompts import (
    PRECISION_REFACTOR_SYSTEM, PRECISION_REFACTOR_USER,
    REGENERATE_USER, EXTRACT_PRECISION_SYSTEM, EXTRACT_PRECISION_USER,
    PERF_REQUEST_GEN_SYSTEM, PERF_REQUEST_GEN_USER,
    DATASET_ANALYSIS_SYSTEM, DATASET_ANALYSIS_USER
)

logger = setup_logger("phase3_eval")


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

    # ── 步骤9：精度测试重构 ──────────────────────

    def _step09_refactor_precision_test(self) -> dict:
        project_dir = Path(self.state.get_project_dir())
        precision_test = project_dir / "val_precision.py"

        if not precision_test.exists():
            logger.warning("  val_precision.py 不存在，跳过精度测试重构")
            return {"precision_test_skipped": True}

        ip = self.config.server_ip
        port = self.config.server_port
        server_url = f"http://{ip}:{port}/infer"

        # ── 9a: 启动服务 ──
        logger.info(f"  [Act] 启动服务 ({server_url})")
        server_script = project_dir / "server_refactor.py"
        venv_python = self.state.get_venv_python()
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

        # ── 9b: 生成数据集验证脚本 ──
        val_precision = (project_dir / "val_precision.py").read_text(encoding="utf-8")
        server_refactor = (project_dir / "server_refactor.py").read_text(encoding="utf-8")
        request_json = (project_dir / "request.json").read_text(encoding="utf-8")
        response_json = (project_dir / "response.json").read_text(encoding="utf-8")

        logger.info("  [Act] 调用 LLM 改造精度测试脚本")
        output_path = project_dir / "val_precision_refactor.py"

        self.llm.generate_python_code(
            system_prompt=PRECISION_REFACTOR_SYSTEM,
            user_prompt=PRECISION_REFACTOR_USER.format(
                val_precision=val_precision,
                request_json=request_json,
                response_json=response_json,
                server_url=server_url,
                server_refactor=server_refactor,
            ),
            output_path=output_path,
        )

        logger.info(f"  [Observe] ✓ val_precision_refactor.py: {output_path}")

        # ── 9c: 进行脚本验证 ──
        logger.info(f"  [Act] 验证服务精度")
        executor = ShellExecutor(cwd=project_dir, venv_python=venv_python)
        result = executor.run_python(output_path, timeout=300)
        if not result.success:
            logger.info(f"  [Observe] ✖ error is {result.stderr[-2000:]}\n")
            logger.info("  [Act] 重新调用 LLM 改造精度测试脚本")
            self.llm.generate_python_code(
                system_prompt=PRECISION_REFACTOR_SYSTEM,
                user_prompt=REGENERATE_USER.format(
                    val_precision=val_precision,
                    error_info=result.stderr[-2000:],
                ),
                output_path=output_path,
            )
            logger.info(f"  [Observe] ✓ val_precision_refactor.py: {output_path}")

            logger.info(f"  [Act] 重新验证服务精度")
            executor = ShellExecutor(cwd=project_dir, venv_python=venv_python)
            result = executor.run_python(output_path, timeout=300)
            if not result.success:
                raise RuntimeError(
                    f"val_precision_refactor.py 执行失败 (code={result.returncode})\n"
                    f"stderr: {result.stderr[-2000:]}\n"
                    f"stdout: {result.stdout[-1000:]}"
                )

        logger.info("  [Act] 调用 LLM 提取精度信息")
        precision_info = self.llm.generate_json(
            EXTRACT_PRECISION_SYSTEM,
            EXTRACT_PRECISION_USER.format(
                content=result.stdout[-500:]
            ))

        precision_info = precision_info.get("precision_info", [])

        logger.info(f"  [Observe] ✓ 验证服务精度完成，服务精度为:{precision_info}")
        return {"server_precision": precision_info}

    # ── 步骤10：效率测试 ──────────────────────────

    def _step10_efficiency_test(self) -> dict:
        """
        启动服务 → LLM 依据数据集生成真实压测请求 → 并发压测 → 采集资源监控
        → 数据集分析 → 生成报告
        """
        project_dir = Path(self.state.get_project_dir())
        venv_python = self.state.get_venv_python()
        port = self.config.server_port
        server_url = f"http://localhost:{port}"

        # 启动服务
        self._start_server(project_dir, venv_python, port)

        try:
            # ── 10a: LLM 依据数据集自动生成压测请求数据 ──────────────
            logger.info("  [Act] 调用 LLM 生成压测请求数据")
            request_data_list = self._llm_generate_request_data_list(
                project_dir=project_dir,
                sample_count=50,
            )

            # 若 LLM 生成失败则降级：直接使用 request.json 中的固定数据
            if not request_data_list:
                logger.warning(
                    "  [Observe] LLM 未能生成压测数据，降级使用 request.json 固定数据"
                )
                request_data_list = [
                    json.loads((project_dir / "request.json").read_text())
                ]

            logger.info(
                f"  [Observe] 压测请求数据就绪，共 {len(request_data_list)} 条"
            )

            # ── 10b: 并发压测 + 资源监控 ──────────────────────────────
            logger.info("  [Act] 开始并发压测 + 资源监控")
            report = self._run_load_test(
                server_url=server_url,
                request_data_list=request_data_list,
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

            # ── 10c: 数据集分析（供接口文档使用）────────────────────
            logger.info("  [Act] 分析数据集信息")
            dataset_analysis = self._analyze_dataset(project_dir)
            logger.info(f"  [Observe] ✓ 数据集分析完成: {dataset_analysis.get('dataset_type', '未知')}")

            return {
                "perf_report_path": str(report_path),
                "qps": report.qps,
                "p50_ms": report.latency_p50_ms,
                "p95_ms": report.latency_p95_ms,
                "p99_ms": report.latency_p99_ms,
                # 以下字段供 step13 生成接口文档的性能测试章节使用
                "perf_report": asdict(report),
                "dataset_analysis": dataset_analysis,
            }

        finally:
            self._stop_server()

    def _llm_generate_request_data_list(
        self,
        project_dir: Path,
        sample_count: int = 50,
    ) -> list[dict]:
        """
        调用 LLM，依据数据集目录结构和 request.json 模板，
        自动生成一批真实的压测请求体列表。

        流程：
          1. 扫描 project_dir 下常见的数据集目录，收集文件列表
          2. 读取 request.json 模板和 server_refactor.py 中的 pre_process 函数
          3. 将上述信息拼入 Prompt，让 LLM 生成 JSON 数组
          4. 解析结果并校验，失败时返回空列表（由调用方降级处理）
        """
        # ── 1. 扫描数据集目录结构 ────────────────────────────────────
        dataset_structure = self._scan_dataset_structure(project_dir)
        if not dataset_structure:
            logger.warning("  [Observe] 未找到数据集目录，LLM 生成请求数据将依赖模板")
            dataset_structure = "（未发现数据集目录，请根据 request.json 模板生成合理的测试数据）"

        # ── 2. 读取 request.json 模板 ────────────────────────────────
        request_template_path = project_dir / "request.json"
        if not request_template_path.exists():
            logger.warning("  request.json 不存在，跳过 LLM 生成")
            return []
        request_template = request_template_path.read_text(encoding="utf-8")

        # ── 3. 提取 server_refactor.py 中的 pre_process 函数 ─────────
        pre_process_code = self._extract_pre_process_func(project_dir)

        # ── 4. 调用 LLM 生成请求数据 ────────────────────────────────
        try:
            result = self.llm.generate_json(
                system_prompt=PERF_REQUEST_GEN_SYSTEM,
                user_prompt=PERF_REQUEST_GEN_USER.format(
                    sample_count=sample_count,
                    dataset_structure=dataset_structure,
                    request_template=request_template,
                    pre_process_code=pre_process_code,
                ),
            )
        except Exception as e:
            logger.warning(f"  [Observe] LLM 生成压测数据失败: {e}")
            return []

        # ── 5. 解析并校验结果 ────────────────────────────────────────
        if isinstance(result, list):
            data_list = result
        elif isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list) and len(v) > 0:
                    data_list = v
                    break
            else:
                logger.warning(f"  [Observe] LLM 返回结构异常: {list(result.keys())}")
                return []
        else:
            return []

        valid = [item for item in data_list if isinstance(item, dict)]
        if len(valid) < len(data_list):
            logger.warning(
                f"  [Observe] 过滤掉 {len(data_list) - len(valid)} 条非法请求体"
            )

        logger.info(f"  [Observe] LLM 生成压测请求数据 {len(valid)} 条")
        return valid

    @staticmethod
    def _scan_dataset_structure(project_dir: Path, max_files: int = 200) -> str:
        """
        扫描 project_dir 下常见的数据集子目录，
        返回文件相对路径列表（字符串），供 LLM 选取真实文件名。
        """
        candidates = [
            "data", "dataset", "datasets",
            "images", "imgs", "image",
            "test", "val", "validation",
            "samples", "input", "inputs",
        ]

        lines: list[str] = []
        for name in candidates:
            for d in project_dir.iterdir() if project_dir.exists() else []:
                if d.is_dir() and d.name.lower() == name:
                    for f in sorted(d.rglob("*"))[:max_files]:
                        if f.is_file():
                            lines.append(str(f.relative_to(project_dir)))
                    break

        if not lines:
            lines = [
                str(f.relative_to(project_dir))
                for f in sorted(project_dir.iterdir())
                if f.is_file()
            ][:50]

        return "\n".join(lines[:max_files])

    @staticmethod
    def _extract_pre_process_func(project_dir: Path) -> str:
        """
        从 server_refactor.py 中提取 pre_process 函数的源码，
        帮助 LLM 理解各字段的实际含义和类型要求。
        若提取失败则返回空字符串。
        """
        server_path = project_dir / "server_refactor.py"
        if not server_path.exists():
            return ""

        import ast
        try:
            source = server_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception:
            return ""

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "pre_process":
                    lines = source.splitlines()
                    start = node.lineno - 1
                    end = node.end_lineno  # Python 3.8+
                    return "\n".join(lines[start:end])

        return ""

    def _analyze_dataset(self, project_dir: Path) -> dict:
        """
        扫描项目下数据集目录，收集文件统计信息，
        调用 LLM 生成结构化的数据集说明（供接口文档性能章节使用）。

        LLM 失败时返回基于规则统计的降级结果，不影响主流程。
        """
        # ── 1. 扫描常见数据集目录 ────────────────────────────────────
        dataset_dirs = [
            "data", "dataset", "datasets",
            "images", "imgs", "image",
            "test", "val", "validation",
            "samples", "input", "inputs",
        ]
        found_files: list[Path] = []
        for name in dataset_dirs:
            for d in (project_dir.iterdir() if project_dir.exists() else []):
                if d.is_dir() and d.name.lower() == name:
                    found_files.extend(
                        f for f in d.rglob("*") if f.is_file()
                    )
                    break

        if not found_files:
            found_files = [f for f in project_dir.iterdir() if f.is_file()]

        if not found_files:
            logger.warning("  [Observe] 未发现数据集文件，跳过数据集分析")
            return {"dataset_type": "未知", "summary": "未发现数据集文件"}

        # ── 2. 计算基础统计信息 ──────────────────────────────────────
        suffix_counter: Counter = Counter()
        total_size = 0
        for f in found_files:
            suffix_counter[f.suffix.lower()] += 1
            try:
                total_size += f.stat().st_size
            except OSError:
                pass

        total_files = len(found_files)
        total_size_mb = round(total_size / 1024 / 1024, 2)
        avg_size_mb = round(total_size_mb / total_files, 3) if total_files else 0

        file_stats = (
            f"文件总数: {total_files}\n"
            f"总大小: {total_size_mb} MB\n"
            f"平均大小: {avg_size_mb} MB\n"
            f"格式分布: {dict(suffix_counter.most_common(10))}"
        )

        sample_paths = [
            str(f.relative_to(project_dir))
            for f in found_files[:100]
        ]
        file_samples = "\n".join(sample_paths)

        # ── 3. 调用 LLM 生成结构化分析 ──────────────────────────────
        try:
            analysis = self.llm.generate_json(
                system_prompt=DATASET_ANALYSIS_SYSTEM,
                user_prompt=DATASET_ANALYSIS_USER.format(
                    file_samples=file_samples,
                    file_stats=file_stats,
                ),
            )
            analysis.setdefault("total_files", total_files)
            analysis.setdefault("total_size_mb", total_size_mb)
            analysis.setdefault("avg_file_size_mb", avg_size_mb)
            analysis.setdefault("format_distribution", dict(suffix_counter.most_common(10)))
            return analysis

        except Exception as e:
            logger.warning(f"  [Observe] LLM 数据集分析失败，降级为规则统计: {e}")
            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".aac"}
            top_exts = set(dict(suffix_counter.most_common(3)).keys())

            if top_exts & image_exts:
                dtype = "图像"
            elif top_exts & video_exts:
                dtype = "视频"
            elif top_exts & audio_exts:
                dtype = "音频"
            else:
                dtype = "其他"

            return {
                "dataset_type": dtype,
                "total_files": total_files,
                "total_size_mb": total_size_mb,
                "avg_file_size_mb": avg_size_mb,
                "format_distribution": dict(suffix_counter.most_common(10)),
                "extra_info": {},
                "summary": (
                    f"数据集共 {total_files} 个文件，"
                    f"总大小 {total_size_mb} MB，"
                    f"主要格式：{dict(suffix_counter.most_common(3))}"
                ),
            }

    def _start_server(self, project_dir: Path, venv_python: str, port: int) -> None:
        logger.info(f"  [Act] 启动推理服务 (port={port})")
        self._server_proc = subprocess.Popen(
            [venv_python, str(project_dir / "server_refactor.py")],
            cwd=str(project_dir),
            env={**os.environ, "SERVER_PORT": str(port)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
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
        request_data_list: list[dict],
        concurrent_users: int = 10,
        duration_seconds: int = 30,
    ) -> PerformanceReport:
        """
        多线程并发压测：
        - N 个工作线程持续发 POST /infer，从 request_data_list 中轮询取请求体
        - 1 个监控线程采集 CPU/GPU 资源
        """
        latencies: List[float] = []
        errors: List[str] = []
        lock = threading.Lock()
        stop_event = threading.Event()

        data_len = len(request_data_list)

        # ── 压测工作线程 ──
        def worker(worker_idx: int):
            session = requests.Session()
            call_idx = worker_idx  # 各 worker 从不同位置起步，错开请求
            while not stop_event.is_set():
                request_data = request_data_list[call_idx % data_len]
                call_idx += concurrent_users  # 步进 = 并发数，避免多 worker 重复同一条
                start = time.time()
                try:
                    resp = session.post(
                        f"{server_url}/infer",
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
        workers = [
            threading.Thread(target=worker, args=(i,), daemon=True)
            for i in range(concurrent_users)
        ]
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