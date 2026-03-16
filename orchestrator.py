"""
Orchestrator Agent — Hierarchical Multi-Agent Pipeline
将 Python 原型代码标准化为可部署微服务的全自动化编排器

架构：Orchestrator (Plan-and-Execute) + 4个 Phase Sub-Agent (ReAct)
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from agents.phase1_env import Phase1EnvAgent
from agents.phase2_service import Phase2ServiceAgent
from agents.phase3_eval import Phase3EvalAgent
from agents.phase4_docker import Phase4DockerAgent
from utils.state_store import StateStore, StepStatus
from utils.logger import setup_logger

logger = setup_logger("orchestrator")


# ─────────────────────────────────────────────
#  数据结构
# ─────────────────────────────────────────────

class PipelineStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    PAUSED     = "paused"       # 等待人工确认
    SUCCESS    = "success"
    FAILED     = "failed"


@dataclass
class PipelineConfig:
    """Pipeline 运行配置"""
    gitlab_url: str                          # GitLab 仓库地址
    project_name: str                        # 项目名称
    work_dir: str = "/tmp/ml_pipeline"       # 工作目录
    llm_model: str = "claude-sonnet-4-20250514"
    max_retries: int = 3                     # 每步最大重试次数
    human_in_the_loop: bool = True           # 是否启用人工检查点
    gpu_available: bool = False              # 是否有 GPU
    server_ip: str = "localhost"             # 服务端口
    server_port: int = 8080                  # 服务端口
    docker_image_name: str = ""              # Docker 镜像名（空则自动生成）


@dataclass
class PipelineResult:
    """Pipeline 最终产出物路径集合"""
    project_dir: str = ""
    refactor_py: str = ""
    request_json: str = ""
    response_json: str = ""
    server_py: str = ""
    precision_test_py: str = ""
    perf_report: str = ""
    docker_scripts: dict[str, str] = field(default_factory=dict)
    api_doc: str = ""
    status: PipelineStatus = PipelineStatus.PENDING
    error: str = ""


# ─────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────

class OrchestratorAgent:
    """
    编排器：维护全局状态机，按顺序调度四个 Phase Sub-Agent。
    每个检查点前暂停，等待人工确认后继续。
    """

    # 13个步骤的定义
    STEPS = [
        # (step_id, phase, description, is_checkpoint_after)
        ("step_01", 1, "克隆 GitLab 仓库",                False),
        ("step_02", 1, "创建 uv 虚拟环境并安装依赖",       False),
        ("step_03", 1, "LLM 解析 README 并下载权重/数据集", True),       # ← 检查点
        ("step_04", 1, "运行 single_inference.py 验证原型", True),      # ← 检查点
        ("step_05", 2, "LLM 重构为四个标准函数",              True),     # ← 检查点
        ("step_06", 2, "LLM 生成 request/response.json",   True),     # ← 检查点
        ("step_07", 2, "LLM 融合生成 server_refactor.py",  True),
        ("step_08", 2, "自动冒烟测试",                      True),   # ← 检查点
        # ("step_09", 3, "LLM 改造精度测试脚本",              False),
        # ("step_10", 3, "效率测试（QPS/延迟/资源）",          True),   # ← 检查点
        # ("step_11", 4, "LLM 生成四个 Docker Shell 脚本",   False),
        # ("step_12", 4, "执行容器启动并验证服务",             True),   # ← 检查点
        # ("step_13", 4, "LLM 生成接口文档",                  False),
    ]

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.work_dir = Path(config.work_dir) / config.project_name
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # 持久化状态存储
        self.state = StateStore(Path(".pipeline_state.json"))
        self.result = PipelineResult()

        # 初始化四个 Phase Sub-Agent
        self.phase_agents = {
            1: Phase1EnvAgent(config, self.state),
            2: Phase2ServiceAgent(config, self.state),
            3: Phase3EvalAgent(config, self.state),
            4: Phase4DockerAgent(config, self.state),
        }

    # ── 主入口 ──────────────────────────────────

    def run(self) -> PipelineResult:
        """执行完整 Pipeline，返回产出物集合"""
        logger.info("=" * 60)
        logger.info(f"Pipeline 启动: {self.config.project_name}")
        logger.info(f"工作目录: {self.work_dir}")
        logger.info("=" * 60)

        self.state.set_pipeline_status(PipelineStatus.RUNNING)

        for step_id, phase, description, is_checkpoint in self.STEPS:
            # 跳过已成功完成的步骤（支持断点续跑）
            if self.state.get_step_status(step_id) == StepStatus.SUCCESS:
                logger.info(f"[跳过] {step_id}: {description} (已完成)")
                continue

            logger.info(f"\n{'─'*50}")
            logger.info(f"[执行] {step_id}: {description}")

            success = self._execute_step_with_retry(step_id, phase, description)

            if not success:
                self.result.status = PipelineStatus.FAILED
                self.result.error = f"{step_id} 执行失败，已超过最大重试次数"
                self.state.set_pipeline_status(PipelineStatus.FAILED)
                logger.error(f"Pipeline 终止于 {step_id}")
                return self.result

            # 检查点：等待人工确认
            if is_checkpoint and self.config.human_in_the_loop:
                approved = self._human_checkpoint(step_id, description)
                if not approved:    # TODO: 即使被暂停了，但是后续依然在此停留，重新执行
                    self.result.status = PipelineStatus.PAUSED
                    self.state.set_step_status(step_id, PipelineStatus.PAUSED)
                
                    logger.warning(f"Pipeline 在检查点 {step_id} 被人工暂停")
                    return self.result

        # 收集所有产出物
        self._collect_results()
        self.result.status = PipelineStatus.SUCCESS
        self.state.set_pipeline_status(PipelineStatus.SUCCESS)
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline 全部完成！")
        self._print_summary()
        return self.result

    # ── 步骤执行（带重试） ────────────────────────

    def _execute_step_with_retry(
        self, step_id: str, phase: int, description: str
    ) -> bool:
        """执行单步，失败时自动重试，超限后上报"""
        agent = self.phase_agents[phase]
        max_retries = self.config.max_retries

        for attempt in range(1, max_retries + 1):
            try:
                self.state.set_step_status(step_id, StepStatus.RUNNING)
                self.state.increment_retry(step_id)

                result = agent.execute_step(step_id)

                self.state.set_step_status(step_id, StepStatus.SUCCESS)
                self.state.save_step_result(step_id, result)
                logger.info(f"  ✓ {step_id} 成功 (第{attempt}次尝试)")
                return True

            except Exception as e:
                logger.warning(f"  ✗ {step_id} 第{attempt}次失败: {e}")
                self.state.set_step_status(step_id, StepStatus.FAILED)
                self.state.save_step_error(step_id, str(e))

                if attempt < max_retries:
                    wait = 2 ** attempt  # 指数退避
                    logger.info(f"  等待 {wait}s 后重试...")
                    time.sleep(wait)
                else:
                    logger.error(f"  {step_id} 已达最大重试次数 ({max_retries})")

        return False

    # ── 人工检查点 ────────────────────────────────

    def _human_checkpoint(self, step_id: str, description: str) -> bool:
        """
        人工检查点：打印当前状态后等待用户输入。
        在自动化 CI 环境中可设置 human_in_the_loop=False 自动通过。
        """
        self.state.set_pipeline_status(PipelineStatus.PAUSED)
        step_result = self.state.get_step_result(step_id)

        print("\n" + "═" * 60)
        print(f"  🔍 检查点: {step_id} — {description}")
        print("═" * 60)
        if step_result:
            print("  执行结果摘要:")
            for k, v in step_result.items():
                print(f"    {k}: {v}")

        while True:
            answer = input("  是否通过此检查点继续执行？[y/n/s(跳过此步)]: ").strip().lower()
            if answer in ("y", "yes"):
                self.state.set_pipeline_status(PipelineStatus.RUNNING)
                logger.info(f"  检查点 {step_id} 已通过")
                return True
            elif answer in ("n", "no"):
                logger.warning(f"  检查点 {step_id} 被拒绝，Pipeline 暂停")
                return False
            elif answer == "s":
                logger.info(f"  跳过检查点 {step_id}")
                return True
            else:
                print("  请输入 y / n / s")

    # ── 收集产出物 ────────────────────────────────

    def _collect_results(self) -> None:
        project_dir = self.state.get("project_dir", "")
        self.result.project_dir    = project_dir
        self.result.refactor_py    = self.state.get("refactor_py_path", "")
        self.result.request_json   = self.state.get("request_json_path", "")
        self.result.response_json  = self.state.get("response_json_path", "")
        self.result.server_py      = self.state.get("server_refactor_path", "")
        self.result.precision_test_py = self.state.get("precision_test_refactor_path", "")
        self.result.perf_report    = self.state.get("perf_report_path", "")
        self.result.docker_scripts = self.state.get("docker_scripts", {})
        self.result.api_doc        = self.state.get("api_doc_path", "")

    def _print_summary(self) -> None:
        print("\n  📦 产出物清单:")
        for k, v in asdict(self.result).items():
            if v and k not in ("status", "error"):
                print(f"    {k}: {v}")


# ─────────────────────────────────────────────
#  CLI 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML Service Pipeline Orchestrator")
    parser.add_argument("--gitlab-url",    required=True,  help="GitLab 仓库地址")
    parser.add_argument("--project-name",  required=True,  help="项目名称")
    parser.add_argument("--work-dir",      default="/tmp/ml_pipeline")
    parser.add_argument("--model",         default="deepseek-chat")
    parser.add_argument("--ip",            default="localhost", type=str)
    parser.add_argument("--port",          default=8080, type=int)
    parser.add_argument("--gpu",           action="store_true")
    parser.add_argument("--no-human",      action="store_true", help="关闭人工检查点（CI 模式）")
    parser.add_argument("--docker-image",  default="")
    args = parser.parse_args()

    config = PipelineConfig(
        gitlab_url=args.gitlab_url,
        project_name=args.project_name,
        work_dir=args.work_dir,
        llm_model=args.model,
        server_ip=args.ip,
        server_port=args.port,
        gpu_available=args.gpu,
        human_in_the_loop=not args.no_human,
        docker_image_name=args.docker_image or f"{args.project_name}:latest",
    )

    orchestrator = OrchestratorAgent(config)
    result = orchestrator.run()

    exit(0 if result.status == PipelineStatus.SUCCESS else 1)
