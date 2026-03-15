"""
example_usage.py — Pipeline 使用示例

演示三种用法：
1. 标准 CLI 运行
2. Python API 调用（带自定义回调）
3. 单步调试模式
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 将 pipeline 目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import OrchestratorAgent, PipelineConfig, PipelineStatus


# ─────────────────────────────────────────────
#  示例1：最简单的完整运行
# ─────────────────────────────────────────────

def example_full_run():
    """完整自动化运行（含人工检查点）"""
    config = PipelineConfig(
        gitlab_url="https://gitlab.example.com/ml-team/image-classifier.git",
        project_name="image-classifier",
        work_dir="/tmp/ml_pipeline",
        llm_model="claude-sonnet-4-20250514",
        server_port=8080,
        gpu_available=False,
        human_in_the_loop=True,        # 交互式检查点
        docker_image_name="image-classifier:v1.0",
    )

    orchestrator = OrchestratorAgent(config)
    result = orchestrator.run()

    if result.status == PipelineStatus.SUCCESS:
        print("\n✅ Pipeline 完成！产出物：")
        print(f"  重构代码:   {result.refactor_py}")
        print(f"  接口文档:   {result.api_doc}")
        print(f"  性能报告:   {result.perf_report}")
    else:
        print(f"\n❌ Pipeline 失败: {result.error}")
        sys.exit(1)


# ─────────────────────────────────────────────
#  示例2：CI/CD 无人值守模式
# ─────────────────────────────────────────────

def example_ci_mode():
    """CI/CD 模式：关闭人工检查点，自动通过所有步骤"""
    config = PipelineConfig(
        gitlab_url="https://gitlab.example.com/ml-team/bert-classifier.git",
        project_name="bert-classifier",
        work_dir="/opt/pipeline",
        llm_model="claude-sonnet-4-20250514",
        server_port=9090,
        gpu_available=True,
        human_in_the_loop=False,       # CI 模式：自动通过
        docker_image_name="bert-classifier:latest",
    )

    orchestrator = OrchestratorAgent(config)
    result = orchestrator.run()

    # CI 输出 JSON 摘要
    summary = {
        "status": result.status.value,
        "artifacts": {
            "refactor_py":    result.refactor_py,
            "server_py":      result.server_py,
            "api_doc":        result.api_doc,
            "perf_report":    result.perf_report,
            "docker_scripts": result.docker_scripts,
        },
        "error": result.error,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    sys.exit(0 if result.status == PipelineStatus.SUCCESS else 1)


# ─────────────────────────────────────────────
#  示例3：断点续跑（从失败处继续）
# ─────────────────────────────────────────────

def example_resume():
    """
    Pipeline 中途失败后，修复问题再重新运行。
    StateStore 会跳过已成功完成的步骤，从失败处继续。
    """
    config = PipelineConfig(
        gitlab_url="https://gitlab.example.com/ml-team/yolo-detector.git",
        project_name="yolo-detector",
        work_dir="/tmp/ml_pipeline",
        llm_model="claude-sonnet-4-20250514",
        server_port=8080,
        human_in_the_loop=False,
    )

    orchestrator = OrchestratorAgent(config)

    # 第一次运行（假设在步骤7失败）
    print("第一次运行（可能在中途失败）...")
    result1 = orchestrator.run()
    print(f"状态: {result1.status}")

    # 修复问题后，用同一个 config 重新创建 Orchestrator
    # StateStore 会从 .pipeline_state.json 恢复状态
    print("\n修复后继续运行...")
    orchestrator2 = OrchestratorAgent(config)

    # 查看当前状态
    state = orchestrator2.state.dump()
    steps = state.get("steps", {})
    for step_id, info in steps.items():
        print(f"  {step_id}: {info.get('status', 'pending')}")

    result2 = orchestrator2.run()
    print(f"最终状态: {result2.status}")


# ─────────────────────────────────────────────
#  示例4：单步调试（直接调用某个 Phase Agent）
# ─────────────────────────────────────────────

def example_single_step_debug():
    """
    直接调试某个具体步骤，无需运行完整 Pipeline。
    适用于开发和调试场景。
    """
    from utils.state_store import StateStore
    from agents.phase2_service import Phase2ServiceAgent

    config = PipelineConfig(
        gitlab_url="",
        project_name="debug-project",
        work_dir="/tmp/debug",
        llm_model="claude-sonnet-4-20250514",
        server_port=8080,
    )

    # 手动设置状态（模拟前序步骤已完成）
    work_dir = Path(config.work_dir) / config.project_name
    work_dir.mkdir(parents=True, exist_ok=True)
    state = StateStore(work_dir / ".pipeline_state.json")
    state.set("project_dir", "/path/to/your/project")
    state.set("venv_python", "/path/to/venv/bin/python")

    # 直接调用步骤5
    agent = Phase2ServiceAgent(config, state)
    result = agent.execute_step("step_05")
    print(f"步骤5结果: {result}")


# ─────────────────────────────────────────────
#  示例5：自定义 LLM Prompt 覆盖
# ─────────────────────────────────────────────

def example_custom_prompts():
    """
    通过继承 Phase Agent 来覆盖 LLM Prompt，适应特定项目需求。
    例如：项目使用 ONNX Runtime 而非 PyTorch，需要定制重构 Prompt。
    """
    from agents.phase2_service import Phase2ServiceAgent, REFACTOR_SYSTEM_PROMPT

    class OnnxPhase2Agent(Phase2ServiceAgent):
        """针对 ONNX Runtime 项目定制的 Phase2 Agent"""

        CUSTOM_SYSTEM_PROMPT = """你是 ONNX Runtime MLOps 专家。
在重构代码时，注意：
- init_model() 使用 ort.InferenceSession 加载 .onnx 文件
- pre_process() 返回 numpy array（不是 torch.Tensor）
- process() 使用 session.run() 而非 model()
- post_process() 处理 numpy 输出
"""

        def _step05_refactor_code(self):
            # 临时替换 system prompt
            original = self.llm.complete.__defaults__
            import agents.phase2_service as m
            m.REFACTOR_SYSTEM_PROMPT = self.CUSTOM_SYSTEM_PROMPT
            result = super()._step05_refactor_code()
            m.REFACTOR_SYSTEM_PROMPT = REFACTOR_SYSTEM_PROMPT  # 恢复
            return result

    print("自定义 Agent 示例（不实际执行）")


# ─────────────────────────────────────────────
#  Pipeline 状态查看工具
# ─────────────────────────────────────────────

def show_pipeline_status(project_name: str, work_dir: str = "/tmp/ml_pipeline"):
    """查看 Pipeline 当前执行状态（可在运行中调用）"""
    from utils.state_store import StateStore

    state_file = Path(work_dir) / project_name / ".pipeline_state.json"
    if not state_file.exists():
        print(f"状态文件不存在: {state_file}")
        return

    state = StateStore(state_file)
    data = state.dump()

    print(f"\n{'='*50}")
    print(f"Pipeline 状态: {data.get('pipeline_status', 'unknown')}")
    print(f"{'='*50}")

    steps = data.get("steps", {})
    status_icons = {"success": "✅", "failed": "❌", "running": "🔄",
                    "pending": "⏳", "skipped": "⏭️"}

    for step_id, info in sorted(steps.items()):
        status = info.get("status", "pending")
        icon = status_icons.get(status, "?")
        retries = info.get("retries", 0)
        elapsed = info.get("finished_at", 0) - info.get("started_at", 0)
        retry_info = f" (重试{retries}次)" if retries > 1 else ""
        time_info = f" [{elapsed:.1f}s]" if elapsed > 0 else ""
        print(f"  {icon} {step_id}: {status}{retry_info}{time_info}")

        if status == "failed":
            err = info.get("last_error", "")
            print(f"       错误: {err[:100]}")

    print()

    # 打印产出物
    artifacts = {
        "project_dir":    "项目目录",
        "refactor_py_path":     "重构代码",
        "server_refactor_path": "FastAPI 服务",
        "perf_report_path":     "性能报告",
        "api_doc_path":         "接口文档",
    }
    print("产出物:")
    for key, label in artifacts.items():
        val = data.get(key, "")
        if val:
            print(f"  {label}: {val}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "ci", "resume", "status"],
                        help="运行模式")
    parser.add_argument("--project", default="my-project")
    parser.add_argument("--work-dir", default="/tmp/ml_pipeline")
    args = parser.parse_args()

    if args.mode == "run":
        example_full_run()
    elif args.mode == "ci":
        example_ci_mode()
    elif args.mode == "resume":
        example_resume()
    elif args.mode == "status":
        show_pipeline_status(args.project, args.work_dir)
