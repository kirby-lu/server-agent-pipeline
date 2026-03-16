export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export ANTHROPIC_API_KEY=sk-5da256b6d34c41dd821b65ef4162ab25

source .venv/bin/activate
# 工作目录是work-dir/project-name
uv run orchestrator.py \
        --gitlab-url https://github.com/kirby-lu/yolov8.git \
        --work-dir /Users/penglu/Desktop/ \
        --project-name YOLOv8-server    # 既是本地项目名称，也是docker容器的名称
        