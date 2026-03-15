export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export ANTHROPIC_API_KEY=sk-5da256b6d34c41dd821b65ef4162ab25

source .venv/bin/activate
uv run orchestrator.py \
        --gitlab-url https://github.com/kirby-lu/yolov8.git \
        --work-dir /Users/penglu/Desktop/ \
        --project-name YOLOv8-server
        