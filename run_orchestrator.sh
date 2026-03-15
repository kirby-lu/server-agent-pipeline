export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export ANTHROPIC_API_KEY=${sk-800259e420324b48977d7c6072a100f4}

source .venv/bin/activate
uv run orchestrator.py \
        --gitlab-url https://github.com/kirby-lu/yolov8.git \
        --work-dir /Users/penglu/Desktop/ \
        --project-name YOLOv8-server
        