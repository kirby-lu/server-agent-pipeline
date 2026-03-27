# 1. 设置环境变量
$env:ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
$env:ANTHROPIC_API_KEY = "sk-5da256b6d34c41dd821b65ef4162ab25"

# 2. 杀死占用 8080 端口的进程 (相当于 lsof + kill)
$portProcess = Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue
if ($portProcess) {
    Write-Host "正在停止占用 8080 端口的进程..." -ForegroundColor Yellow
    Stop-Process -Id $portProcess.OwningProcess -Force
}

# 3. 激活虚拟环境 (假设 Windows 下路径为 .venv\Scripts\activate)
# 注意：在 PowerShell 中通常直接运行脚本即可，或者使用下面的方式
& .\.venv\Scripts\Activate.ps1

# 4. 执行项目 (使用 ` 符号进行换行连接)
uv run orchestrator_with_critic.py `
        --gitlab-url https://gitee.com/oreolp/yolov8.git `
        --work-dir "C:\Users\penglu\Desktop\" `
        --project-name "YOLOv8-server"