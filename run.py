"""
启动脚本，用于启动FastAPI应用程序
"""
import uvicorn
import os

if __name__ == "__main__":
    # 检查是否设置了API密钥
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("\033[93m警告: 未设置DEEPSEEK_API_KEY环境变量，请确保在app/config.py中设置了API密钥\033[0m")

    # 启动应用
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
