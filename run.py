"""
启动脚本，用于启动FastAPI应用程序
"""
import uvicorn
import os

if __name__ == "__main__":
    # 设置OpenAI API密钥（如果有）
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("警告: 未设置OPENAI_API_KEY环境变量，请确保在app/config.py中设置了API密钥")
    
    # 启动应用
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
