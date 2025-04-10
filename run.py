"""
启动脚本，用于启动FastAPI应用程序
"""
import uvicorn
import os

if __name__ == "__main__":
    # 启动应用
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
