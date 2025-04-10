"""
配置文件，包含应用程序的配置信息
"""
import os
from typing import List, ClassVar
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用程序设置"""
    # 应用名称
    APP_NAME: str = "数学题练习系统"

    # API 设置
    OPENAI_API_BASE: str = "https://api.deepseek.com"
    OPENAI_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "<deepseek_apikey>")

    # 兼容性属性
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_api_key: str = os.environ.get("DEEPSEEK_API_KEY", "<deepseek_apikey>")
    deepseek_model: str = "deepseek-chat"

    # LLM 模型名称
    LLM_MODEL_NAME: str = "deepseek-chat"

    # 数学题难度级别
    DIFFICULTY_LEVELS: ClassVar[List[str]] = ["简单", "中等", "困难"]

    # 数学题类型
    QUESTION_TYPES: ClassVar[List[str]] = ["选择题", "填空题", "计算题"]

    # 数学题目领域
    MATH_DOMAINS: ClassVar[List[str]] = ["代数", "几何", "三角函数", "微积分", "概率统计"]

    model_config = {
        "env_file": ".env"
    }

# 创建设置实例
settings = Settings()
