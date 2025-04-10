"""
数据模型，定义应用程序中使用的数据结构
"""
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class QuestionType(str, Enum):
    """问题类型枚举"""
    MULTIPLE_CHOICE = "选择题"
    FILL_IN_BLANK = "填空题"
    CALCULATION = "计算题"

class DifficultyLevel(str, Enum):
    """难度级别枚举"""
    EASY = "简单"
    MEDIUM = "中等"
    HARD = "困难"

class MathDomain(str, Enum):
    """数学领域枚举"""
    ALGEBRA = "代数"
    GEOMETRY = "几何"
    TRIGONOMETRY = "三角函数"
    CALCULUS = "微积分"
    PROBABILITY = "概率统计"

class Option(BaseModel):
    """选择题选项"""
    id: str  # A, B, C, D
    content: str

class Question(BaseModel):
    """数学问题基类"""
    id: str
    type: QuestionType
    difficulty: DifficultyLevel
    domain: MathDomain
    content: str
    answer: str
    explanation: str = ""

class MultipleChoiceQuestion(Question):
    """选择题"""
    options: List[Option]
    type: QuestionType = QuestionType.MULTIPLE_CHOICE

class FillInBlankQuestion(Question):
    """填空题"""
    type: QuestionType = QuestionType.FILL_IN_BLANK

class CalculationQuestion(Question):
    """计算题"""
    type: QuestionType = QuestionType.CALCULATION

class QuestionRequest(BaseModel):
    """生成问题的请求"""
    type: QuestionType
    difficulty: DifficultyLevel
    domain: MathDomain

class UserAnswer(BaseModel):
    """用户回答"""
    question_id: str
    answer: str

class Feedback(BaseModel):
    """AI反馈"""
    is_correct: bool
    explanation: str
    correct_answer: str
    user_answer: str
    improvement_suggestions: Optional[str] = None
