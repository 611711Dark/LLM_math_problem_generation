"""
主应用程序文件，包含FastAPI应用程序和API路由
"""
import os
import uuid
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import (
    Question, MultipleChoiceQuestion, FillInBlankQuestion,
    CalculationQuestion, QuestionType, DifficultyLevel,
    MathDomain, UserAnswer, Feedback, QuestionRequest, Option
)
from app.question_gen import QuestionGenerator
from app.feedback_gen import FeedbackGenerator

# 创建FastAPI应用
app = FastAPI(title=settings.APP_NAME)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 设置模板
templates = Jinja2Templates(directory="templates")

# 创建问题生成器和反馈生成器
question_generator = QuestionGenerator()
feedback_generator = FeedbackGenerator()

# 内存中存储生成的问题
questions_db: Dict[str, Question] = {}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """返回主页"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/api/question-types")
async def get_question_types():
    """获取所有问题类型"""
    return [qt.value for qt in QuestionType]

@app.get("/api/difficulty-levels")
async def get_difficulty_levels():
    """获取所有难度级别"""
    return [dl.value for dl in DifficultyLevel]

@app.get("/api/math-domains")
async def get_math_domains():
    """获取所有数学领域"""
    return [md.value for md in MathDomain]

@app.post("/api/questions")
async def create_question(request: QuestionRequest):
    """生成新问题"""
    try:
        print(f"\n\n收到生成问题请求: {request}\n\n")

        # 创建默认问题，以防生成失败
        default_question = None
        question_id = str(uuid.uuid4())

        if request.type == QuestionType.MULTIPLE_CHOICE:
            # 根据领域生成不同的默认问题
            if request.domain == MathDomain.ALGEBRA:
                default_question = MultipleChoiceQuestion(
                    id=question_id,
                    type=request.type,
                    difficulty=request.difficulty,
                    domain=request.domain,
                    content="解方程: 2x + 3 = 7",
                    options=[
                        Option(id="A", content="x = 1"),
                        Option(id="B", content="x = 2"),
                        Option(id="C", content="x = 3"),
                        Option(id="D", content="x = 4")
                    ],
                    answer="B",
                    explanation="解法：将方程两边减去3得到2x = 4，然后两边除以2得到x = 2。所以答案是B。"
                )
            elif request.domain == MathDomain.GEOMETRY:
                default_question = MultipleChoiceQuestion(
                    id=question_id,
                    type=request.type,
                    difficulty=request.difficulty,
                    domain=request.domain,
                    content="一个正方形的面积是16平方厘米，它的周长是多少？",
                    options=[
                        Option(id="A", content="8厘米"),
                        Option(id="B", content="16厘米"),
                        Option(id="C", content="32厘米"),
                        Option(id="D", content="64厘米")
                    ],
                    answer="A",
                    explanation="正方形面积为16平方厘米，所以边长为4厘米。周长 = 4 × 4 = 16厘米。"
                )
            elif request.domain == MathDomain.TRIGONOMETRY:
                default_question = MultipleChoiceQuestion(
                    id=question_id,
                    type=request.type,
                    difficulty=request.difficulty,
                    domain=request.domain,
                    content="sin(π/6)的值是多少？",
                    options=[
                        Option(id="A", content="1/2"),
                        Option(id="B", content="√2/2"),
                        Option(id="C", content="√3/2"),
                        Option(id="D", content="1")
                    ],
                    answer="A",
                    explanation="sin(π/6) = sin(30°) = 1/2是一个基本的三角函数值。"
                )
            elif request.domain == MathDomain.CALCULUS:
                default_question = MultipleChoiceQuestion(
                    id=question_id,
                    type=request.type,
                    difficulty=request.difficulty,
                    domain=request.domain,
                    content="函数f(x) = x²的导数是？",
                    options=[
                        Option(id="A", content="f'(x) = x"),
                        Option(id="B", content="f'(x) = 2x"),
                        Option(id="C", content="f'(x) = x²"),
                        Option(id="D", content="f'(x) = 2x + 1")
                    ],
                    answer="B",
                    explanation="幂函数的导数公式为: d/dx(x^n) = n×x^(n-1)。因此f'(x) = 2x。"
                )
            else:  # 概率统计
                default_question = MultipleChoiceQuestion(
                    id=question_id,
                    type=request.type,
                    difficulty=request.difficulty,
                    domain=request.domain,
                    content="投一个六面骰子，投出偶数的概率是？",
                    options=[
                        Option(id="A", content="1/6"),
                        Option(id="B", content="1/3"),
                        Option(id="C", content="1/2"),
                        Option(id="D", content="2/3")
                    ],
                    answer="C",
                    explanation="六面骰子的偶数点数是2、4、6，共有3个。所以投出偶数的概率是3/6 = 1/2。"
                )
        elif request.type == QuestionType.FILL_IN_BLANK:
            # 根据领域生成不同的默认填空题
            if request.domain == MathDomain.ALGEBRA:
                default_question = FillInBlankQuestion(
                    id=question_id,
                    type=request.type,
                    difficulty=request.difficulty,
                    domain=request.domain,
                    content="如果 3x + 2 = 14，那么 x = _____",
                    answer="4",
                    explanation="将方程两边减去2得到3x = 12，然后两边除以3得到x = 4。"
                )
            elif request.domain == MathDomain.GEOMETRY:
                default_question = FillInBlankQuestion(
                    id=question_id,
                    type=request.type,
                    difficulty=request.difficulty,
                    domain=request.domain,
                    content="一个圆的半径是5厘米，它的面积是 _____ 平方厘米。(使用 π 表示圆周率)",
                    answer="25π",
                    explanation="圆的面积公式是 A = πr²，其中 r 是半径。代入 r = 5，得到 A = π × 5² = 25π 平方厘米。"
                )
            elif request.domain == MathDomain.TRIGONOMETRY:
                default_question = FillInBlankQuestion(
                    id=question_id,
                    type=request.type,
                    difficulty=request.difficulty,
                    domain=request.domain,
                    content="如果 sin(θ) = 0.5，那么 θ 的一个可能的值是 _____ 度。(给出最小正值)",
                    answer="30",
                    explanation="当 sin(θ) = 0.5 时，θ = 30° 或 θ = 150°。最小正值是 30°。"
                )
            elif request.domain == MathDomain.CALCULUS:
                default_question = FillInBlankQuestion(
                    id=question_id,
                    type=request.type,
                    difficulty=request.difficulty,
                    domain=request.domain,
                    content="函数 f(x) = 3x² + 2x - 1 的导数 f'(x) = _____",
                    answer="6x + 2",
                    explanation="使用导数的基本性质：(xⁿ)' = nxⁿ⁻¹ 和 (ax + b)' = a。因此 f'(x) = (3x²)' + (2x)' + (-1)' = 6x + 2 + 0 = 6x + 2。"
                )
            else:  # 概率统计
                default_question = FillInBlankQuestion(
                    id=question_id,
                    type=request.type,
                    difficulty=request.difficulty,
                    domain=request.domain,
                    content="从一副52张的扑克牌中随机抽取一张牌，抽到红桃A的概率是 _____",
                    answer="1/52",
                    explanation="扑克牌中只有1张红桃A，总共有52张牌。因此概率是 1/52。"
                )
        else:  # 计算题
            # 根据领域生成不同的默认计算题
            if request.domain == MathDomain.ALGEBRA:
                default_question = CalculationQuestion(
                    id=question_id,
                    type=request.type,
                    difficulty=request.difficulty,
                    domain=request.domain,
                    content="计算表达式 2(3x - 4) + 5x 的值，其中 x = 3",
                    answer="25",
                    explanation="将 x = 3 代入表达式：2(3 × 3 - 4) + 5 × 3 = 2(9 - 4) + 15 = 2 × 5 + 15 = 10 + 15 = 25"
                )
            elif request.domain == MathDomain.GEOMETRY:
                default_question = CalculationQuestion(
                    id=question_id,
                    type=request.type,
                    difficulty=request.difficulty,
                    domain=request.domain,
                    content="一个长方形的长是8厘米，宽是5厘米。计算这个长方形的面积和周长。",
                    answer="面积 = 40平方厘米，周长 = 26厘米",
                    explanation="长方形的面积 = 长 × 宽 = 8 × 5 = 40平方厘米。\n长方形的周长 = 2 × (长 + 宽) = 2 × (8 + 5) = 2 × 13 = 26厘米。"
                )
            elif request.domain == MathDomain.TRIGONOMETRY:
                default_question = CalculationQuestion(
                    id=question_id,
                    type=request.type,
                    difficulty=request.difficulty,
                    domain=request.domain,
                    content="在直角三角形ABC中，角C是直角，角A = 30°，边AC = 10厘米。计算边AB的长度。",
                    answer="20厘米",
                    explanation="在直角三角形中，如果角A = 30°，那么角B = 60°。\n使用正弦定理：\nsin(A) = 对边 / 斜边 = BC / AB\nsin(30°) = BC / AB\n1/2 = BC / AB\nAB = BC / (1/2) = BC × 2\n使用勾股定理：\nAC² + BC² = AB²\nBC² = AB² - AC²\nBC = √(AB² - AC²)\n由于 AB = BC × 2，所以 BC = AB / 2\n代入勾股定理：\nBC = √(AB² - AC²)\nAB / 2 = √(AB² - AC²)\n(AB / 2)² = AB² - AC²\nAB² / 4 = AB² - AC²\nAB² - AB² / 4 = AC²\n3AB² / 4 = AC²\nAB² = 4AC² / 3\nAB = √(4AC² / 3)\nAB = 2AC / √3\n代入 AC = 10：\nAB = 2 × 10 / √3 ≈ 20厘米"
                )
            elif request.domain == MathDomain.CALCULUS:
                default_question = CalculationQuestion(
                    id=question_id,
                    type=request.type,
                    difficulty=request.difficulty,
                    domain=request.domain,
                    content="计算函数 f(x) = x³ - 3x² + 2x 在区间 [0, 2] 上的定积分。",
                    answer="0",
                    explanation="计算定积分：\n∫₂₀ (x³ - 3x² + 2x) dx = [x⁴/4 - 3x³/3 + 2x²/2]₂₀\n= (2⁴/4 - 3×2³/3 + 2×2²/2) - (0⁴/4 - 3×0³/3 + 2×0²/2)\n= (16/4 - 24/3 + 8/2) - 0\n= 4 - 8 + 4\n= 0"
                )
            else:  # 概率统计
                default_question = CalculationQuestion(
                    id=question_id,
                    type=request.type,
                    difficulty=request.difficulty,
                    domain=request.domain,
                    content="一个箱子里有3个红球和5个蓝球。如果随机抽取2个球，计算抽到两个红球的概率。",
                    answer="3/28",
                    explanation="总共有8个球，从中抽取2个球的组合数是 C(8,2) = 28。\n从3个红球中抽取2个球的组合数是 C(3,2) = 3。\n因此抽到两个红球的概率是 3/28。"
                )

        try:
            # 尝试生成问题
            question = await question_generator.generate_question(
                question_type=request.type,
                difficulty=request.difficulty,
                domain=request.domain
            )

            # 存储问题
            questions_db[question.id] = question

            return question
        except Exception as e:
            print(f"\n\n生成问题失败: {e}\n\n")

            # 尝试从错误信息中提取JSON数据
            error_str = str(e)
            import re
            import json

            # 直接从错误信息中提取JSON字符串
            # 匹配第一个大括号到最后一个大括号
            start_idx = error_str.find('{')
            end_idx = error_str.rfind('}')

            json_str = None
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                # 提取大括号之间的内容
                json_str = error_str[start_idx:end_idx+1]

                # 删除可能的注释和其他非JSON内容
                json_str = re.sub(r'\n\s*\(.+?\)\s*\n', '\n', json_str)

                # 尝试解析JSON
                try:
                    json.loads(json_str)
                    print(f"\n从错误信息中提取的完整JSON: {json_str}\n")
                except json.JSONDecodeError:
                    # 如果解析失败，尝试清理JSON字符串
                    try:
                        # 删除可能的反引号
                        cleaned_str = json_str.replace('`', '')
                        # 删除可能的注释和其他非JSON内容
                        cleaned_str = re.sub(r'\n\s*\(.+?\)\s*\n', '\n', cleaned_str)
                        # 删除可能的注释和其他非JSON内容
                        cleaned_str = re.sub(r'\n\s*注：.+', '', cleaned_str)

                        # 尝试解析清理后的JSON
                        json.loads(cleaned_str)
                        json_str = cleaned_str
                        print(f"\n清理后的JSON: {json_str}\n")
                    except json.JSONDecodeError:
                        json_str = None

            # 如果找到了有效的JSON字符串
            if json_str:

                print(f"\n从错误信息中提取的完整JSON: {json_str}\n")

                try:
                    # 尝试解析完整的JSON
                    # 先尝试清理JSON字符串
                    # 删除可能的注释和其他非JSON内容
                    json_str = re.sub(r'\n\s*\(.+?\)\s*\n', '\n', json_str)
                    # 删除可能的修正版题目部分
                    json_str = re.sub(r'\n\s*\(注：.+?\)\s*\n', '\n', json_str)
                    json_str = re.sub(r'\n\s*修正版题目.+', '', json_str, flags=re.DOTALL)

                    # 尝试解析JSON
                    extracted_data = json.loads(json_str)
                    print(f"\n成功解析完整JSON: {extracted_data}\n")

                    # 如果成功解析了JSON，直接使用它来创建问题
                    if extracted_data and isinstance(extracted_data, dict) and "content" in extracted_data:
                        print("\n使用提取的完整JSON数据创建问题\n")

                        # 提取必要的字段
                        content = extracted_data.get("content", "")
                        answer = extracted_data.get("answer", "")
                        explanation = extracted_data.get("explanation", "")

                        if request.type == QuestionType.MULTIPLE_CHOICE and "options" in extracted_data:
                            # 处理选项
                            options_data = extracted_data.get("options", [])
                            options = []

                            if options_data and isinstance(options_data, list):
                                for opt in options_data:
                                    if isinstance(opt, dict) and "id" in opt and "content" in opt:
                                        options.append(Option(id=opt["id"], content=opt["content"]))

                            # 如果成功提取了选项，创建选择题
                            if options and content and answer:
                                # 验证答案是否在选项中
                                answer_valid = False
                                for opt in options:
                                    if opt.id == answer:
                                        answer_valid = True
                                        break

                                if not answer_valid:
                                    print(f"\n警告：答案 {answer} 不在选项中，使用默认问题\n")
                                else:
                                    custom_question = MultipleChoiceQuestion(
                                        id=question_id,
                                        type=request.type,
                                        difficulty=request.difficulty,
                                        domain=request.domain,
                                        content=content,
                                        options=options,
                                        answer=answer,
                                        explanation=explanation
                                    )

                                    questions_db[custom_question.id] = custom_question
                                    return custom_question

                        elif (request.type == QuestionType.FILL_IN_BLANK or request.type == QuestionType.CALCULATION) and content and answer:
                            # 创建填空题或计算题
                            if request.type == QuestionType.FILL_IN_BLANK:
                                custom_question = FillInBlankQuestion(
                                    id=question_id,
                                    type=request.type,
                                    difficulty=request.difficulty,
                                    domain=request.domain,
                                    content=content,
                                    answer=answer,
                                    explanation=explanation
                                )
                            else:
                                custom_question = CalculationQuestion(
                                    id=question_id,
                                    type=request.type,
                                    difficulty=request.difficulty,
                                    domain=request.domain,
                                    content=content,
                                    answer=answer,
                                    explanation=explanation
                                )

                            questions_db[custom_question.id] = custom_question
                            return custom_question
                except json.JSONDecodeError as e:
                    print(f"\n解析完整JSON失败: {e}\n")

                    # 尝试修复JSON字符串并重新解析
                    try:
                        # 尝试删除可能的干扰字符
                        cleaned_str = re.sub(r'[\n\r\t]', '', json_str)
                        # 尝试修复常见的JSON错误
                        cleaned_str = cleaned_str.replace("'", '"')

                        # 尝试删除可能的注释和其他非JSON内容
                        cleaned_str = re.sub(r'\(\u6ce8：.+?\)', '', cleaned_str)
                        cleaned_str = re.sub(r'修正版题目.+', '', cleaned_str)

                        # 尝试提取第一个有效的JSON对象
                        json_obj_match = re.search(r'\{.+?\}', cleaned_str, re.DOTALL)
                        if json_obj_match:
                            cleaned_str = json_obj_match.group(0)

                        # 尝试解析清理后的字符串
                        extracted_data = json.loads(cleaned_str)
                        print(f"\n成功解析清理后的JSON: {extracted_data}\n")

                        # 如果成功解析了JSON，直接使用它来创建问题
                        if extracted_data and isinstance(extracted_data, dict):
                            print("\n使用清理后的JSON数据创建问题\n")

                            # 提取必要的字段
                            content = extracted_data.get("content", "")
                            answer = extracted_data.get("answer", "")
                            explanation = extracted_data.get("explanation", "")

                            # 继续处理...
                    except Exception as clean_e:
                        print(f"\n清理后的JSON解析仍然失败: {clean_e}\n")

            # 如果无法从错误信息中提取完整的JSON，尝试其他方法
            # 这里我们直接使用默认问题

            print("\n无法从错误信息中提取有效数据，使用默认问题\n")
            # 如果无法提取有效数据，使用默认问题
            questions_db[default_question.id] = default_question
            return default_question
    except Exception as e:
        print(f"\n\n创建问题时出错: {e}\n\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/questions/{question_id}")
async def get_question(question_id: str):
    """获取特定问题"""
    if question_id not in questions_db:
        raise HTTPException(status_code=404, detail="问题未找到")

    return questions_db[question_id]

@app.post("/api/feedback")
async def submit_answer(user_answer: UserAnswer):
    """提交答案并获取反馈"""
    if user_answer.question_id not in questions_db:
        raise HTTPException(status_code=404, detail="问题未找到")

    question = questions_db[user_answer.question_id]

    try:
        feedback = await feedback_generator.generate_feedback(
            question=question,
            user_answer=user_answer
        )

        return feedback
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
