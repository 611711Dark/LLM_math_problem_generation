"""
反馈生成模块，使用LLM评估用户回答并生成反馈
"""
import json
import re
import sympy as sp
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent

from app.config import settings
from app.models import (
    Question, MultipleChoiceQuestion, FillInBlankQuestion,
    CalculationQuestion, Feedback, UserAnswer, QuestionType
)

class FeedbackGenerator:
    """反馈生成器"""

    def __init__(self):
        """初始化反馈生成器"""
        self.llm = ChatOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_api_base,
            model=settings.deepseek_model,
            temperature=0.7
        )

        # 导入计算表达式函数
        from app.question_gen import calculate_expression

        # 创建高级计算工具
        self.calculate_tool = StructuredTool.from_function(
            name="calculate_expression",
            func=calculate_expression,
            description="""Compute complex mathematical expressions, to be used only when calculating complex mathematical expressions is necessary.
    calculate mathematical expressions using the `sympify` function from `sympy`, parse and compute the input mathematical expression string, supports direct calls to SymPy functions (automatically recognizes x, y, z as symbolic variables)
    Parameters:
        expression (str): Mathematical expression, e.g., "223 - 344 * 6" or "sin(pi/2) + log(10)".Replace special symbols with approximate values, e.g., pi → 3.1415"
    Example expressions:
        "2 + 3*5"                          # Basic arithmetic → 17
        "expand((x + 1)**2)"               # Expand → x² + 2x + 1
        "diff(sin(x), x)"                  # Derivative → cos(x)
        "integrate(exp(x), (x, 0, 1))"      # Definite integral → E - 1
        "solve(x**2 - 4, x)"               # Solve equation → [-2, 2]
        "limit(tan(x)/x, x, 0)"            # Limit → 1
        "Sum(k, (k, 1, 10)).doit()"        # Summation → 55
        "Matrix([[1, 2], [3, 4]]).inv()"   # Matrix inverse → [[-2, 1], [3/2, -1/2]]
        "simplify((x**2 - 1)/(x + 1))"     # Simplify → x - 1
        "factor(x**2 - 2*x - 15)"          # Factorize → (x - 5)(x + 3)
        "series(cos(x), x, 0, 4)"          # Taylor series → 1 - x²/2 + x⁴/24 + O(x⁴)
        "integrate(exp(-x**2)*sin(x), (x, -oo, oo))"  # Complex integral
        "solve([x**2 + y**2 - 1, x + y - 1], [x, y])"  # Solve system of equations
        "Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).eigenvals()"  # Matrix eigenvalues
    Returns:
        str: Calculation result. If the expression cannot be parsed or computed, returns an error message (str).
    """
        )

        # 创建验证工具
        self.verify_tool = StructuredTool.from_function(
            func=lambda equation: sp.solve(sp.sympify(equation), sp.symbols('x')),
            name="verify_answer",
            description="验证数学答案是否正确"
        )

        # 合并工具
        self.math_tools = [self.calculate_tool, self.verify_tool]

        # 创建代理
        self.agent = initialize_agent(
            tools=self.math_tools,
            llm=self.llm,
            agent="zero-shot-react-description",  # 使用支持多工具的 Agent
            verbose=True,
            handle_parsing_errors=True,  # 处理解析错误
            max_iterations=5,  # 增加最大迭代次数
            early_stopping_method="generate"  # 当达到最大迭代次数时，生成最终答案
        )

        # 创建反馈生成提示模板
        self.feedback_template = ChatPromptTemplate.from_template(
            """你是一个专业的数学老师，请评估学生对以下数学问题的回答：

            问题类型: {question_type}
            问题内容: {question_content}
            正确答案: {correct_answer}
            学生回答: {user_answer}

            请提供详细的评估和反馈。

            输出格式必须是有效的JSON，包含以下字段：
            - is_correct: 布尔值，表示学生的回答是否正确
            - explanation: 详细解释为什么答案是正确或错误的
            - improvement_suggestions: 如果答案错误，提供改进建议

            请确保你的反馈是鼓励性的、教育性的，并且能帮助学生理解问题。
            """
        )

    async def generate_feedback(
        self,
        question: Question,
        user_answer: UserAnswer
    ) -> Feedback:
        """生成对用户回答的反馈"""

        # 准备提示参数
        prompt_args = {
            "question_type": question.type.value,
            "question_content": question.content,
            "correct_answer": question.answer,
            "user_answer": user_answer.answer
        }

        # 生成提示
        prompt = self.feedback_template.format_messages(**prompt_args)

        # 获取选项内容，如果是选择题
        options_info = ""
        if question.type == QuestionType.MULTIPLE_CHOICE and hasattr(question, 'options'):
            options_info = "\n        选项信息:\n"
            for option in question.options:
                options_info += f"        {option.id}: {option.content}\n"

        # 使用代理生成反馈
        agent_prompt = f"""你是一个专业的数学教师，请评估学生对以下数学问题的回答：

        问题类型: {question.type.value}
        问题内容: {question.content}{options_info}
        正确答案: {question.answer}
        学生回答: {user_answer.answer}

        请提供详细的评估和反馈。

        请使用中文回答，不要使用英文。

        如果是选择题，请在解释中明确指出正确选项的内容，而不仅仅是选项的字母。例如，不要只说“正确答案是A”，而应该说“正确答案是A（xxx）”。

        请按照以下结构进行回答，但不要使用JSON格式：

        1. 首先指出学生的回答是否正确（正确或错误）
        2. 然后详细解释为什么答案是正确或错误的
        3. 如果答案错误，提供改进建议

        请确保你的回答包含这三个部分，并且每个部分都有清晰的标记，例如“答案评估：”、“解释：”、“改进建议：”。

        对于复杂的数学问题，在解析问题时必须使用数学计算工具进行验证。在解析学生的答案时，请首先使用计算工具计算出正确答案，然后再评估学生的答案是否正确。这尤其适用于极限、导数、积分、方程求解等复杂计算。在解析过程中，请展示使用计算工具的步骤和结果。

        请在解释和改进建议中使用LaTeX格式的数学公式，使用$...$或$$...$$包围数学公式。例如，使用$f(x) = x^2$表示函数f(x) = x的平方，使用$\\frac{{1}}{{2}}$表示分数。请确保所有的数学公式都使用正确的LaTeX语法。
        """

        try:
            # 调用代理而不是直接调用LLM，这样可以使用计算工具
            response = await self.agent.ainvoke({"input": agent_prompt})

            # 打印完整的代理响应以进行调试
            print(f"\n\n反馈代理响应: {response}\n\n")

            # 从代理响应中提取输出
            output = response.get("output", "")
            print(f"\n反馈输出: {output}\n")

            # 检查代理是否成功生成反馈
            if "Agent stopped due to iteration limit or time limit" in output or not output:
                # 代理失败，直接使用LLM生成反馈
                print("\n代理失败，尝试直接使用LLM生成反馈\n")

                # 使用简化的提示，不要求使用工具
                simplified_prompt = f"""你是一个专业的数学教师，请评估学生对以下数学问题的回答：

                问题类型: {question.type.value}
                问题内容: {question.content}{options_info}
                正确答案: {question.answer}
                学生回答: {user_answer.answer}

                请提供详细的评估和反馈。请使用中文回答。

                请按照以下格式直接输出（不要使用JSON格式）：

                答案评估：[学生的答案是否正确]

                解释：[详细解释为什么答案是正确或错误的]

                改进建议：[如果答案错误，提供改进建议]
                """

                # 直接调用LLM
                llm_response = await self.llm.ainvoke(simplified_prompt)
                output = llm_response.content
                print(f"\nLLM直接生成的反馈: {output}\n")

            # 分析文本中的关键信息
            is_correct = False
            explanation = ""
            improvement_suggestions = ""

            # 检查学生答案是否正确
            if "正确" in output[:100] and "错误" not in output[:100]:
                is_correct = True
            elif question.type == QuestionType.MULTIPLE_CHOICE:
                # 选择题直接比较选项ID
                is_correct = user_answer.answer.strip().upper() == question.answer.strip().upper()
            elif question.type == QuestionType.FILL_IN_BLANK or question.type == QuestionType.CALCULATION:
                # 填空题和计算题需要更灵活的比较
                try:
                    # 尝试将答案转换为数字进行比较
                    user_num = float(user_answer.answer.strip())
                    correct_num = float(question.answer.strip())
                    # 允许小的误差
                    is_correct = abs(user_num - correct_num) < 0.0001
                except ValueError:
                    # 如果无法转换为数字，则进行字符串比较
                    is_correct = user_answer.answer.strip().lower() == question.answer.strip().lower()

            # 提取解释部分
            explanation_start = output.find("解释：")
            if explanation_start == -1:
                explanation_start = output.find("解释:")
            if explanation_start == -1:
                explanation_start = output.find("解释")

            if explanation_start != -1:
                explanation_end = output.find("改进建议", explanation_start)
                if explanation_end == -1:
                    explanation = output[explanation_start:].strip()
                else:
                    explanation = output[explanation_start:explanation_end].strip()
            else:
                # 如果没有找到解释标记，尝试提取中间部分
                lines = output.split('\n')
                if len(lines) > 2:
                    explanation = '\n'.join(lines[1:-1])
                else:
                    explanation = output

            # 提取改进建议部分
            improvement_start = output.find("改进建议：")
            if improvement_start == -1:
                improvement_start = output.find("改进建议:")
            if improvement_start == -1:
                improvement_start = output.find("改进建议")

            if improvement_start != -1:
                improvement_suggestions = output[improvement_start:].strip()
            elif not is_correct:
                # 如果答案错误但没有找到改进建议，使用最后一部分
                lines = output.split('\n')
                if len(lines) > 2:
                    improvement_suggestions = lines[-1]

            # 如果提取失败，使用默认值
            if not explanation or explanation == "Agent stopped due to iteration limit or time limit":
                print("\n创建默认反馈数据\n")
                # 简单比较答案是否相同
                is_correct = user_answer.answer.strip().lower() == question.answer.strip().lower()

                if is_correct:
                    explanation = f"你的答案是正确的！你的答案是{user_answer.answer}，这与正确答案{question.answer}一致。"
                    improvement_suggestions = "继续保持这种良好的学习状态，可以尝试更高难度的问题。"
                else:
                    explanation = f"你的答案是错误的。你的答案是{user_answer.answer}，但正确答案是{question.answer}。"
                    improvement_suggestions = "请仔细阅读题目和解析，理解问题的关键点。如果有不清楚的地方，可以向老师请教。"

            # 创建反馈数据，但不使用JSON格式
            # 将反馈格式化为纯文本格式
            correct_text = "正确" if is_correct else "错误"
            formatted_feedback = f"""答案评估：{correct_text}

{explanation}

{improvement_suggestions if not is_correct else ''}"""

            # 仍然需要为API返回创建数据结构
            feedback_data = {
                "is_correct": is_correct,
                "explanation": explanation,
                "improvement_suggestions": improvement_suggestions
            }

            # 打印格式化的反馈
            print(f"\n格式化的反馈:\n{formatted_feedback}\n")

            # 确保必要的字段存在
            if "is_correct" not in feedback_data:
                # 简单比较答案是否相同
                feedback_data["is_correct"] = user_answer.answer.strip().lower() == question.answer.strip().lower()

            if "explanation" not in feedback_data:
                feedback_data["explanation"] = "无法生成详细解释"

            # 创建反馈对象，包含格式化的反馈文本
            return Feedback(
                is_correct=feedback_data["is_correct"],
                explanation=formatted_feedback,  # 使用格式化的反馈文本
                correct_answer=question.answer,
                user_answer=user_answer.answer,
                improvement_suggestions=feedback_data.get("improvement_suggestions", "")
            )

        except Exception as e:
            # 处理解析错误
            print(f"生成反馈时出错: {e}")

            # 如果错误信息中包含"Could not parse LLM output"，则直接使用LLM的输出
            if "Could not parse LLM output" in str(e):
                # 提取LLM的原始输出
                raw_output = str(e).split("Could not parse LLM output: `")[1].rsplit("`", 1)[0]
                print(f"\n直接使用LLM的原始输出: {raw_output}\n")

                # 检查学生答案是否正确
                is_correct = False
                if "正确" in raw_output[:100] and "错误" not in raw_output[:100]:
                    is_correct = True

                # 返回一个使用原始输出的反馈
                return Feedback(
                    is_correct=is_correct,
                    explanation=raw_output,  # 直接使用原始输出
                    correct_answer=question.answer,
                    user_answer=user_answer.answer,
                    improvement_suggestions=""  # 不需要单独的改进建议，因为它已经包含在原始输出中
                )

            # 如果不是解析错误，返回一个基本的反馈
            is_correct = user_answer.answer.strip().lower() == question.answer.strip().lower()

            # 创建默认反馈数据
            print("\n创建默认异常反馈数据\n")

            if is_correct:
                explanation = f"无法生成详细反馈，但你的答案是正确的！你的答案是{user_answer.answer}，这与正确答案{question.answer}一致。"
                improvement_suggestions = "继续保持这种良好的学习状态，可以尝试更高难度的问题。"
            else:
                explanation = f"无法生成详细反馈，但你的答案是错误的。你的答案是{user_answer.answer}，但正确答案是{question.answer}。"
                improvement_suggestions = "请仔细阅读题目和解析，理解问题的关键点。如果有不清楚的地方，可以向老师请教。"

            # 格式化反馈文本
            correct_text = "正确" if is_correct else "错误"
            formatted_feedback = f"""答案评估：{correct_text}

{explanation}

{improvement_suggestions if not is_correct else ''}"""

            # 打印格式化的反馈
            print(f"\n异常情况下的格式化反馈:\n{formatted_feedback}\n")

            feedback_data = {
                "is_correct": is_correct,
                "explanation": explanation,
                "improvement_suggestions": improvement_suggestions
            }

            return Feedback(
                is_correct=feedback_data["is_correct"],
                explanation=formatted_feedback,  # 使用格式化的反馈文本
                correct_answer=question.answer,
                user_answer=user_answer.answer,
                improvement_suggestions=feedback_data.get("improvement_suggestions", "")
            )

