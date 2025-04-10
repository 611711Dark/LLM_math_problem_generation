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
            description="""计算复杂数学表达式，仅在需要计算复杂数学表达式时使用。对于简单的计算（如基本的加减乘除、简单的乘方等），请直接计算而不要使用此工具。仅在需要计算导数、积分、极限、解方程等复杂运算时使用。"""
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
            max_iterations=3  # 限制最大迭代次数
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

        请在解释和改进建议中使用LaTeX格式的数学公式，使用$...$或$$...$$包围数学公式。例如，使用$f(x) = x^2$表示函数f(x) = x的平方，使用$\\frac{{1}}{{2}}$表示分数。请确保所有的数学公式都使用正确的LaTeX语法。
        """

        try:
            # 直接调用LLM而不是代理
            response = await self.llm.ainvoke(agent_prompt)

            # 打印完整的LLM响应以进行调试
            print(f"\n\n反馈LLM响应: {response.content}\n\n")

            # 从响应中提取文本
            output = response.content
            print(f"\n反馈输出: {output}\n")

            # 分析文本中的关键信息
            is_correct = False
            explanation = ""
            improvement_suggestions = ""

            # 检查学生答案是否正确
            if "正确" in output[:100] and "错误" not in output[:100]:
                is_correct = True
            elif user_answer.answer.strip().lower() == question.answer.strip().lower():
                is_correct = True

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

            # 创建反馈数据
            feedback_data = {
                "is_correct": is_correct,
                "explanation": explanation,
                "improvement_suggestions": improvement_suggestions
            }

            # 如果提取失败，使用默认值
            if not explanation:
                print("\n创建默认反馈数据\n")
                # 简单比较答案是否相同
                is_correct = user_answer.answer.strip().lower() == question.answer.strip().lower()

                if is_correct:
                    explanation = f"你的答案是正确的！你选择了选项{question.answer}，这是正确的答案。"
                    improvement_suggestions = "继续保持这种良好的学习状态，可以尝试更高难度的问题。"
                else:
                    explanation = f"你的答案是错误的。你选择了选项{user_answer.answer}，但正确答案是{question.answer}。"
                    improvement_suggestions = "请仔细阅读题目和解析，理解问题的关键点。如果有不清楚的地方，可以向老师请教。"

                feedback_data = {
                    "is_correct": is_correct,
                    "explanation": explanation,
                    "improvement_suggestions": improvement_suggestions
                }

            # 确保必要的字段存在
            if "is_correct" not in feedback_data:
                # 简单比较答案是否相同
                feedback_data["is_correct"] = user_answer.answer.strip().lower() == question.answer.strip().lower()

            if "explanation" not in feedback_data:
                feedback_data["explanation"] = "无法生成详细解释"

            # 创建反馈对象
            return Feedback(
                is_correct=feedback_data["is_correct"],
                explanation=feedback_data["explanation"],
                correct_answer=question.answer,
                user_answer=user_answer.answer,
                improvement_suggestions=feedback_data.get("improvement_suggestions", "")
            )

        except Exception as e:
            # 处理解析错误
            print(f"生成反馈时出错: {e}")

            # 返回一个基本的反馈
            is_correct = user_answer.answer.strip().lower() == question.answer.strip().lower()

            # 创建默认反馈数据
            print("\n创建默认异常反馈数据\n")

            if is_correct:
                explanation = f"无法生成详细反馈，但你的答案是正确的！你选择了选项{question.answer}，这是正确的答案。"
                improvement_suggestions = "继续保持这种良好的学习状态，可以尝试更高难度的问题。"
            else:
                explanation = f"无法生成详细反馈，但你的答案是错误的。你选择了选项{user_answer.answer}，但正确答案是{question.answer}。"
                improvement_suggestions = "请仔细阅读题目和解析，理解问题的关键点。如果有不清楚的地方，可以向老师请教。"

            feedback_data = {
                "is_correct": is_correct,
                "explanation": explanation,
                "improvement_suggestions": improvement_suggestions
            }

            return Feedback(
                is_correct=feedback_data["is_correct"],
                explanation=feedback_data["explanation"],
                correct_answer=question.answer,
                user_answer=user_answer.answer,
                improvement_suggestions=feedback_data.get("improvement_suggestions", "")
            )

