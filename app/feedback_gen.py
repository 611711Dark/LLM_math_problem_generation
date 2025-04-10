"""
反馈生成模块，使用LLM评估用户回答并生成反馈
"""
import json
import sympy as sp
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent

from app.config import settings
from app.models import (
    Question, MultipleChoiceQuestion, FillInBlankQuestion,
    CalculationQuestion, Feedback, UserAnswer
)

class FeedbackGenerator:
    """反馈生成器"""

    def __init__(self):
        """初始化反馈生成器"""
        self.llm = ChatOpenAI(
            openai_api_base=settings.OPENAI_API_BASE,
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.LLM_MODEL_NAME,
            temperature=0.3  # 使用较低的温度以获得更确定性的回答
        )

        # 导入计算表达式函数
        from app.question_gen import calculate_expression

        # 创建高级计算工具
        self.calculate_tool = StructuredTool.from_function(
            name="calculate_expression",
            func=calculate_expression,
            description="""计算数学表达式，使用 `sympy` 中的 `sympify` 函数实现,解析并计算输入的数学表达式字符串,支持直接调用SymPy函数（自动识别x,y,z为符号变量）"""
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
            verbose=True
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

        # 使用代理生成反馈
        agent_prompt = f"""你是一个专业的数学教师，请评估学生对以下数学问题的回答：

        问题类型: {question.type.value}
        问题内容: {question.content}
        正确答案: {question.answer}
        学生回答: {user_answer.answer}

        请提供详细的评估和反馈。

        请使用中文回答，不要使用英文。

        输出格式必须是有效的JSON，包含以下字段：
        - is_correct: 布尔值，表示学生的回答是否正确
        - explanation: 详细解释为什么答案是正确或错误的（使用中文）
        - improvement_suggestions: 如果答案错误，提供改进建议（使用中文）
        """

        try:
            # 调用代理
            response = await self.agent.ainvoke({"input": agent_prompt})

            # 打印完整的代理响应以进行调试
            print(f"\n\n反馈代理响应: {response}\n\n")

            # 从代理响应中提取JSON
            output = response.get("output", "")
            print(f"\n提取的反馈输出: {output}\n")

            # 寻找可能的JSON字符串
            import re

            # 直接提取完整的JSON字符串
            # 首先尝试提取完整的JSON对象
            full_json_match = re.search(r'`([^`]*\{[\s\S]*\}[^`]*)`', output)
            feedback_data = None

            if full_json_match:
                # 如果找到了完整的JSON字符串，直接使用它
                json_str = full_json_match.group(1).strip()
                # 删除可能的反引号
                json_str = json_str.replace('`', '')
                # 删除可能的"json"标记
                json_str = re.sub(r'^json\s*', '', json_str)

                print(f"\n从反馈中提取的完整JSON: {json_str}\n")

                try:
                    # 尝试解析完整的JSON
                    feedback_data = json.loads(json_str)
                    print(f"\n成功解析完整反馈JSON: {feedback_data}\n")
                except json.JSONDecodeError as e:
                    print(f"\n解析完整反馈JSON失败: {e}\n")

            # 如果无法提取完整的JSON，尝试使用正则表达式提取
            if not feedback_data:
                json_patterns = [
                    r'```+json\s*\n(.+?)\n\s*```+',  # Markdown代码块格式
                    r'````json\s*\n(.+?)\n\s*````',  # 四个反引号
                    r'`(.+?)`',  # 反引号内的内容
                    r'\{\s*"is_correct":[\s\S]+?\}',  # 直接匹配包含is_correct字段的JSON对象
                    r'\{[\s\S]+?\}'  # 匹配任何JSON对象
                ]

                for pattern in json_patterns:
                    json_match = re.search(pattern, output, re.DOTALL)
                    if json_match:
                        # 处理不同的正则表达式匹配结果
                        if len(json_match.groups()) > 0:
                            # 如果有捕获组，使用第一个捕获组
                            json_str = json_match.group(1).strip()
                        else:
                            # 否则使用整个匹配
                            json_str = json_match.group(0).strip()

                        print(f"\n找到的反馈JSON字符串: {json_str}\n")
                        try:
                            feedback_data = json.loads(json_str)
                            print(f"\n成功解析反馈JSON: {feedback_data}\n")
                            break
                        except json.JSONDecodeError as je:
                            print(f"\n反馈JSON解析错误: {je}\n")

                            # 尝试修复JSON字符串并重新解析
                            try:
                                # 尝试删除可能的干扰字符
                                cleaned_str = re.sub(r'[\n\r\t]', '', json_str)
                                # 尝试修复常见的JSON错误
                                cleaned_str = cleaned_str.replace("'", '"')
                                # 尝试解析清理后的字符串
                                feedback_data = json.loads(cleaned_str)
                                print(f"\n成功解析清理后的反馈JSON: {feedback_data}\n")
                                break
                            except Exception as clean_e:
                                print(f"\n清理后的反馈JSON解析仍然失败: {clean_e}\n")
                                continue

            # 如果仍然没有找到JSON，尝试使用更宽松的方法
            if not feedback_data:
                # 尝试在文本中寻找大括号对
                start_idx = output.find('{')
                if start_idx != -1:
                    # 找到开始大括号，现在寻找匹配的结束大括号
                    brace_count = 1
                    for i in range(start_idx + 1, len(output)):
                        if output[i] == '{':
                            brace_count += 1
                        elif output[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                # 找到匹配的结束大括号
                                json_str = output[start_idx:i+1]
                                print(f"\n手动提取的反馈JSON字符串: {json_str}\n")
                                try:
                                    feedback_data = json.loads(json_str)
                                    print(f"\n成功解析手动提取的反馈JSON: {feedback_data}\n")
                                    break
                                except json.JSONDecodeError as je:
                                    print(f"\n手动提取的反馈JSON解析错误: {je}\n")
                                break

            # 如果没有找到JSON，回退到原始方法
            if not feedback_data:
                print("\n反馈代理方法失败，回退到原始方法\n")
                response = await self.llm.ainvoke(prompt)
                print(f"\n反馈LLM响应: {response.content}\n")
                feedback_data = json.loads(response.content)
                print(f"\n成功解析反馈LLM响应: {feedback_data}\n")

            # 如果所有方法都失败，创建一个默认的反馈数据
            if not feedback_data:
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
