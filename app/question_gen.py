"""
问题生成模块，使用LLM生成数学问题
"""
import uuid
import json
import re
import sympy as sp
from typing import Dict, List, Any, Union

def preprocess_expression(expression):
    """
    预处理数学表达式，修复常见错误

    Args:
        expression (str): 要预处理的数学表达式

    Returns:
        str: 预处理后的表达式
    """
    # 替换中文括号为英文括号
    expression = expression.replace('（', '(').replace('）', ')')

    # 替换中文逗号为英文逗号
    expression = expression.replace('，', ',')

    # 替换全角符号为半角符号
    expression = expression.replace('＋', '+').replace('－', '-')
    expression = expression.replace('＊', '*').replace('／', '/')
    expression = expression.replace('＝', '=')

    # 修复常见的输入错误
    # 替换 sinθ 为 sin(theta)
    expression = re.sub(r'sinθ', 'sin(theta)', expression)
    expression = re.sub(r'cosθ', 'cos(theta)', expression)
    expression = re.sub(r'tanθ', 'tan(theta)', expression)

    # 替换 sinx 为 sin(x)
    expression = re.sub(r'sin([a-zA-Z])', r'sin(\1)', expression)
    expression = re.sub(r'cos([a-zA-Z])', r'cos(\1)', expression)
    expression = re.sub(r'tan([a-zA-Z])', r'tan(\1)', expression)

    # 替换 sin^2(x) 为 sin(x)**2
    expression = re.sub(r'sin\^([0-9]+)\(([^\)]+)\)', r'sin(\2)**\1', expression)
    expression = re.sub(r'cos\^([0-9]+)\(([^\)]+)\)', r'cos(\2)**\1', expression)
    expression = re.sub(r'tan\^([0-9]+)\(([^\)]+)\)', r'tan(\2)**\1', expression)

    # 替换 x^2 为 x**2
    expression = re.sub(r'([a-zA-Z0-9])\^([0-9]+)', r'\1**\2', expression)

    # 替换 2x 为 2*x
    expression = re.sub(r'([0-9])([a-zA-Z])', r'\1*\2', expression)

    return expression

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent

from app.config import settings
from app.models import (
    Question, MultipleChoiceQuestion, FillInBlankQuestion,
    CalculationQuestion, QuestionType, DifficultyLevel,
    MathDomain, Option
)

# 数学工具函数
import re

def calculate_expression(expression: str) -> str:
    """
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
    try:
        # 去除表达式两端的引号，如果有的话
        expression = expression.strip()
        if (expression.startswith('"') and expression.endswith('"')) or \
           (expression.startswith("'") and expression.endswith("'")):
            expression = expression[1:-1]

        # 预处理表达式，修复常见错误
        expression = preprocess_expression(expression)
        print(f"\n预处理后的表达式: {expression}\n")

        # Define common symbolic variables
        x, y, z = sp.symbols('x y z')
        pi = sp.pi
        e = sp.E

        # Create local namespace containing all sympy functions and symbolic variables
        locals_dict = {**sp.__dict__, 'x': x, 'y': y, 'z': z, 'pi': pi, 'e': e}

        # Special handling for various types of expressions

        # 1. Handle complex integral expressions
        if "integrate" in expression and ("oo" in expression or "-oo" in expression):
            return handle_complex_integration(expression, locals_dict)

        # 2. Handle system of equations solving expressions
        elif "solve(" in expression and "[" in expression and "]" in expression:
            return handle_equation_solving(expression, locals_dict)

        # 3. Handle matrix eigenvalue calculation expressions
        elif "eigenvals" in expression or "eigenvects" in expression:
            return handle_matrix_eigenvalues(expression, locals_dict)

        # 4. Handle basic arithmetic expressions
        elif all(c in "0123456789+-*/()^. " for c in expression):
            # 尝试直接使用 sympify 计算
            try:
                result = sp.sympify(expression)
                return str(result)
            except Exception:
                # 如果失败，尝试使用 eval
                result = eval(expression)
                return str(result)

        # 5. General expression calculation
        else:
            # 尝试使用 sympify 计算
            try:
                result = sp.sympify(expression, locals=locals_dict)
                return str(result)
            except Exception:
                # 如果失败，尝试使用 eval
                result = eval(expression, globals(), locals_dict)

            # Process based on result type
            return format_result(result)

    except Exception as e:
        return f"Error: {e}"

def handle_complex_integration(expression, locals_dict):
    """Handle complex integral expressions"""
    try:
        # Check if it's an infinite integral
        if "-oo" in expression or "oo" in expression:
            # Try symbolic computation
            expr = eval(expression, globals(), locals_dict)

            # If it's an integral object but not computed
            if isinstance(expr, sp.Integral):
                try:
                    # Try to perform the integral
                    result = expr.doit()

                    # Try to compute numerical result
                    try:
                        numerical = result.evalf()
                        return str(numerical)
                    except:
                        return str(result)
                except Exception as e:
                    # If symbolic integration fails, try alternative methods
                    try:
                        # Extract integral expression information
                        match = re.search(r"integrate\((.*?), \((.*?), (.*?), (.*?)\)\)", expression)
                        if match:
                            integrand, var, lower, upper = match.groups()

                            # For infinite integrals, use finite approximation
                            if (lower == "-oo" or lower == "oo") or (upper == "oo" or upper == "-oo"):
                                # Replace infinity with a large value
                                if lower == "-oo":
                                    lower = "-100"
                                elif lower == "oo":
                                    lower = "100"

                                if upper == "-oo":
                                    upper = "-100"
                                elif upper == "oo":
                                    upper = "100"

                                # Build finite range integral expression
                                finite_expr = f"integrate({integrand}, ({var}, {lower}, {upper}))"
                                result = eval(finite_expr, globals(), locals_dict)

                                try:
                                    numerical = result.evalf()
                                    return f"Approximate numerical result: {numerical} (using finite range integral)"
                                except:
                                    return f"Approximate result: {result} (using finite range integral)"
                    except Exception as e2:
                        return f"Integration error: {e}, finite approximation failed: {e2}"

            # Try to compute result directly
            try:
                numerical = expr.evalf()
                return str(numerical)
            except:
                return str(expr)

        # Regular integral
        result = eval(expression, globals(), locals_dict)
        return format_result(result)

    except Exception as e:
        return f"Integration error: {e}"

def handle_equation_solving(expression, locals_dict):
    """Handle system of equations solving expressions"""
    try:
        # Compute result
        result = eval(expression, globals(), locals_dict)

        # Format result
        return format_result(result)

    except Exception as e:
        return f"Equation solving error: {e}"

def handle_matrix_eigenvalues(expression, locals_dict):
    """Handle matrix eigenvalue calculation expressions"""
    try:
        # Extract matrix expression
        matrix_expr = expression.split(".eigen")[0]
        operation = "eigenvals" if "eigenvals" in expression else "eigenvects"

        # Compute matrix
        matrix = eval(matrix_expr, globals(), locals_dict)

        # Compute eigenvalues or eigenvectors
        if operation == "eigenvals":
            result = matrix.eigenvals()
        else:
            result = matrix.eigenvects()

        # Format result
        return format_result(result)

    except Exception as e:
        return f"Matrix eigenvalue calculation error: {e}"

def format_result(result):
    """Format output based on result type"""
    try:
        # Handle dictionary type results (e.g., eigenvalues)
        if isinstance(result, dict):
            formatted = "{"
            for key, value in result.items():
                # Try numerical computation
                try:
                    key_eval = key.evalf()
                except:
                    key_eval = key

                formatted += f"{key_eval}: {value}, "

            if formatted.endswith(", "):
                formatted = formatted[:-2]

            formatted += "}"
            return formatted

        # Handle list type results (e.g., solutions to equations)
        elif isinstance(result, list):
            formatted = "["
            for item in result:
                # Check if it's a tuple (e.g., coordinate points)
                if isinstance(item, tuple):
                    coords = []
                    for val in item:
                        # Try numerical computation
                        try:
                            val_eval = val.evalf()
                            coords.append(str(val_eval))
                        except:
                            coords.append(str(val))

                    formatted += "(" + ", ".join(coords) + "), "
                else:
                    # Try numerical computation
                    try:
                        item_eval = item.evalf()
                        formatted += f"{item_eval}, "
                    except:
                        formatted += f"{item}, "

            if formatted.endswith(", "):
                formatted = formatted[:-2]

            formatted += "]"
            return formatted

        # Other types of results
        else:
            # Try numerical computation
            try:
                return str(result.evalf())
            except:
                return str(result)

    except Exception as e:
        return f"Result formatting error: {e}, original result: {result}"

# 简单工具函数
def solve_equation(equation_str: str) -> str:
    """解方程式，如 '2*x + 5 = 15' 或 'x^2 - 4 = 0'"""
    try:
        # 检查是否包含等号
        if '=' in equation_str:
            # 分离等号两边
            left_side, right_side = equation_str.split('=')
            # 将等式转换为标准形式：左边 - 右边 = 0
            equation = f"({left_side.strip()}) - ({right_side.strip()})"
        else:
            # 如果没有等号，假设输入已经是标准形式
            equation = equation_str

        x = sp.symbols('x')
        expr = sp.sympify(equation)
        solution = sp.solve(expr, x)
        return str(solution)
    except Exception as e:
        return f"无法解决方程式: {str(e)}"

def calculate_derivative(expression_str: str) -> str:
    """计算导数，默认对x求导。如需对其他变量求导，请在表达式中指定，如 'diff(x^2 + y^2, y)'"""
    try:
        # 检查是否已经是diff函数调用
        if expression_str.strip().startswith('diff('):
            # 直接计算
            x, y, z = sp.symbols('x y z')
            result = eval(expression_str, {'diff': sp.diff, 'x': x, 'y': y, 'z': z, **sp.__dict__})
            return str(result)
        else:
            # 默认对x求导
            x = sp.symbols('x')
            expression = sp.sympify(expression_str)
            derivative = sp.diff(expression, x)
            return str(derivative)
    except Exception as e:
        return f"无法计算导数: {str(e)}"

def calculate_integral(expression_str: str) -> str:
    """计算积分，默认对x积分。如需对其他变量积分或计算定积分，请在表达式中指定，如 'integrate(x^2 + y^2, y)' 或 'integrate(sin(x), (x, 0, pi))'"""
    try:
        # 检查是否已经是integrate函数调用
        if expression_str.strip().startswith('integrate('):
            # 直接计算
            x, y, z = sp.symbols('x y z')
            result = eval(expression_str, {'integrate': sp.integrate, 'x': x, 'y': y, 'z': z, **sp.__dict__})
            return str(result)
        else:
            # 默认对x积分
            x = sp.symbols('x')
            expression = sp.sympify(expression_str)
            integral = sp.integrate(expression, x)
            return str(integral)
    except Exception as e:
        return f"无法计算积分: {str(e)}"

class QuestionGenerator:
    """数学问题生成器"""

    def __init__(self):
        """初始化问题生成器"""
        self.llm = ChatOpenAI(
            openai_api_base=settings.OPENAI_API_BASE,
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.LLM_MODEL_NAME,
            temperature=0.7
        )

        # 创建数学工具
        self.calculate_tool = StructuredTool.from_function(
            name="calculate_expression",
            func=calculate_expression,
            description="""计算数学表达式，使用 `sympy` 中的 `sympify` 函数实现,解析并计算输入的数学表达式字符串,支持直接调用SymPy函数（自动识别x,y,z为符号变量）

参数:
    expression (str): 数学表达式，例如 "223 - 344 * 6" 或 "sin(pi/2) + log(10)"。
示例表达式：
    "2 + 3*5"                          # 基础算术 → 17
    "expand((x + 1)**2)"               # 展开 → x² + 2x + 1
    "diff(sin(x), x)"                  # 导数 → cos(x)
    "integrate(exp(x), (x, 0, 1))"      # 定积分 → E - 1
    "solve(x**2 - 4, x)"               # 解方程 → [-2, 2]
    "limit(tan(x)/x, x, 0)"            # 极限 → 1
    "Sum(k, (k, 1, 10)).doit()"        # 求和 → 55
    "Matrix([[1, 2], [3, 4]]).inv()"   # 矩阵逆 → [[-2, 1], [3/2, -1/2]]
    "simplify((x**2 - 1)/(x + 1))"     # 化简 → x - 1
    "factor(x**2 - 2*x - 15)"          # 因式分解 → (x - 5)(x + 3)
    "series(cos(x), x, 0, 4)"          # 泰勒展开 → 1 - x²/2 + x⁴/24 + O(x⁴)
返回:
    str: 计算结果。如果表达式无法解析或计算失败，返回错误信息（str）。"""
        )

        # 创建其他数学工具
        self.basic_math_tools = [
            StructuredTool.from_function(
                func=solve_equation,
                name="solve_equation",
                description="解决数学方程式"
            ),
            StructuredTool.from_function(
                func=calculate_derivative,
                name="calculate_derivative",
                description="计算函数的导数"
            ),
            StructuredTool.from_function(
                func=calculate_integral,
                name="calculate_integral",
                description="计算函数的积分"
            )
        ]

        # 合并所有工具
        self.math_tools = [self.calculate_tool] + self.basic_math_tools

        # 创建代理
        self.agent = initialize_agent(
            tools=self.math_tools,
            llm=self.llm,
            agent="zero-shot-react-description",  # 使用支持多工具的 Agent
            verbose=True
        )

        # 创建问题生成提示模板
        self.multiple_choice_template = ChatPromptTemplate.from_template(
            """你是一个专业的数学老师，请根据以下要求生成一道高质量的数学选择题：

            领域: {domain}
            难度: {difficulty}

            请生成一道选择题，包含4个选项(A, B, C, D)，并提供正确答案和详细解析。

            输出格式必须是有效的JSON，包含以下字段：
            - content: 问题内容
            - options: 选项列表，每个选项包含id(A,B,C,D)和content
            - answer: 正确答案(A,B,C,D)
            - explanation: 详细解析

            确保生成的问题在指定难度和领域内，并且问题清晰明确。
            """
        )

        self.fill_in_blank_template = ChatPromptTemplate.from_template(
            """你是一个专业的数学老师，请根据以下要求生成一道高质量的数学填空题：

            领域: {domain}
            难度: {difficulty}

            请生成一道填空题，并提供正确答案和详细解析。

            输出格式必须是有效的JSON，包含以下字段：
            - content: 问题内容，使用"_____"表示填空处
            - answer: 正确答案
            - explanation: 详细解析

            确保生成的问题在指定难度和领域内，并且问题清晰明确。
            """
        )

        self.calculation_template = ChatPromptTemplate.from_template(
            """你是一个专业的数学老师，请根据以下要求生成一道高质量的数学计算题：

            领域: {domain}
            难度: {difficulty}

            请生成一道计算题，并提供正确答案和详细解析。

            输出格式必须是有效的JSON，包含以下字段：
            - content: 问题内容
            - answer: 正确答案(包含计算过程)
            - explanation: 详细解析

            确保生成的问题在指定难度和领域内，并且问题清晰明确。
            """
        )

    async def generate_question(
        self,
        question_type: QuestionType,
        difficulty: DifficultyLevel,
        domain: MathDomain
    ) -> Union[MultipleChoiceQuestion, FillInBlankQuestion, CalculationQuestion]:
        """生成数学问题"""

        # 根据问题类型选择不同的提示模板
        if question_type == QuestionType.MULTIPLE_CHOICE:
            template = self.multiple_choice_template
        elif question_type == QuestionType.FILL_IN_BLANK:
            template = self.fill_in_blank_template
        else:  # 计算题
            template = self.calculation_template

        # 准备提示参数
        prompt_args = {
            "difficulty": difficulty.value,
            "domain": domain.value
        }

        # 生成提示
        prompt = template.format_messages(**prompt_args)

        # 生成随机数作为种子，确保每次生成不同的问题
        import random
        import time
        random_seed = int(time.time()) % 10000

        # 使用代理生成问题
        agent_prompt = """
        你是一个专业的数学教师，请生成一道{domain}领域的{difficulty}难度的{question_type}。

        请使用中文生成问题、选项和解析，不要使用英文。

        请生成原创的、有趣的、多样化的问题。不要生成简单的“2x + 3 = 7”这类基础问题。根据难度级别，生成相应复杂度的问题。

        这是随机种子 {seed}，请使用它来生成不同的问题。

        如果是选择题，请生成以下格式的JSON：
        {{
            "content": "问题内容（中文）",
            "options": [
                {{
                    "id": "A",
                    "content": "选项A的内容（中文）"
                }},
                {{
                    "id": "B",
                    "content": "选项B的内容（中文）"
                }},
                {{
                    "id": "C",
                    "content": "选项C的内容（中文）"
                }},
                {{
                    "id": "D",
                    "content": "选项D的内容（中文）"
                }}
            ],
            "answer": "A",
            "explanation": "详细解析（中文）"
        }}

        如果是填空题或计算题，请生成以下格式的JSON：
        {{
            "content": "问题内容（中文）",
            "answer": "正确答案",
            "explanation": "详细解析（中文）"
        }}

        如果需要计算，请使用数学工具来计算答案。在使用计算工具时，请仔细检查输入格式，确保输入的表达式是正确的。
        输出必须是有效的JSON格式。
        再次强调，请使用中文生成所有内容。
        """.format(
            domain=domain.value,
            difficulty=difficulty.value,
            question_type=question_type.value,
            seed=random_seed
        )

        # 调用代理
        response = await self.agent.ainvoke({"input": agent_prompt})

        # 从代理响应中提取JSON
        question_data = None
        try:
            # 打印完整的代理响应以进行调试
            print(f"\n\n代理响应: {response}\n\n")

            # 尝试从代理输出中提取JSON
            output = response.get("output", "")
            print(f"\n提取的输出: {output}\n")

            # 寻找可能的JSON字符串
            import re
            # 尝试不同的正则表达式来提取JSON
            json_patterns = [
                r'\{[^\{\}]*\{[^\{\}]*\}[^\{\}]*\}',  # 嵌套的JSON
                r'\{[^\{\}]*\}',  # 简单的JSON
                r'\{.*\}'  # 贪婪匹配
            ]

            for pattern in json_patterns:
                json_match = re.search(pattern, output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    print(f"\n找到的JSON字符串: {json_str}\n")
                    try:
                        question_data = json.loads(json_str)
                        print(f"\n成功解析JSON: {question_data}\n")
                        break
                    except json.JSONDecodeError as je:
                        print(f"\nJSON解析错误: {je}\n")
                        continue

            # 如果仍然没有找到JSON，尝试使用更宽松的方法
            if not question_data:
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
                                print(f"\n手动提取的JSON字符串: {json_str}\n")
                                try:
                                    question_data = json.loads(json_str)
                                    print(f"\n成功解析手动提取的JSON: {question_data}\n")
                                    break
                                except json.JSONDecodeError as je:
                                    print(f"\n手动提取的JSON解析错误: {je}\n")
                                break
        except Exception as e:
            print(f"从代理响应中提取JSON时出错: {e}")

        # 如果代理方法失败，回退到原始方法
        if not question_data:
            print("\n代理方法失败，回退到原始方法\n")
            try:
                response = await self.llm.ainvoke(prompt)
                print(f"\nLLM响应: {response.content}\n")
                question_data = json.loads(response.content)
                print(f"\n成功解析LLM响应: {question_data}\n")
            except Exception as e:
                print(f"解析LLM响应时出错: {e}")
                print(f"原始响应: {response}")

                # 如果所有方法都失败，创建一个默认的问题数据
                print("\n创建默认问题数据\n")
                if question_type == QuestionType.MULTIPLE_CHOICE:
                    question_data = {
                        "content": f"这是一道{domain.value}领域的{difficulty.value}难度选择题",
                        "options": [
                            {"id": "A", "content": "选项A"},
                            {"id": "B", "content": "选项B"},
                            {"id": "C", "content": "选项C"},
                            {"id": "D", "content": "选项D"}
                        ],
                        "answer": "A",
                        "explanation": "这是默认的解析"
                    }
                elif question_type == QuestionType.FILL_IN_BLANK:
                    question_data = {
                        "content": f"这是一道{domain.value}领域的{difficulty.value}难度填空题，请填写_____",
                        "answer": "答案",
                        "explanation": "这是默认的解析"
                    }
                else:  # 计算题
                    question_data = {
                        "content": f"这是一道{domain.value}领域的{difficulty.value}难度计算题",
                        "answer": "答案",
                        "explanation": "这是默认的解析"
                    }

        # 生成唯一ID
        question_id = str(uuid.uuid4())

        # 根据问题类型创建不同的问题对象
        try:
            # 处理可能的不同格式
            # 检查是否有content字段，如果没有，尝试使用question字段
            content = question_data.get("content", question_data.get("question", ""))

            # 检查是否有explanation字段，如果没有，使用空字符串
            explanation = question_data.get("explanation", "")

            # 检查是否有answer字段，如果没有，尝试使用correct_answer字段
            answer = question_data.get("answer", question_data.get("correct_answer", ""))

            if question_type == QuestionType.MULTIPLE_CHOICE:
                # 处理选项格式
                options_data = question_data.get("options", [])
                options = []

                # 处理不同的选项格式
                if options_data and isinstance(options_data, list):
                    # 检查选项格式
                    if options_data and isinstance(options_data[0], dict):
                        # 如果是字典列表，检查是否有id和content字段
                        if "id" in options_data[0] and "content" in options_data[0]:
                            # 标准格式
                            options = [Option(id=opt["id"], content=opt["content"]) for opt in options_data]
                        else:
                            # 可能是{"A": "content"}格式
                            for opt_dict in options_data:
                                for key, value in opt_dict.items():
                                    options.append(Option(id=key, content=value))
                    else:
                        # 如果是简单列表，使用A, B, C, D作为id
                        option_ids = ["A", "B", "C", "D"]
                        for i, opt_content in enumerate(options_data):
                            if i < len(option_ids):
                                options.append(Option(id=option_ids[i], content=str(opt_content)))

                # 如果没有选项，创建一些默认选项
                if not options:
                    options = [
                        Option(id="A", content="选项A"),
                        Option(id="B", content="选项B"),
                        Option(id="C", content="选项C"),
                        Option(id="D", content="选项D")
                    ]

                return MultipleChoiceQuestion(
                    id=question_id,
                    type=question_type,
                    difficulty=difficulty,
                    domain=domain,
                    content=content,
                    options=options,
                    answer=answer,
                    explanation=explanation
                )

            elif question_type == QuestionType.FILL_IN_BLANK:
                return FillInBlankQuestion(
                    id=question_id,
                    type=question_type,
                    difficulty=difficulty,
                    domain=domain,
                    content=content,
                    answer=answer,
                    explanation=explanation
                )

            else:  # 计算题
                return CalculationQuestion(
                    id=question_id,
                    type=question_type,
                    difficulty=difficulty,
                    domain=domain,
                    content=content,
                    answer=answer,
                    explanation=explanation
                )
        except Exception as e:
            # 处理解析错误
            print(f"创建问题对象时出错: {e}")
            print(f"问题数据: {question_data}")
            raise Exception(f"生成问题失败: {e}")
