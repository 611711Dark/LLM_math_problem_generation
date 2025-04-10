# Math Problem Practice System

English | [中文](README.md)

This is a web application that uses LLM (Large Language Model) to generate math problems, supporting multiple-choice, fill-in-the-blank, and calculation questions, and providing AI feedback. The system generates questions and feedback in Chinese, suitable for Chinese-speaking users learning mathematics. The system supports LaTeX format for mathematical formulas, providing clear and beautiful mathematical expression display.

## Screenshots

### Question Page

![Page_effect](Page_effect.png)

### Answer Feedback

![Answer_effect](Answer_effect.png)

## Features

- Uses LLM to generate formatted math problems with Chinese output
- Supports multiple question types: multiple-choice, fill-in-the-blank, and calculation
- Supports different difficulty levels (easy, medium, difficult) and math domains (algebra, geometry, trigonometry, calculus, probability & statistics)
- Provides detailed Chinese AI feedback and improvement suggestions after user answers
- Responsive web interface, adaptable to various devices
- Integrates powerful mathematical calculation tools, supporting complex expression calculations
- Intelligent error handling, capable of fixing common mathematical expression input errors
- Generates different random questions each time, providing diverse practice content
- Supports LaTeX format for mathematical formulas, rendered by KaTeX, providing clear and beautiful mathematical expression display
- Simplified JSON parsing logic, improving system stability and efficiency
- Optimized calculation tool usage, reducing unnecessary calculation calls

## Tech Stack

- Backend: FastAPI, Langchain, DeepSeek API
- Frontend: HTML, CSS, JavaScript, Bootstrap
- Math Formula Rendering: KaTeX
- Mathematical Computation: SymPy, supporting expansion, factorization, differentiation, integration, matrix operations, etc.
- Data Models: Pydantic, supporting type checking and data validation
- Template Engine: Jinja2, for rendering HTML templates

## Installation and Running

1. Ensure Python 3.8+ is installed

2. Clone the repository
   ```bash
   git clone https://github.com/yourusername/math_questions.git
   cd math_questions
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Configure API key
   Configure your DeepSeek API key in the `app/config.py` file. By default, the system tries to read the key from the `DEEPSEEK_API_KEY` environment variable. You can set the environment variable or modify it directly in the configuration file.
   ```bash
   # Linux/Mac
   export DEEPSEEK_API_KEY="your-api-key"

   # Windows
   set DEEPSEEK_API_KEY=your-api-key
   ```

5. Run the application
   ```bash
   python run.py
   ```

6. Visit http://localhost:8000 in your browser

## Project Structure

```
math_questions/
├── app/                  # Application code
│   ├── __init__.py
│   ├── main.py           # FastAPI main application
│   ├── models.py         # Data models
│   ├── question_gen.py   # Question generator
│   ├── feedback_gen.py   # Feedback generator
│   └── config.py         # Configuration file
├── static/               # Static files
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── main.js
├── templates/            # HTML templates
│   └── index.html
├── requirements.txt      # Dependencies list
├── README.md             # Project description (Chinese)
├── README_EN.md          # Project description (English)
└── run.py                # Startup script
```

## Usage Instructions

1. Select the question type, difficulty level, and math domain on the homepage
   - **Question Type**: Multiple-choice, fill-in-the-blank, or calculation
   - **Difficulty Level**: Easy, medium, or difficult
   - **Math Domain**: Algebra, geometry, trigonometry, calculus, or probability & statistics

2. Click the "Generate Question" button to generate a new question
   - Multiple-choice questions will display four options, select the correct answer
   - Fill-in-the-blank questions require entering the answer in the text box
   - Calculation questions require entering the calculation result in the text box

3. After answering the question, click the "Submit Answer" button

4. The system will generate detailed Chinese AI feedback, including:
   - Whether the answer is correct
   - Detailed explanation
   - Improvement suggestions (if the answer is incorrect)

5. View the feedback and click "Generate New Question" to continue practicing

## Special Features

1. **Mathematical Expression Error Handling**: The system can automatically fix common mathematical expression input errors, such as:
   - Converting Chinese brackets and punctuation marks
   - Fixing sinx to sin(x) format
   - Converting x^2 to x**2 format
   - Converting 2x to 2*x format

2. **Random Question Generation**: Generates different random questions each time, avoiding repetitive practice of the same questions.

3. **Chinese Feedback**: The system provides detailed Chinese feedback and explanations, helping users better understand the problems.

4. **LaTeX Mathematical Formula Support**: The system uses LaTeX format for mathematical formulas in questions and feedback, such as $f(x) = x^2$ and $\frac{1}{2}$, providing clear and beautiful mathematical expression display.

5. **Intelligent Calculation Tool Usage**: The system intelligently determines when to use calculation tools, avoiding using tools for simple calculations, improving efficiency.

6. **Simplified JSON Parsing**: Uses direct text extraction instead of JSON parsing, improving system stability and efficiency.

## Notes

- DeepSeek API is used by default, configured in config.py, with the API key read from the DEEPSEEK_API_KEY environment variable
- Generating questions and feedback may take some time, please be patient
- Mathematical formulas use LaTeX format, rendered by KaTeX, supporting various complex mathematical expressions
- Uses LangChain for mathematical calculations, optimized to only use tools for complex calculations
- Integrates advanced mathematical calculation tools, supporting complex expression calculations, infinite integrals, matrix operations, etc.
- The system uses direct text extraction instead of JSON parsing, improving stability
- If you encounter problems, you can check the console logs for details

## Contributing

Welcome to submit issues and improvement suggestions! If you want to contribute code, please create an issue first to discuss what you want to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
