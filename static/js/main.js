// 主JavaScript文件

// 全局变量
let currentQuestion = null;
let selectedOption = null;

// DOM元素
const questionForm = document.getElementById('questionForm');
const questionLoadingIndicator = document.getElementById('questionLoadingIndicator');
const feedbackLoadingIndicator = document.getElementById('feedbackLoadingIndicator');
const questionContainer = document.getElementById('questionContainer');
const feedbackContainer = document.getElementById('feedbackContainer');
const optionsContainer = document.getElementById('optionsContainer');
const answerInputContainer = document.getElementById('answerInputContainer');
const submitAnswerBtn = document.getElementById('submitAnswerBtn');

// 初始化
document.addEventListener('DOMContentLoaded', async () => {
    // 加载问题类型
    await loadQuestionTypes();

    // 加载难度级别
    await loadDifficultyLevels();

    // 加载数学领域
    await loadMathDomains();

    // 表单提交事件
    questionForm.addEventListener('submit', handleFormSubmit);

    // 提交答案按钮事件
    submitAnswerBtn.addEventListener('click', handleSubmitAnswer);

    // 返回顶部按钮事件
    const backToTopBtn = document.getElementById('backToTopBtn');

    // 滚动事件监听
    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            backToTopBtn.classList.add('show');
        } else {
            backToTopBtn.classList.remove('show');
        }
    });

    // 点击返回顶部
    backToTopBtn.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    // 初始化KaTeX
    renderMathInElement(document.body, {
        delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false}
        ]
    });
});

// 加载问题类型
async function loadQuestionTypes() {
    try {
        const response = await fetch('/api/question-types');
        const types = await response.json();

        const selectElement = document.getElementById('questionType');
        types.forEach(type => {
            const option = document.createElement('option');
            option.value = type;
            option.textContent = type;
            selectElement.appendChild(option);
        });
    } catch (error) {
        console.error('加载问题类型失败:', error);
    }
}

// 加载难度级别
async function loadDifficultyLevels() {
    try {
        const response = await fetch('/api/difficulty-levels');
        const levels = await response.json();

        const selectElement = document.getElementById('difficulty');
        levels.forEach(level => {
            const option = document.createElement('option');
            option.value = level;
            option.textContent = level;
            selectElement.appendChild(option);
        });
    } catch (error) {
        console.error('加载难度级别失败:', error);
    }
}

// 加载数学领域
async function loadMathDomains() {
    try {
        const response = await fetch('/api/math-domains');
        const domains = await response.json();

        const selectElement = document.getElementById('domain');
        domains.forEach(domain => {
            const option = document.createElement('option');
            option.value = domain;
            option.textContent = domain;
            selectElement.appendChild(option);
        });
    } catch (error) {
        console.error('加载数学领域失败:', error);
    }
}

// 处理表单提交
async function handleFormSubmit(event) {
    event.preventDefault();

    // 获取表单数据
    const questionType = document.getElementById('questionType').value;
    const difficulty = document.getElementById('difficulty').value;
    const domain = document.getElementById('domain').value;

    // 显示问题加载指示器
    questionLoadingIndicator.classList.remove('d-none');
    questionContainer.classList.add('d-none');
    feedbackContainer.classList.add('d-none');

    try {
        // 发送请求生成问题
        const response = await fetch('/api/questions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                type: questionType,
                difficulty: difficulty,
                domain: domain
            })
        });

        if (!response.ok) {
            throw new Error('生成问题失败');
        }

        // 获取问题数据
        currentQuestion = await response.json();

        // 显示问题
        displayQuestion(currentQuestion);
    } catch (error) {
        console.error('生成问题失败:', error);
        alert('生成问题失败，请重试');
    } finally {
        // 隐藏问题加载指示器
        questionLoadingIndicator.classList.add('d-none');
    }
}

// 显示问题
function displayQuestion(question) {
    // 设置问题内容
    // 处理问题内容，保留换行符但去除多余的空格
    let cleanContent = question.content.replace(/([^\n])\s+/g, '$1 ');
    document.getElementById('questionContent').innerHTML = cleanContent;
    document.getElementById('questionDomain').textContent = question.domain;
    document.getElementById('questionDifficulty').textContent = question.difficulty;

    // 渲染问题内容中的LaTeX
    setTimeout(() => {
        renderMathInElement(document.getElementById('questionContent'), {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ],
            throwOnError: false
        });
    }, 0);

    // 根据问题类型显示不同的UI
    if (question.type === '选择题') {
        // 显示选择题选项
        optionsContainer.classList.remove('d-none');
        answerInputContainer.classList.add('d-none');

        // 清空选项列表
        const optionsList = document.getElementById('optionsList');
        optionsList.innerHTML = '';

        // 添加选项
        question.options.forEach(option => {
            // 处理选项内容，去除多余的空格和换行符
            let cleanContent = option.content.trim().replace(/\s+/g, ' ');

            const optionElement = document.createElement('button');
            optionElement.type = 'button';
            optionElement.className = 'list-group-item list-group-item-action';
            optionElement.innerHTML = `<strong>${option.id}.</strong> ${cleanContent}`;
            optionElement.dataset.optionId = option.id;

            // 选项点击事件
            optionElement.addEventListener('click', () => {
                // 移除其他选项的选中状态
                document.querySelectorAll('#optionsList .list-group-item').forEach(item => {
                    item.classList.remove('selected');
                });

                // 添加选中状态
                optionElement.classList.add('selected');

                // 记录选中的选项
                selectedOption = option.id;
            });

            optionsList.appendChild(optionElement);

            // 渲染选项中的LaTeX
            setTimeout(() => {
                renderMathInElement(optionElement, {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '$', right: '$', display: false}
                    ],
                    throwOnError: false
                });
            }, 0);
        });
    } else {
        // 显示填空题或计算题输入框
        optionsContainer.classList.add('d-none');
        answerInputContainer.classList.remove('d-none');

        // 清空输入框
        document.getElementById('userAnswer').value = '';
    }

    // 显示问题容器
    questionContainer.classList.remove('d-none');

    // 渲染数学公式
    renderMathInElement(document.getElementById('questionContent'), {
        delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false}
        ]
    });
}

// 处理提交答案
async function handleSubmitAnswer() {
    // 获取用户答案
    let userAnswer = '';

    if (currentQuestion.type === '选择题') {
        if (!selectedOption) {
            alert('请选择一个选项');
            return;
        }
        userAnswer = selectedOption;
    } else {
        userAnswer = document.getElementById('userAnswer').value.trim();
        if (!userAnswer) {
            alert('请输入你的答案');
            return;
        }
    }

    // 显示反馈加载指示器
    feedbackLoadingIndicator.classList.remove('d-none');

    try {
        // 发送请求提交答案
        const response = await fetch('/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question_id: currentQuestion.id,
                answer: userAnswer
            })
        });

        if (!response.ok) {
            throw new Error('提交答案失败');
        }

        // 获取反馈数据
        const feedback = await response.json();

        // 显示反馈
        displayFeedback(feedback);
    } catch (error) {
        console.error('提交答案失败:', error);
        alert('提交答案失败，请重试');
    } finally {
        // 隐藏反馈加载指示器
        feedbackLoadingIndicator.classList.add('d-none');
    }
}

// 显示反馈
function displayFeedback(feedback) {
    // 设置反馈标题样式
    const feedbackHeader = document.getElementById('feedbackHeader');
    if (feedback.is_correct) {
        feedbackHeader.className = 'card-header bg-success text-white';
        feedbackHeader.innerHTML = '<h5 class="mb-0"><i class="bi bi-check-circle-fill me-2"></i>回答正确</h5>';
    } else {
        feedbackHeader.className = 'card-header bg-danger text-white';
        feedbackHeader.innerHTML = '<h5 class="mb-0"><i class="bi bi-x-circle-fill me-2"></i>回答错误</h5>';
    }

    // 设置反馈内容
    document.getElementById('userAnswerDisplay').textContent = feedback.user_answer;
    document.getElementById('correctAnswerDisplay').textContent = feedback.correct_answer;
    document.getElementById('explanationDisplay').innerHTML = feedback.explanation;

    // 设置改进建议
    const improvementContainer = document.getElementById('improvementContainer');
    if (feedback.improvement_suggestions && !feedback.is_correct) {
        improvementContainer.classList.remove('d-none');
        document.getElementById('improvementDisplay').innerHTML = feedback.improvement_suggestions;
    } else {
        improvementContainer.classList.add('d-none');
    }

    // 显示反馈容器
    feedbackContainer.classList.remove('d-none');

    // 渲染数学公式 - 分别渲染各个元素以确保正确渲染
    setTimeout(() => {
        // 渲染解释部分
        renderMathInElement(document.getElementById('explanationDisplay'), {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ],
            throwOnError: false
        });

        // 渲染改进建议部分
        if (feedback.improvement_suggestions && !feedback.is_correct) {
            renderMathInElement(document.getElementById('improvementDisplay'), {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false}
                ],
                throwOnError: false
            });
        }

        // 最后渲染整个容器，确保所有内容都被渲染
        renderMathInElement(feedbackContainer, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ],
            throwOnError: false
        });
    }, 10);
}

// 重置UI
function resetUI() {
    // 重置表单
    questionForm.reset();

    // 隐藏问题和反馈容器
    questionContainer.classList.add('d-none');
    feedbackContainer.classList.add('d-none');

    // 重置全局变量
    currentQuestion = null;
    selectedOption = null;

    // 滚动到顶部
    window.scrollTo(0, 0);
}
