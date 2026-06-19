"""
LangChain 基础篇 —— Chain（链式调用）

核心概念：LCEL（LangChain Expression Language）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LCEL 是 LangChain 的表达式语言，用管道符 | 连接组件：

    prompt | model | output_parser

就像 Unix 管道：cat file | grep "hello" | sort

主要优势：
1. 流式处理 —— 数据像水一样流过管道，支持 streaming
2. 异步支持 —— 所有组件自动支持 async
3. 并行执行 —— 可以轻松实现 RunnableParallel
4. 自动重试 —— 失败自动重试和回退
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

本模块演示的 Chain 类型：
├── 基础对话链       —— Prompt → LLM → StrOutputParser
├── 翻译链           —— 多 Prompt 串联
├── 代码审阅链       —— System + Human Prompt 组合
└── 带结构输出的链    —— 使用 with_structured_output
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from app.langchain_utils.llm_factory import get_llm
from app.core.logging import get_logger

logger = get_logger(__name__)


# ================================================================
# 1. 基础对话链（最基本的 LCEL 用法）
# ================================================================
async def basic_chat(
    message: str,
    system_prompt: str | None = None,
    temperature: float = 0.7,
):
    """
    最基础的 LangChain 对话

    LCEL 表达式：
        prompt | llm | output_parser

    等价于：
        1. 用 prompt 格式化消息
        2. 将格式化后的消息发送给 LLM
        3. 将 LLM 的响应解析为字符串

    Args:
        message: 用户消息
        system_prompt: 系统提示词（设定 AI 角色）
        temperature: 温度参数

    Returns:
        AI 的回复文本
    """
    # 第1步：创建 LLM 实例
    # 这一步通常很快（复用缓存或创建新对象）
    llm = get_llm(temperature=temperature)

    # 第2步：构建 Prompt 模板
    # ChatPromptTemplate 可以包含多种角色的消息：
    # - SystemMessagePromptTemplate：设定 AI 角色
    # - HumanMessagePromptTemplate：用户消息模板
    # - AIMessagePromptTemplate：AI 消息模板
    messages = []

    if system_prompt:
        # 如果提供了系统提示词，使用它来设定 AI 角色
        messages.append(("system", system_prompt))
    else:
        # 使用默认的系统提示词
        messages.append(("system", "你是一个有帮助的AI助手。"))

    # {message} 是占位符，会被实际消息替换
    messages.append(("human", "{message}"))

    prompt = ChatPromptTemplate.from_messages(messages)

    # 第3步：创建输出解析器
    # StrOutputParser：将 AIMessage 对象转换为纯文本字符串
    output_parser = StrOutputParser()

    # 第4步：构建 LCEL 链
    # | 是 LCEL 的管道符：prompt → llm → output_parser
    chain = prompt | llm | output_parser

    # 第5步：执行链
    logger.info("basic_chat: message='%.60s...' temperature=%.2f", message, temperature)
    logger.debug("basic_chat: %d prompt messages", len(messages))
    response = await chain.ainvoke({"message": message})
    logger.info("basic_chat completed: response='%.100s...'", response)

    return response


# ================================================================
# 2. 翻译链（多语言支持）
# ================================================================
async def translate_chain(
    text: str,
    source_lang: str = "自动检测",
    target_lang: str = "英文",
):
    """
    翻译链 —— 展示如何使用专用的 Prompt 模板

    这个 Chain 的特点：
    - 有明确的任务目标（翻译）
    - 使用格式化的输出要求
    - 可以处理多种语言
    """
    llm = get_llm(temperature=0.3)  # 翻译任务用低温度，确保准确性

    logger.info("translate_chain: %s -> %s, text_len=%d", source_lang, target_lang, len(text))

    # 翻译专用的 System Prompt
    system_prompt = f"""你是一个专业的翻译专家。
你的任务是将用户输入的文本从 {source_lang} 翻译为 {target_lang}。

要求：
1. 保持原文的语气和风格
2. 专业术语翻译准确
3. 输出格式：
   【原文】原文内容
   【译文】翻译内容
4. 如果原文已经是目标语言，请告知用户"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "请翻译以下文本：\n{text}"),
    ])

    chain = prompt | llm | StrOutputParser()

    result = await chain.ainvoke({"text": text})
    logger.info("translate_chain completed: result_len=%d", len(result))
    return result


# ================================================================
# 3. 代码审阅链（结构化输出示例）
# ================================================================
async def code_review_chain(code: str, language: str = "Python"):
    """
    代码审阅链 —— 展示专业的代码分析

    用途：自动审阅代码质量、安全性、性能
    """
    llm = get_llm(temperature=0.3)

    logger.info("code_review_chain: language=%s, code_len=%d", language, len(code))

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是一个资深的{language}代码审阅专家。
请对提供的代码进行全面审阅，包括：
1. 代码质量和可读性
2. 潜在的 Bug 和安全漏洞
3. 性能优化建议
4. 最佳实践建议
5. 改进后的代码示例

请用 Markdown 格式输出审阅报告。"""),
        ("human", "```{language}\n{code}\n```"),
    ])

    chain = prompt | llm | StrOutputParser()

    result = await chain.ainvoke({
        "code": code,
        "language": language,
    })
    logger.info("code_review_chain completed: result_len=%d", len(result))
    return result


# ================================================================
# 4. 自定义角色对话（Role Prompting）
# ================================================================
async def role_chat(
    message: str,
    role: str = "历史学家",
):
    """
    角色对话 —— 展示 Role Prompting 技巧

    Role Prompting 是让 LLM 扮演特定角色的技术：
    - "你是一个资深历史学家，专注于中国古代史..."
    - "你是一个 Socratic 老师，通过提问来引导学生学习..."

    效果：角色设定越具体，回答质量越高
    """
    role_prompts = {
        "历史学家": "你是一位资深历史学家，擅长用生动的故事讲述历史。回答时请引用具体的历史事件和年份。",
        "编程导师": "你是一位耐心的编程导师，擅长用简单的类比解释复杂概念。回答时要包含代码示例和练习建议。",
        "医生": "你是一位经验丰富的医生，但请注意在回答前声明'我不是真正的医生，以下仅供参考'。",
        "厨师": "你是一位米其林三星厨师，热爱分享烹饪技巧。回答时请包含详细的步骤和食材清单。",
        "诗人": "你是一位现代诗人，回答问题时也要保持诗意的语言风格。",
    }

    system_prompt = role_prompts.get(role, f"你是一位专业的{role}。请以这个身份回答用户的问题。")

    logger.info("role_chat: role='%s', message='%.50s...'", role, message)

    return await basic_chat(
        message=message,
        system_prompt=system_prompt,
        temperature=0.8,  # 角色对话可以稍微创造性
    )


# ================================================================
# 5. 链的调试和检查
# ================================================================
def explain_chain_structure(chain) -> str:
    """
    查看 Chain 的结构（学习/调试用）

    使用方法：
        chain = prompt | llm | output_parser
        print(explain_chain_structure(chain))
    """
    # 获取链的图结构（DAG: Directed Acyclic Graph）
    try:
        graph_repr = chain.get_graph().draw_ascii()
    except Exception:
        graph_repr = "<无法生成图结构>"

    return f"Chain 结构:\n{graph_repr}"


# ================================================================
# 6. 流式输出（Streaming）
# ================================================================
from typing import AsyncIterator


async def basic_chat_stream(
    message: str,
    system_prompt: str | None = None,
    temperature: float = 0.7,
) -> AsyncIterator[str]:
    """
    基础对话流式 — 逐 token 返回 AI 回复

    与 basic_chat 的区别：
    - basic_chat: await chain.ainvoke(...) → 完整 str
    - basic_chat_stream: async for chunk in chain.astream(...) → 逐个 yield token

    astream() 返回 AsyncIterator[str]，每迭代一次就收到一个 token
    例：迭代返回 "Python" "装饰器" "是" "一种" ...
    """
    llm = get_llm(temperature=temperature)

    messages = []
    if system_prompt:
        messages.append(("system", system_prompt))
    else:
        messages.append(("system", "你是一个有帮助的AI助手。"))
    messages.append(("human", "{message}"))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | StrOutputParser()

    logger.info("basic_chat_stream: message='%.60s...'", message)
    async for chunk in chain.astream({"message": message}):
        if chunk:
            yield chunk
    logger.info("basic_chat_stream completed")


async def translate_chain_stream(
    text: str,
    source_lang: str = "自动检测",
    target_lang: str = "英文",
) -> AsyncIterator[str]:
    """翻译链流式"""
    llm = get_llm(temperature=0.3)

    logger.info("translate_chain_stream: %s -> %s", source_lang, target_lang)
    system_prompt = f"""你是一个专业的翻译专家。
将用户输入的文本从 {source_lang} 翻译为 {target_lang}。
保持原文的语气和风格，专业术语翻译准确。"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "请翻译以下文本：\n{text}"),
    ])
    chain = prompt | llm | StrOutputParser()

    async for chunk in chain.astream({"text": text}):
        if chunk:
            yield chunk


async def code_review_chain_stream(
    code: str,
    language: str = "Python",
) -> AsyncIterator[str]:
    """代码审阅链流式"""
    llm = get_llm(temperature=0.3)

    logger.info("code_review_chain_stream: language=%s, code_len=%d", language, len(code))
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是一个资深的{language}代码审阅专家。
请对提供的代码进行全面审阅，包括：
1. 代码质量和可读性
2. 潜在的 Bug 和安全漏洞
3. 性能优化建议
4. 最佳实践建议
5. 改进后的代码示例

请用 Markdown 格式输出审阅报告。"""),
        ("human", "```{language}\n{code}\n```"),
    ])
    chain = prompt | llm | StrOutputParser()

    async for chunk in chain.astream({"code": code, "language": language}):
        if chunk:
            yield chunk
