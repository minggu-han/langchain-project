"""
LangChain 进阶篇 —— Memory（对话记忆）

为什么需要 Memory？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
默认情况下，LLM 是无状态的 —— 每次调用都是独立的。
你说 "我叫小明"，下次再问 "我叫什么？"，LLM 不知道。

Memory 的作用：
1. 记住对话历史 —— 多次对话间保持上下文
2. 自动构建 Prompt —— 将历史消息注入到每次请求中
3. 管理 Token 消耗 —— 避免历史过长导致成本过高

LangChain Memory 类型：
┌────────────────────────────┬──────────────────────────────────┐
│ 类型                        │ 说明                             │
├────────────────────────────┼──────────────────────────────────┤
│ ConversationBufferMemory    │ 完整保存所有对话（最消耗 Token） │
│ ConversationSummaryMemory   │ 用摘要代替历史（节省 Token）     │
│ ConversationBufferWindow    │ 只保留最近 N 轮对话              │
│ ConversationTokenBuffer     │ 按 Token 数限制历史              │
│ ConversationSummaryBuffer   │ 摘要 + 最近窗口（推荐！）        │
└────────────────────────────┴──────────────────────────────────┘

本模块使用 RedisChatMessageHistory 持久化对话历史：
- 服务重启后历史不丢失
- 支持多进程/多实例共享
- 自动过期清理（TTL 7 天）
"""
from typing import Dict, List, Set
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories.redis import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from app.langchain_utils.llm_factory import get_llm
from app.core.logging import get_logger
from app.config import get_settings

settings = get_settings()
logger = get_logger(__name__)

# 会话历史 TTL：7 天后自动清理
SESSION_TTL = 7 * 24 * 60 * 60  # 7 days in seconds


# ================================================================
# 1. Redis 会话存储
# ================================================================
def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """
    获取指定会话的 Redis 历史记录

    每次调用返回一个新的 RedisChatMessageHistory 实例，
    但底层数据存在 Redis 中，所以服务重启后历史不丢失。

    LangChain 调用这个函数来：
    1. 读取当前会话的历史消息（通过 aget_messages()）
    2. 将历史消息注入到 Prompt 中
    3. AI 看到完整的对话上下文后生成回复

    Args:
        session_id: 会话 ID（通常是 UUID 或自定义标识符）

    Returns:
        RedisChatMessageHistory 实例
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        url=settings.REDIS_URL,
        ttl=SESSION_TTL,
    )


async def clear_session(session_id: str):
    """清除指定会话的所有历史"""
    history = get_session_history(session_id)
    history.clear()
    logger.info("Session cleared: session_id=%s", session_id)


async def get_all_sessions() -> List[str]:
    """
    获取所有活跃会话 ID 列表

    通过 Redis SCAN 查找所有 message_store: 前缀的 key，
    提取 session_id 部分。每个 key 对应一个活跃会话。
    """
    import redis.asyncio as aioredis
    client = aioredis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    session_ids: Set[str] = set()
    try:
        cursor = 0
        while True:
            cursor, keys = await client.scan(
                cursor, match="message_store:*", count=100
            )
            for key in keys:
                # key 格式: message_store:default:<session_id>
                # 提取 session_id（最后一部分）
                parts = key.split(":", 2)
                if len(parts) >= 3:
                    session_ids.add(parts[2])
            if cursor == 0:
                break
    finally:
        await client.aclose()

    return sorted(session_ids)


# ================================================================
# 2. 带记忆的对话链
# ================================================================
async def chat_with_memory(
    message: str,
    session_id: str,
    system_prompt: str | None = None,
    temperature: float = 0.7,
):
    """
    带记忆的对话 —— LangChain 最常用的对话模式

    工作流程：
    ┌─────────────────────────────────────────────────┐
    │ 1. 用户发送消息 "我叫小明"                       │
    │ 2. 从 Redis 加载历史（首次为空）                 │
    │ 3. 构建 Prompt：System + History + 用户消息      │
    │ 4. LLM 生成回复                                  │
    │ 5. 将用户消息和 AI 回复添加到 Redis 历史         │
    │ 6. 返回 AI 回复                                  │
    │                                                  │
    │ 下次对话：                                       │
    │ 1. 用户发送消息 "我叫什么？"                     │
    │ 2. 从 Redis 加载历史（包含上一轮对话）           │
    │ 3. LLM 看到历史中有 "我叫小明"，回答 "你叫小明"   │
    └─────────────────────────────────────────────────┘

    Args:
        message: 用户消息
        session_id: 会话 ID（用于关联多轮对话）
        system_prompt: 系统提示词
        temperature: 温度参数

    Returns:
        AI 的回复文本
    """
    llm = get_llm(temperature=temperature)

    # 获取当前会话的历史消息数
    history_obj = get_session_history(session_id)
    all_msgs = history_obj.messages
    msg_count = len(all_msgs)
    logger.info("chat_with_memory: session=%s, message='%.50s...', history_msgs=%d",
                session_id, message, msg_count)

    # 构建 Prompt 模板
    messages = []

    if system_prompt:
        messages.append(("system", system_prompt))
    else:
        messages.append(("system", "你是一个有帮助的AI助手。请记住我们对话的上下文。"))

    messages.append(MessagesPlaceholder(variable_name="history"))
    messages.append(("human", "{input}"))

    prompt = ChatPromptTemplate.from_messages(messages)

    # 构建基础链
    chain = prompt | llm | StrOutputParser()

    # 包装为带历史记录的链
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    response = await chain_with_history.ainvoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}},
    )

    logger.info("chat_with_memory completed: response='%.100s...'", response)
    return response


# ================================================================
# 3. 带摘要记忆的对话（节省 Token）
# ================================================================
async def chat_with_summary_memory(
    message: str,
    session_id: str,
    system_prompt: str | None = None,
):
    """
    带摘要记忆的对话 —— 适合长时间对话

    问题：对话越长，历史消息越多，Token 消耗越大
    解决方案：将历史对话摘要成一段文字

    Token 消耗对比（假设每轮 100 token）：
    ┌──────────┬──────────────────┬────────────────┐
    │ 对话轮数  │ 传统 Buffer       │ 摘要方式        │
    ├──────────┼──────────────────┼────────────────┤
    │ 10 轮     │ ~2000 tokens     │ ~300 tokens    │
    │ 50 轮     │ ~10000 tokens    │ ~500 tokens    │
    │ 100 轮    │ ~20000 tokens    │ ~500 tokens    │
    └──────────┴──────────────────┴────────────────┘

    实现思路：
    1. 保留最近几轮完整对话（窗口）
    2. 更早的对话用 LLM 生成摘要
    3. Prompt = System + 摘要 + 最近窗口 + 新消息
    """
    llm = get_llm(temperature=0.7)

    history_obj = get_session_history(session_id)
    all_messages = history_obj.messages

    logger.info("chat_with_summary_memory: session=%s, message='%.50s...', history_msgs=%d",
                session_id, message, len(all_messages))

    # 如果历史超过 10 条消息，生成摘要
    summary = ""
    recent_messages = all_messages

    if len(all_messages) > 10:
        logger.info("chat_with_summary_memory: generating summary for %d old messages...",
                    len(all_messages) - 6)
        recent_messages = all_messages[-6:]
        old_messages = all_messages[:-6]

        old_messages_text = "\n".join([
            f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in old_messages
        ])

        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "请用 2-3 句话总结以下对话的主要内容："),
            ("human", old_messages_text),
        ])
        summary_chain = summary_prompt | llm | StrOutputParser()
        summary = await summary_chain.ainvoke({})
        logger.debug("chat_with_summary_memory: summary='%s'", summary)

        # 用摘要 + 最近历史替换完整历史
        history_obj.clear()
        for msg in recent_messages:
            history_obj.add_message(msg)

    # 构建包含摘要的 System Prompt
    full_system_prompt = system_prompt or "你是一个有帮助的AI助手。"
    if summary:
        full_system_prompt += f"\n\n【之前的对话摘要】\n{summary}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", full_system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    response = await chain_with_history.ainvoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}},
    )

    return response


# ================================================================
# 4. 流式输出
# ================================================================
from typing import AsyncIterator


async def chat_with_memory_stream(
    message: str,
    session_id: str,
    system_prompt: str | None = None,
    temperature: float = 0.7,
) -> AsyncIterator[str]:
    """
    带记忆的流式对话 — 逐 token 输出，历史保存在 Redis

    RunnableWithMessageHistory 支持 .astream()，
    历史在流开始前加载，在流结束后保存。
    """
    llm = get_llm(temperature=temperature)

    messages = []
    if system_prompt:
        messages.append(("system", system_prompt))
    else:
        messages.append(("system", "你是一个有帮助的AI助手。请记住我们对话的上下文。"))
    messages.append(MessagesPlaceholder(variable_name="history"))
    messages.append(("human", "{input}"))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    logger.info("chat_with_memory_stream: session=%s", session_id)
    async for chunk in chain_with_history.astream(
        {"input": message},
        config={"configurable": {"session_id": session_id}},
    ):
        if chunk:
            yield chunk
    logger.info("chat_with_memory_stream completed")
