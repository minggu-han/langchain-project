"""
聊天 API 路由 - LangChain 各种对话模式的 HTTP 接口

本模块提供的接口：
┌────────────────────────┬─────────────────────────────────────────┐
│ 接口                    │ LangChain 功能                          │
├────────────────────────┼─────────────────────────────────────────┤
│ POST /chat/simple       │ 基础对话（Prompt + LLM Chain）          │
│ POST /chat/role         │ 角色扮演对话                            │
│ POST /chat/with-memory  │ 带记忆的对话（ConversationMemory）      │
│ POST /chat/translate    │ 翻译链                                  │
│ POST /chat/code-review  │ 代码审阅链                             │
│ POST /chat/agent        │ Agent 智能代理（工具调用）              │
│ POST /chat/rag          │ RAG 检索增强生成                        │
│ GET  /chat/history      │ 查询聊天历史                            │
│ GET  /chat/sessions     │ 查询所有会话                            │
└────────────────────────┴─────────────────────────────────────────┘

每个接口对应 LangChain 的一个核心概念，循序渐进学习。
"""
import json
import uuid
from typing import List, AsyncIterator
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.logging import get_logger
from app.models.user import User
from app.models.chat import ChatHistory, MessageRole
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatHistoryResponse,
    ChatHistoryItem,
    RAGQueryRequest,
    AgentQueryRequest,
    ChatSessionList,
)

# LangChain 工具模块
from app.langchain_utils.chains import (
    basic_chat,
    basic_chat_stream,
    translate_chain,
    translate_chain_stream,
    code_review_chain,
    code_review_chain_stream,
    role_chat,
)
from app.langchain_utils.memory import (
    chat_with_memory,
    chat_with_memory_stream,
    get_session_history,
    clear_session,
    get_all_sessions,
)
from app.langchain_utils.rag import rag_query, rag_query_stream
from app.langchain_utils.agents import run_functions_agent, run_functions_agent_stream, get_available_tools_info
from app.langchain_utils.llm_factory import get_llm, get_embeddings

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["LangChain 对话"])


# ================================================================
# 1. 基础对话（学习 LangChain 链式调用）
# ================================================================
@router.post(
    "/simple",
    response_model=ChatResponse,
    summary="[基础] 简单对话",
    description="""
LangChain 最基础的对话模式：Prompt → LLM → OutputParser

学习要点：
- ChatPromptTemplate 构建提示词模板
- LCEL (|) 管道操作符串联组件
- StrOutputParser 解析 LLM 输出
""",
)
async def simple_chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """基础对话：演示 LangChain 最基本的链式调用"""
    # 生成或使用传入的 session_id
    session_id = request.session_id or str(uuid.uuid4())

    logger.info("POST /chat/simple - user=%s, session=%s, msg_len=%d",
                current_user.username, session_id, len(request.message))

    # 调用 LangChain 链
    ai_response = await basic_chat(
        message=request.message,
        system_prompt=request.system_prompt,
        temperature=request.temperature or 0.7,
    )

    # 保存聊天记录到数据库
    await _save_chat_history(
        db=db,
        user_id=current_user.id,
        session_id=session_id,
        user_message=request.message,
        ai_response=ai_response,
        model_name=get_llm().model_name,
    )

    return ChatResponse(
        session_id=session_id,
        user_message=request.message,
        ai_response=ai_response,
        model_name=get_llm().model_name,
    )


# ================================================================
# 2. 角色扮演对话（学习 Role Prompting）
# ================================================================
@router.post(
    "/role",
    response_model=ChatResponse,
    summary="[基础] 角色扮演对话",
    description="""
让 AI 扮演特定角色来回答问题。

学习要点：
- System Prompt 如何控制 AI 行为
- Role Prompting 技巧
- Temperature 参数对创造性的影响

可选角色：历史学家、编程导师、医生、厨师、诗人
""",
)
async def role_based_chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """角色对话：学习 System Prompt 如何影响 AI 行为"""
    session_id = request.session_id or str(uuid.uuid4())

    # 从 system_prompt 中提取角色名
    role = request.system_prompt or "编程导师"
    logger.info("POST /chat/role - user=%s, session=%s, role=%s",
                current_user.username, session_id, role)

    ai_response = await role_chat(
        message=request.message,
        role=role,
    )

    await _save_chat_history(
        db=db,
        user_id=current_user.id,
        session_id=session_id,
        user_message=request.message,
        ai_response=ai_response,
        model_name=get_llm().model_name,
    )

    return ChatResponse(
        session_id=session_id,
        user_message=request.message,
        ai_response=ai_response,
        model_name=get_llm().model_name,
    )


# ================================================================
# 3. 带记忆的对话（学习 Memory）
# ================================================================
@router.post(
    "/with-memory",
    response_model=ChatResponse,
    summary="[进阶] 带记忆的对话",
    description="""
支持多轮对话记忆。AI 会记住之前的对话内容。

学习要点：
- ConversationBufferMemory 保存对话历史
- MessagesPlaceholder 动态注入历史消息
- RunnableWithMessageHistory 自动管理历史

用法：传入相同的 session_id 即可保持对话上下文。
""",
)
async def memory_chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """带记忆的对话：学习对话历史管理"""
    # 必须提供 session_id
    session_id = request.session_id or str(uuid.uuid4())
    logger.info("POST /chat/with-memory - user=%s, session=%s",
                current_user.username, session_id)

    ai_response = await chat_with_memory(
        message=request.message,
        session_id=session_id,
        system_prompt=request.system_prompt,
        temperature=request.temperature or 0.7,
    )

    # 保存聊天记录
    await _save_chat_history(
        db=db,
        user_id=current_user.id,
        session_id=session_id,
        user_message=request.message,
        ai_response=ai_response,
        model_name=get_llm().model_name,
    )

    return ChatResponse(
        session_id=session_id,
        user_message=request.message,
        ai_response=ai_response,
        model_name=get_llm().model_name,
    )


# ================================================================
# 4. 翻译链
# ================================================================
@router.post(
    "/translate",
    summary="[基础] 翻译链",
    description="将文本翻译为指定语言。演示专用 Prompt 模板的使用。",
)
async def translate_text(
    text: str,
    target_lang: str = "英文",
    current_user: User = Depends(get_current_user),
):
    """翻译链：学习专用 Prompt 模板"""
    result = await translate_chain(text=text, target_lang=target_lang)
    return {"translation": result}


# ================================================================
# 5. 代码审阅链
# ================================================================
@router.post(
    "/code-review",
    summary="[基础] 代码审阅链",
    description="对代码进行全面审阅，包括质量、安全、性能分析。",
)
async def review_code(
    code: str,
    language: str = "Python",
    current_user: User = Depends(get_current_user),
):
    """代码审阅链：学习结构化 Prompt"""
    result = await code_review_chain(code=code, language=language)
    return {"review": result}


# ================================================================
# 6. Agent 智能代理
# ================================================================
@router.post(
    "/agent",
    summary="[高级] Agent 智能代理",
    description="""
使用 Agent 执行复杂任务。Agent 可以自主选择和使用工具。

学习要点：
- Agent 的 ReAct 循环（思考 → 行动 → 观察）
- create_openai_functions_agent 创建 Agent
- AgentExecutor 管理执行流程

可用工具：calculator（计算器）、get_current_time（时间）、text_statistics（文本统计）
""",
)
async def agent_chat(
    request: AgentQueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Agent 对话：学习智能代理和工具调用"""
    session_id = request.session_id or str(uuid.uuid4())
    logger.info("POST /chat/agent - user=%s, session=%s",
                current_user.username, session_id)

    result = await run_functions_agent(
        message=request.message,
        tool_names=None,  # 使用所有可用工具
    )

    # 保存聊天记录
    await _save_chat_history(
        db=db,
        user_id=current_user.id,
        session_id=session_id,
        user_message=request.message,
        ai_response=result["answer"],
        model_name=get_llm().model_name,
    )

    return {
        "session_id": session_id,
        "answer": result["answer"],
        "thinking_steps": result["steps"],  # Agent 的思考步骤
        "total_steps": result["total_steps"],
    }


# ================================================================
# 7. 获取可用工具列表
# ================================================================
@router.get(
    "/tools",
    summary="[高级] 查看可用工具",
    description="返回 Agent 可以使用的所有工具及其说明。",
)
async def list_tools(current_user: User = Depends(get_current_user)):
    """列出所有可用的 Agent 工具"""
    return {"tools": get_available_tools_info()}


# ================================================================
# 8. RAG 查询（需要先索引文档）
# ================================================================
@router.post(
    "/rag",
    summary="[实战] RAG 检索增强生成",
    description="""
基于已索引的文档回答问题。

学习要点：
- 文档加载 → 切分 → Embedding → 向量存储 → 检索 → 生成
- similarity_search 向量相似度检索
- RAG Prompt 模板（"严格根据提供的上下文"）

先通过 /api/v1/documents/index 索引文档，再使用此接口查询。
""",
)
async def rag_chat(
    request: RAGQueryRequest,
    current_user: User = Depends(get_current_user),
):
    """RAG 查询：学习检索增强生成"""
    logger.info("POST /chat/rag - user=%s, collection=%s",
                current_user.username, request.collection_name or "default")
    result = await rag_query(
        query=request.query,
        collection_name=request.collection_name or "default",
        top_k=request.top_k or 4,
    )

    return result


# ================================================================
# 9. 查询聊天历史
# ================================================================
@router.get(
    "/history/{session_id}",
    response_model=ChatHistoryResponse,
    summary="查询聊天历史",
    description="获取指定会话的完整聊天记录。",
)
async def get_chat_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """查询指定会话的聊天历史"""
    # 从数据库查询该会话的所有消息
    result = await db.execute(
        select(ChatHistory)
        .where(
            ChatHistory.user_id == current_user.id,
            ChatHistory.session_id == session_id,
        )
        .order_by(ChatHistory.created_at)
    )
    records = result.scalars().all()

    if not records:
        raise HTTPException(status_code=404, detail="未找到该会话")

    messages = [
        ChatHistoryItem(
            role=record.role.value,
            content=record.content,
            created_at=record.created_at,
        )
        for record in records
    ]

    # 计算总 Token 消耗
    total_tokens = sum(r.token_count for r in records if r.token_count)

    return ChatHistoryResponse(
        session_id=session_id,
        messages=messages,
        total_tokens=total_tokens or None,
    )


# ================================================================
# 10. 查询所有会话列表
# ================================================================
@router.get(
    "/sessions",
    response_model=ChatSessionList,
    summary="查询会话列表",
    description="获取当前用户的所有聊天会话 ID。",
)
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取当前用户的所有会话 ID 列表"""
    # 查询数据库中不重复的 session_id
    result = await db.execute(
        select(ChatHistory.session_id)
        .where(ChatHistory.user_id == current_user.id)
        .distinct()
        .order_by(ChatHistory.session_id)
    )
    db_sessions = result.scalars().all()

    # 合并内存中的会话（Memory 模式）
    memory_sessions = await get_all_sessions()
    all_sessions = list(set(list(db_sessions) + memory_sessions))

    return ChatSessionList(
        sessions=all_sessions,
        total=len(all_sessions),
    )


# ================================================================
# 11. 清除会话记忆
# ================================================================
@router.delete(
    "/sessions/{session_id}",
    summary="清除会话",
    description="清除指定会话的记忆和聊天记录。",
)
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """删除指定会话"""
    # 清除 Redis 中的会话
    await clear_session(session_id)

    # 清除数据库中的记录
    from sqlalchemy import delete
    await db.execute(
        delete(ChatHistory).where(
            ChatHistory.user_id == current_user.id,
            ChatHistory.session_id == session_id,
        )
    )
    await db.flush()

    return {"message": f"会话 {session_id} 已清除", "status": "success"}


# ================================================================
# 辅助函数：保存聊天记录
# ================================================================
async def _save_chat_history(
    db: AsyncSession,
    user_id: int,
    session_id: str,
    user_message: str,
    ai_response: str,
    model_name: str | None = None,
):
    """
    将聊天记录保存到数据库

    保存两条记录：
    1. 用户的消息（role=user）
    2. AI 的回复（role=assistant）
    """
    logger.debug("Saving chat history: user_id=%d, session=%s, model=%s",
                 user_id, session_id, model_name or "unknown")
    # 保存用户消息
    user_record = ChatHistory(
        user_id=user_id,
        session_id=session_id,
        role=MessageRole.USER,
        content=user_message,
        model_name=model_name,
    )
    db.add(user_record)

    # 保存 AI 回复
    ai_record = ChatHistory(
        user_id=user_id,
        session_id=session_id,
        role=MessageRole.ASSISTANT,
        content=ai_response,
        model_name=model_name,
    )
    db.add(ai_record)

    # 不在这里 commit，由 get_db 依赖统一提交
    await db.flush()


# ================================================================
# SSE 流式输出工具
# ================================================================
def _sse(data: dict) -> str:
    """将字典转为 SSE 格式：data: {json}\n\n"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_tokens(gen: AsyncIterator[str]) -> AsyncIterator[str]:
    """将 token 流包装为 SSE 事件流"""
    yield _sse({"type": "start"})
    async for chunk in gen:
        if chunk:
            yield _sse({"type": "token", "content": chunk})
    yield _sse({"type": "done"})


def _sse_response(gen):
    """返回带标准头部的 StreamingResponse"""
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ================================================================
# 12. 流式基础对话
# ================================================================
@router.post("/simple/stream", summary="[基础] 流式对话", description="基础对话的 SSE 流式版本，逐 token 返回 AI 回复。")
async def simple_chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    logger.info("POST /chat/simple/stream - user=%s", current_user.username)
    stream = basic_chat_stream(
        message=request.message,
        system_prompt=request.system_prompt,
        temperature=request.temperature or 0.7,
    )
    return _sse_response(_stream_tokens(stream))


# ================================================================
# 13. 流式角色对话
# ================================================================
@router.post("/role/stream", summary="[基础] 流式角色对话", description="角色对话的 SSE 流式版本。")
async def role_based_chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    role = request.system_prompt or "编程导师"
    logger.info("POST /chat/role/stream - user=%s, role=%s", current_user.username, role)
    # role_chat 内部调用 basic_chat，这里直接用 basic_chat_stream
    role_prompts = {
        "历史学家": "你是一位资深历史学家。",
        "编程导师": "你是一位耐心的编程导师，擅长用简单的类比解释复杂概念。回答时要包含代码示例和练习建议。",
        "医生": "你是一位经验丰富的医生。",
        "厨师": "你是一位米其林三星厨师。",
        "诗人": "你是一位现代诗人。",
    }
    system_prompt = role_prompts.get(role, f"你是一位专业的{role}。")
    stream = basic_chat_stream(message=request.message, system_prompt=system_prompt, temperature=0.8)
    return _sse_response(_stream_tokens(stream))


# ================================================================
# 14. 流式记忆对话
# ================================================================
@router.post("/with-memory/stream", summary="[进阶] 流式记忆对话", description="带记忆对话的 SSE 流式版本，历史存储在 Redis。")
async def memory_chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    session_id = request.session_id or str(uuid.uuid4())
    logger.info("POST /chat/with-memory/stream - user=%s, session=%s", current_user.username, session_id)
    stream = chat_with_memory_stream(
        message=request.message,
        session_id=session_id,
        system_prompt=request.system_prompt,
        temperature=request.temperature or 0.7,
    )
    return _sse_response(_stream_tokens(stream))


# ================================================================
# 15. 流式翻译
# ================================================================
@router.post("/translate/stream", summary="[基础] 流式翻译", description="翻译链的 SSE 流式版本。")
async def translate_text_stream(
    text: str,
    target_lang: str = "英文",
    current_user: User = Depends(get_current_user),
):
    logger.info("POST /chat/translate/stream - user=%s, target=%s", current_user.username, target_lang)
    stream = translate_chain_stream(text=text, target_lang=target_lang)
    return _sse_response(_stream_tokens(stream))


# ================================================================
# 16. 流式代码审阅
# ================================================================
@router.post("/code-review/stream", summary="[基础] 流式代码审阅", description="代码审阅链的 SSE 流式版本。")
async def review_code_stream(
    code: str,
    language: str = "Python",
    current_user: User = Depends(get_current_user),
):
    logger.info("POST /chat/code-review/stream - user=%s, language=%s", current_user.username, language)
    stream = code_review_chain_stream(code=code, language=language)
    return _sse_response(_stream_tokens(stream))


# ================================================================
# 17. 流式 RAG 查询
# ================================================================
@router.post("/rag/stream", summary="[实战] 流式 RAG 查询", description="RAG 的 SSE 流式版本：逐 token + 最后返回引用来源。")
async def rag_chat_stream(
    request: RAGQueryRequest,
    current_user: User = Depends(get_current_user),
):
    logger.info("POST /chat/rag/stream - user=%s, collection=%s",
                current_user.username, request.collection_name or "default")

    async def event_generator():
        async for event in rag_query_stream(
            query=request.query,
            collection_name=request.collection_name or "default",
            top_k=request.top_k or 4,
        ):
            yield _sse(event)

    return _sse_response(event_generator())


# ================================================================
# 18. 流式 Agent
# ================================================================
@router.post("/agent/stream", summary="[高级] 流式 Agent", description="Agent 的 SSE 流式版本：展示思考过程和工具调用。")
async def agent_chat_stream(
    request: AgentQueryRequest,
    current_user: User = Depends(get_current_user),
):
    logger.info("POST /chat/agent/stream - user=%s", current_user.username)

    async def event_generator():
        async for event in run_functions_agent_stream(message=request.message):
            yield _sse(event)

    return _sse_response(event_generator())
