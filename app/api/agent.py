"""
Full-Featured Agent API — 全功能智能代理

整合 Memory + Skills + Tools + RAG + Chain 的统一对话端点。

能力一览：
┌─────────────────┬──────────────────────────────────────────────┐
│ 能力             │ 说明                                         │
├─────────────────┼──────────────────────────────────────────────┤
│ 🧠 Memory       │ Redis 持久化对话历史，多轮上下文               │
│ 🎯 Skills       │ 8 个预置技能（代码审阅/翻译/摘要/...）        │
│ 🔧 Tools        │ 计算器、时间、文本统计                        │
│ 📚 RAG          │ Milvus 文档检索                               │
│ 💬 Chain        │ 通用对话（无需任何工具时）                     │
│ ⚡ Stream       │ SSE 流式输出 + 中间步骤展示                    │
└─────────────────┴──────────────────────────────────────────────┘

Agent 自主决策：
  "帮我审阅这段代码，然后计算它的时间复杂度"
    → use_skill("code-review", code) → calculator(expression)
    → 综合两个结果输出完整答案

  "现在几点了？顺便翻译成英文：你好世界"
    → get_current_time() → use_skill("translate", "你好世界")
    → 返回时间 + 翻译结果

  "今天天气怎么样"
    → 不需要任何工具 → 直接回复
"""
import json
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.logging import get_logger
from app.models.user import User
from app.models.chat import ChatHistory, MessageRole
from app.langchain_utils.full_agent import run_full_agent, run_full_agent_stream
from app.langchain_utils.llm_factory import get_llm
from app.schemas.skill import (
    AgentChatRequest,
    AgentChatResponse,
    AgentStepInfo,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/agent", tags=["Full Agent 全功能代理"])


# ================================================================
# SSE 工具
# ================================================================
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_response(gen):
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ================================================================
# 辅助：保存聊天记录
# ================================================================
async def _save_agent_history(
    db: AsyncSession,
    user_id: int,
    session_id: str,
    user_message: str,
    ai_response: str,
    model_name: str | None = None,
    steps: list | None = None,
):
    """保存 Agent 对话到 PostgreSQL"""
    ai_meta = ai_response
    if steps:
        tools_used = [s.get("tool", "") for s in steps]
        ai_meta = f"[tools: {', '.join(tools_used)}] {ai_response}"

    user_record = ChatHistory(
        user_id=user_id,
        session_id=session_id,
        role=MessageRole.USER,
        content=user_message,
        model_name=model_name,
    )
    db.add(user_record)

    ai_record = ChatHistory(
        user_id=user_id,
        session_id=session_id,
        role=MessageRole.ASSISTANT,
        content=ai_meta,
        model_name=model_name,
    )
    db.add(ai_record)
    await db.flush()

    logger.debug("Saved agent chat: user_id=%d, session=%s", user_id, session_id)


# ================================================================
# 1. 全功能 Agent 对话（同步）
# ================================================================
@router.post(
    "/chat",
    response_model=AgentChatResponse,
    summary="全功能 Agent 对话",
    description="""
**终极端点**：一个接口整合所有能力。

Agent 自主决定每一步该用什么：
- 需要专业技能？ → 调用 use_skill（8 个技能任选）
- 需要查文档？ → 调用 search_documents（RAG）
- 需要计算/时间/统计？ → 调用对应工具
- 什么都不需要？ → 直接回复

**多轮对话**：传入相同的 `session_id` 即可保持上下文。

**示例**：
```json
// 复杂任务：审阅代码 + 计算复杂度
{
    "message": "帮我审阅这段代码并计算时间复杂度：\\n\\ndef fib(n):\\n    if n <= 1: return n\\n    return fib(n-1) + fib(n-2)",
    "session_id": "my-session"
}

// 响应会展示 Agent 的思考步骤：
// Step 1: use_skill("code-review", ...) → 审阅报告
// Step 2: calculator("O(2^n)") → 复杂度分析
// Final: 综合输出
```

**可用工具一览**：
| 工具 | 说明 |
|------|------|
| use_skill | 8 个预置技能 |
| search_documents | Milvus RAG 文档检索 |
| calculator | 数学计算 |
| get_current_time | 获取时间 |
| text_statistics | 文本统计 |
""",
)
async def agent_chat(
    request: AgentChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """全功能 Agent 同步对话"""
    session_id = request.session_id or str(uuid.uuid4())
    message = request.message

    logger.info("POST /agent/chat — user=%s, session=%s, message='%.80s...'",
                current_user.username, session_id, message)

    try:
        result = await run_full_agent(
            message=message,
            session_id=session_id,
            temperature=request.temperature or 0.3,
        )
    except Exception as e:
        logger.error("Agent chat failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Agent 执行失败：{str(e)}")

    # 保存到数据库
    model_name = get_llm().model_name
    await _save_agent_history(
        db=db,
        user_id=current_user.id,
        session_id=session_id,
        user_message=message,
        ai_response=result["answer"],
        model_name=model_name,
        steps=result.get("steps", []),
    )

    # 构建步骤响应
    step_infos = [
        AgentStepInfo(
            tool=s["tool"],
            tool_input=s["tool_input"],
            observation=s["observation"],
        )
        for s in result.get("steps", [])
    ]

    return AgentChatResponse(
        answer=result["answer"],
        session_id=session_id,
        model_name=model_name,
        steps=step_infos,
        total_steps=result["total_steps"],
        rag_used=result.get("rag_used", False),
    )


# ================================================================
# 2. 全功能 Agent 对话（流式）
# ================================================================
@router.post(
    "/chat/stream",
    summary="全功能 Agent 对话（流式）",
    description="""
全功能 Agent 的 SSE 流式版本。

**SSE 事件类型**：
| event | type | 说明 |
|-------|------|------|
| 开始 | `start` | Agent 已就绪 |
| LLM 输出 | `token` | 逐 token 文本 |
| 工具调用 | `tool_start` | Agent 开始调用工具 |
| 工具结果 | `tool_end` | 工具返回结果 |
| 完成 | `done` | 对话结束 |
| 错误 | `error` | 异常信息 |

**为什么流式更好**：
- 实时看到 Agent 的「思考 → 行动 → 观察」循环
- 长任务不会让用户等待
""",
)
async def agent_chat_stream(
    request: AgentChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """全功能 Agent 流式对话"""
    session_id = request.session_id or str(uuid.uuid4())
    message = request.message

    logger.info("POST /agent/chat/stream — user=%s, session=%s",
                current_user.username, session_id)

    async def event_generator():
        collected_tokens = []
        tool_steps = []

        yield _sse({
            "type": "start",
            "session_id": session_id,
            "message": "Agent 已就绪 — Memory + Skills + Tools + RAG",
        })

        try:
            async for event in run_full_agent_stream(
                message=message,
                session_id=session_id,
                temperature=request.temperature or 0.3,
            ):
                event_type = event.get("type")

                if event_type in ("rag_start", "rag_result", "rag_none"):
                    # 转发 RAG 状态事件
                    yield _sse(event)

                elif event_type == "token":
                    collected_tokens.append(event["content"])
                    yield _sse(event)

                elif event_type == "tool_start":
                    tool_steps.append({"tool": event["tool"], "input": event["input"]})
                    yield _sse(event)

                elif event_type == "tool_end":
                    # 补充 output 到最近的 tool_start
                    for step in reversed(tool_steps):
                        if step.get("tool") == event.get("tool") and "output" not in step:
                            step["output"] = event.get("output", "")
                            break
                    yield _sse(event)

                elif event_type == "done":
                    full_answer = "".join(collected_tokens)

                    # 流结束后保存到数据库
                    if full_answer.strip():
                        await _save_agent_history(
                            db=db,
                            user_id=current_user.id,
                            session_id=session_id,
                            user_message=message,
                            ai_response=full_answer,
                            model_name=get_llm().model_name,
                            steps=tool_steps,
                        )

                    yield _sse({
                        "type": "done",
                        "session_id": session_id,
                        "total_tools_used": len(tool_steps),
                    })

                elif event_type == "error":
                    yield _sse(event)

        except Exception as e:
            logger.error("Agent stream failed: %s", e)
            yield _sse({"type": "error", "message": str(e)})

    return _sse_response(event_generator())
