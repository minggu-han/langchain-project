"""
Skills API 路由 — 技能发现、意图路由与调用（带对话记忆）

本模块提供的接口：
┌───────────────────────────────────┬──────────────────────────────────────────┐
│ 接口                               │ 功能                                     │
├───────────────────────────────────┼──────────────────────────────────────────┤
│ GET  /skills                      │ 列出所有可用技能                          │
│ GET  /skills/{name}               │ 获取技能详情（含步骤描述）                 │
│ POST /skills/{name}/invoke        │ 直接调用指定技能（传参数）                 │
│ POST /skills/{name}/invoke/stream │ 直接调用指定技能（SSE 流式）              │
│ POST /skills/chat                 │ 智能对话（自动路由 + 记忆 + 存入DB）      │
│ POST /skills/chat/stream          │ 智能对话（SSE 流式 + 记忆）               │
└───────────────────────────────────┴──────────────────────────────────────────┘

智能对话流程（POST /skills/chat）：
  用户发送自然语言 → 意图路由 → 加载 Redis 历史 → LLM 按 Skill 步骤执行
  → 保存到 Redis（RunnableWithMessageHistory 自动）→ 保存到 PostgreSQL

LLM 自主判断模式（mode=llm）：
  System Prompt 包含所有 Skill 定义 + 对话历史
  LLM 自行判断是否使用技能、使用哪个技能
  → 多轮对话中，LLM 可以根据上下文自主决定
"""
import json
import uuid
from typing import AsyncIterator
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.logging import get_logger
from app.models.user import User
from app.models.chat import ChatHistory, MessageRole
from app.langchain_utils.skills import (
    get_skill,
    get_all_skills,
    get_skills_by_category,
    SKILL_REGISTRY,
    route_skill,
    execute_skill,
    execute_skill_stream,
    execute_with_llm_routing,
    execute_with_llm_routing_stream,
    chat_with_skill_memory,
    chat_with_skill_memory_stream,
)
from app.langchain_utils.llm_factory import get_llm
from app.schemas.skill import (
    SkillInfo,
    SkillDetail,
    SkillParamField,
    SkillInvokeRequest,
    SkillInvokeResponse,
    SkillListResponse,
    SkillChatRequest,
    SkillChatResponse,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/skills", tags=["Skills 技能"])


# ================================================================
# SSE 流式输出工具
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
# 辅助：保存聊天记录到数据库
# ================================================================
async def _save_skill_chat_history(
    db: AsyncSession,
    user_id: int,
    session_id: str,
    user_message: str,
    ai_response: str,
    model_name: str | None = None,
    skill_used: str | None = None,
    route_method: str | None = None,
):
    """保存 Skill 对话记录到 PostgreSQL"""
    # 附加元数据到消息中
    user_meta = user_message
    if skill_used:
        ai_meta = f"[skill: {skill_used}] [route: {route_method}] {ai_response}"
    else:
        ai_meta = f"[route: {route_method}] {ai_response}"

    user_record = ChatHistory(
        user_id=user_id,
        session_id=session_id,
        role=MessageRole.USER,
        content=user_meta,
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

    logger.debug("Saved skill chat: user_id=%d, session=%s, skill=%s",
                 user_id, session_id, skill_used or "none")


# ================================================================
# 1. 技能列表
# ================================================================
@router.get(
    "",
    response_model=SkillListResponse,
    summary="列出所有可用技能",
    description="返回所有已注册的技能列表。支持按分类筛选。",
)
async def list_skills(
    category: str | None = None,
    current_user: User = Depends(get_current_user),
):
    if category:
        skills = get_skills_by_category(category)
    else:
        skills = get_all_skills()

    logger.info("GET /skills%s — %d results",
                f"?category={category}" if category else "", len(skills))

    skill_infos = [
        SkillInfo(
            name=s.name,
            display_name=s.display_name,
            description=s.description,
            category=s.category,
            tags=s.tags,
            has_stream=True,
        )
        for s in skills
    ]

    all_categories = sorted(set(s.category for s in SKILL_REGISTRY.values()))

    return SkillListResponse(
        skills=skill_infos,
        total=len(skill_infos),
        categories=all_categories,
    )


# ================================================================
# 2. 技能详情
# ================================================================
@router.get(
    "/{name}",
    response_model=SkillDetail,
    summary="获取技能详情",
    description="查看指定技能的完整信息，包括触发关键词、参数和步骤。",
)
async def get_skill_detail(
    name: str,
    current_user: User = Depends(get_current_user),
):
    skill = get_skill(name)
    if skill is None:
        available = ", ".join(SKILL_REGISTRY.keys())
        raise HTTPException(
            status_code=404,
            detail=f"技能 '{name}' 不存在。可用技能：{available}",
        )

    logger.info("GET /skills/%s", name)

    param_fields = [
        SkillParamField(
            name=f["name"],
            type=f["type"],
            required=f["required"],
            default=f.get("default"),
            description=f["description"],
        )
        for f in skill.input_fields
    ]

    return SkillDetail(
        name=skill.name,
        display_name=skill.display_name,
        description=skill.description,
        category=skill.category,
        tags=skill.tags,
        has_stream=True,
        input_fields=param_fields,
    )


# ================================================================
# 3. 直接调用技能（传参数）
# ================================================================
def _build_message_from_params(skill_name: str, params: dict) -> str:
    """将结构化参数转换为自然语言消息"""
    if skill_name == "code-review":
        code = params.get("code", "")
        language = params.get("language", "Python")
        return f"请审阅以下 {language} 代码：\n\n```{language}\n{code}\n```"
    elif skill_name == "translate":
        text = params.get("text", "")
        target_lang = params.get("target_lang", "英文")
        return f"请将以下文本翻译为{target_lang}：\n\n{text}"
    elif skill_name == "summarize":
        text = params.get("text", "")
        max_len = params.get("max_length", "300")
        return f"请对以下文本进行摘要（控制在 {max_len} 字以内）：\n\n{text}"
    elif skill_name == "sql-generator":
        query = params.get("query", "")
        schema_info = params.get("schema_info", "")
        schema_part = f"\n\n表结构信息：\n{schema_info}" if schema_info and schema_info != "未提供" else ""
        return f"请生成 SQL 查询：{query}{schema_part}"
    elif skill_name == "email-writer":
        recipient = params.get("recipient", "")
        subject = params.get("subject", "")
        key_points = params.get("key_points", "")
        email_type = params.get("email_type", "商务邮件")
        return f"请撰写一封{email_type}。\n收件人：{recipient}\n主题：{subject}\n要点：{key_points}"
    elif skill_name == "explain-concept":
        concept = params.get("concept", "")
        level = params.get("level", "中级")
        return f"请用{level}难度解释：{concept}"
    elif skill_name == "data-analyzer":
        data_description = params.get("data_description", "")
        goal = params.get("goal", "了解数据特征和规律")
        return f"请分析以下数据：\n{data_description}\n\n分析目标：{goal}"
    elif skill_name == "interview-coach":
        position = params.get("position", "后端开发工程师")
        round_name = params.get("round_name", "技术一面")
        return f"请模拟一次 {position} 的{round_name}面试。"
    else:
        parts = [f"{k}: {v}" for k, v in params.items()]
        return "\n".join(parts)


@router.post(
    "/{name}/invoke",
    response_model=SkillInvokeResponse,
    summary="直接调用指定技能",
    description="按技能名称直接调用，传入结构化参数。",
)
async def invoke_skill(
    name: str,
    request: SkillInvokeRequest,
    current_user: User = Depends(get_current_user),
):
    skill = get_skill(name)
    if skill is None:
        available = ", ".join(SKILL_REGISTRY.keys())
        raise HTTPException(status_code=404, detail=f"技能 '{name}' 不存在。可用技能：{available}")

    logger.info("POST /skills/%s/invoke — user=%s", name, current_user.username)

    params_with_defaults = dict(request.params) if request.params else {}
    for field in skill.input_fields:
        if field["name"] not in params_with_defaults and field.get("default") is not None:
            params_with_defaults[field["name"]] = field["default"]

    for field in skill.input_fields:
        if field["required"] and field["name"] not in params_with_defaults:
            raise HTTPException(status_code=422, detail=f"缺少必填参数 '{field['name']}'")

    message = _build_message_from_params(name, params_with_defaults)

    try:
        result = await execute_skill(skill, message)
    except Exception as e:
        logger.error("Skill '%s' execution failed: %s", name, e)
        raise HTTPException(status_code=500, detail=f"技能执行失败：{str(e)}")

    return SkillInvokeResponse(
        skill_name=name,
        result=result,
        model_name=get_llm().model_name,
        params_used=params_with_defaults,
    )


# ================================================================
# 4. 直接调用技能（流式）
# ================================================================
@router.post(
    "/{name}/invoke/stream",
    summary="流式调用指定技能",
    description="直接调用技能的 SSE 流式版本。",
)
async def invoke_skill_stream(
    name: str,
    request: SkillInvokeRequest,
    current_user: User = Depends(get_current_user),
):
    skill = get_skill(name)
    if skill is None:
        available = ", ".join(SKILL_REGISTRY.keys())
        raise HTTPException(status_code=404, detail=f"技能 '{name}' 不存在。可用技能：{available}")

    logger.info("POST /skills/%s/invoke/stream — user=%s", name, current_user.username)

    params_with_defaults = dict(request.params) if request.params else {}
    for field in skill.input_fields:
        if field["name"] not in params_with_defaults and field.get("default") is not None:
            params_with_defaults[field["name"]] = field["default"]

    for field in skill.input_fields:
        if field["required"] and field["name"] not in params_with_defaults:
            raise HTTPException(status_code=422, detail=f"缺少必填参数 '{field['name']}'")

    message = _build_message_from_params(name, params_with_defaults)

    async def event_generator():
        try:
            async for chunk in execute_skill_stream(skill, message):
                if chunk:
                    yield _sse({"type": "token", "content": chunk})
            yield _sse({"type": "done"})
        except Exception as e:
            logger.error("Skill '%s' stream failed: %s", name, e)
            yield _sse({"type": "error", "message": str(e)})

    return _sse_response(event_generator())


# ================================================================
# 5. 智能 Skill 对话（带记忆 + 数据库存储）
# ================================================================
@router.post(
    "/chat",
    response_model=SkillChatResponse,
    summary="智能 Skill 对话（带记忆）",
    description="""
**核心功能**：用户发送自然语言消息 → 系统自动识别意图 → 利用对话记忆 → 执行 Skill。

## 路由模式

| mode | 说明 |
|------|------|
| `auto` | 关键词优先 → 低分时 LLM 二次判断（默认） |
| `llm` | LLM 自主判断是否使用技能、使用哪个技能 |
| `keyword` | 仅关键词匹配 |

## 对话记忆

**传入相同的 `session_id` 即可保持多轮对话上下文**：
- 第1轮："帮我审阅这段代码" → 系统选择 code-review 技能，给出审阅报告
- 第2轮："再优化一下性能部分" → 系统记得上一轮在审阅代码，继续优化
- 第3轮："翻译成英文" → LLM 自主判断切换为 translate 技能

记忆存储在 Redis，聊天记录持久化到 PostgreSQL。

## LLM 自主判断模式（mode=llm）

LLM 拥有所有 Skill 定义，可以：
- 根据对话历史自主选择技能
- 在技能间自由切换（如审阅完代码后，用户说"翻译成英文"）
- 选择不使用任何技能，进行普通对话
""",
)
async def skill_chat(
    request: SkillChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    智能 Skill 对话 — 带 Redis 记忆 + PostgreSQL 存储
    """
    message = request.message
    mode = request.mode
    session_id = request.session_id or str(uuid.uuid4())

    logger.info("POST /skills/chat — user=%s, session=%s, mode=%s, message='%.80s...'",
                current_user.username, session_id, mode, message)

    # 调用带记忆的 Skill 对话
    try:
        result, skill_used, route_method = await chat_with_skill_memory(
            message=message,
            session_id=session_id,
            mode=mode,
        )
    except Exception as e:
        logger.error("Skill chat failed: %s", e)
        raise HTTPException(status_code=500, detail=f"技能执行失败：{str(e)}")

    # 处理 keyword 模式未匹配的情况
    if result is None and route_method == "none" and mode == "keyword":
        raise HTTPException(
            status_code=404,
            detail="未匹配到任何技能。请尝试使用 mode=auto 或 mode=llm。",
        )

    # 保存到数据库
    model_name = get_llm().model_name
    await _save_skill_chat_history(
        db=db,
        user_id=current_user.id,
        session_id=session_id,
        user_message=message,
        ai_response=result or "",
        model_name=model_name,
        skill_used=skill_used,
        route_method=route_method,
    )

    return SkillChatResponse(
        skill_used=skill_used,
        route_method=route_method,
        result=result or "",
        model_name=model_name,
        session_id=session_id,
    )


# ================================================================
# 6. 智能 Skill 对话（流式 + 记忆）
# ================================================================
@router.post(
    "/chat/stream",
    summary="智能 Skill 对话（流式 + 记忆）",
    description="""
智能 Skill 对话的 SSE 流式版本，带对话记忆。

**SSE 事件类型：**
- `{"type": "start", "session_id": "...", "mode": "auto"}` — 开始
- `{"type": "token", "content": "..."}` — 逐 token 输出
- `{"type": "done", "skill_used": "...", "route_method": "..."}` — 完成
- `{"type": "error", "message": "..."}` — 错误

对话历史自动存入 Redis，聊天记录在流结束后存入 PostgreSQL。
""",
)
async def skill_chat_stream(
    request: SkillChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    message = request.message
    mode = request.mode
    session_id = request.session_id or str(uuid.uuid4())

    logger.info("POST /skills/chat/stream — user=%s, session=%s, mode=%s",
                current_user.username, session_id, mode)

    async def event_generator():
        collected_response = []
        skill_used = None
        route_method = mode

        yield _sse({"type": "start", "session_id": session_id, "mode": mode})

        try:
            async for chunk in chat_with_skill_memory_stream(
                message=message,
                session_id=session_id,
                mode=mode,
            ):
                if chunk:
                    collected_response.append(chunk)
                    yield _sse({"type": "token", "content": chunk})

            full_response = "".join(collected_response)

            # 尝试检测使用的技能（启发式）
            result_lower = full_response.lower()
            for s in get_all_skills():
                if s.display_name in full_response or s.name in result_lower:
                    skill_used = s.name
                    break

            # 流结束后保存到数据库
            if full_response.strip():
                await _save_skill_chat_history(
                    db=db,
                    user_id=current_user.id,
                    session_id=session_id,
                    user_message=message,
                    ai_response=full_response,
                    model_name=get_llm().model_name,
                    skill_used=skill_used,
                    route_method=route_method,
                )

            yield _sse({
                "type": "done",
                "skill_used": skill_used,
                "route_method": route_method,
                "session_id": session_id,
            })

        except Exception as e:
            logger.error("Skill chat stream failed: %s", e)
            yield _sse({"type": "error", "message": str(e)})

    return _sse_response(event_generator())


# ================================================================
# 辅助：通用对话（用于无记忆的回退场景）
# ================================================================
async def execute_skill_with_generic(message: str) -> str:
    from app.langchain_utils.chains import basic_chat
    return await basic_chat(
        message=message,
        system_prompt="你是一个有帮助的AI助手。用户的消息没有匹配到特定技能，请以通用方式回答。",
        temperature=0.7,
    )


async def execute_skill_stream_generic(message: str) -> AsyncIterator[str]:
    from app.langchain_utils.chains import basic_chat_stream
    async for chunk in basic_chat_stream(
        message=message,
        system_prompt="你是一个有帮助的AI助手。请以通用方式回答用户的问题。",
        temperature=0.7,
    ):
        yield chunk
