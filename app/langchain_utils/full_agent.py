"""
LangChain 终极篇 —— Full-Featured Agent（全功能智能代理）

全功能 Agent 整合了项目的所有能力：
┌──────────────────────────────────────────────────────────────────┐
│                    Full-Featured Agent                            │
│  自主决策：Memory / Skills / Tools / RAG / Chain                  │
└──────────────────────────────────────────────────────────────────┘
         │            │            │            │            │
    ┌────▼────┐  ┌───▼───┐  ┌───▼───┐  ┌───▼───┐  ┌───▼───┐
    │ Memory  │  │Skills │  │ Tools  │  │  RAG  │  │ Chain │
    │Redis历史│  │8个技能│  │3个工具 │  │Milvus │  │通用对话│
    └─────────┘  └───────┘  └───────┘  └───────┘  └───────┘

Agent 的决策循环（ReAct + Function Calling）：
  Thought → Action(选择工具) → Observation(结果) → Thought → Final Answer

可调用的能力（注册为 OpenAI Functions）：
├── use_skill         → 执行预置技能（code-review / translate / summarize ...）
├── search_documents  → RAG 文档检索（Milvus）
├── calculator        → 数学计算
├── get_current_time  → 获取时间
└── text_statistics   → 文本统计

Agent 自主决定：
- 什么时候用 Skill vs Tool vs RAG
- 是单独使用还是组合使用
- 什么时候直接回答（不需要任何工具）
"""
import json
from typing import AsyncIterator
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent

from app.langchain_utils.llm_factory import get_llm
from app.langchain_utils.memory import get_session_history
from app.langchain_utils.skills import (
    get_skill,
    get_all_skills,
    SKILL_REGISTRY,
    execute_skill,
)
from app.langchain_utils.tools import calculator, get_current_time, text_statistics
from app.core.logging import get_logger

logger = get_logger(__name__)


# ================================================================
# 1. 自定义工具：use_skill（异步）
# ================================================================
_available_skills = get_all_skills()
_SKILL_NAMES = [s.name for s in _available_skills]

# 构建技能描述供 LLM 参考
_SKILL_DESC_BRIEF = "\n".join(
    f"- **{s.name}**（{s.display_name}）：{s.description}"
    for s in _available_skills
)
_SKILL_DESC_DETAIL = "\n\n".join(
    f"### {s.display_name}（`{s.name}`）\n{s.description}\n\n{s.system_prompt[:300]}..."
    for s in _available_skills
)


async def _use_skill_async(skill_name: str, input_text: str) -> str:
    """
    使用预置技能完成专业任务。让技能按照其预设的专家步骤来处理用户的输入。

    适用场景：
    - 用户需要代码审阅 → skill_name="code-review"
    - 用户需要翻译 → skill_name="translate"
    - 用户需要文本摘要 → skill_name="summarize"
    - 用户需要生成SQL → skill_name="sql-generator"
    - 用户需要写邮件 → skill_name="email-writer"
    - 用户需要解释概念 → skill_name="explain-concept"
    - 用户需要数据分析建议 → skill_name="data-analyzer"
    - 用户需要面试模拟 → skill_name="interview-coach"

    Args:
        skill_name: 技能名称
        input_text: 传递给技能的输入（用户的原始需求文本）
    """
    skill = get_skill(skill_name)
    if skill is None:
        return f"技能 '{skill_name}' 不存在。可用技能：{', '.join(_SKILL_NAMES)}"

    logger.info("use_skill tool: skill=%s, input_len=%d", skill_name, len(input_text))
    result = await execute_skill(skill, input_text)
    logger.info("use_skill tool: skill=%s completed, result_len=%d", skill_name, len(result))
    return result


use_skill = StructuredTool.from_function(
    coroutine=_use_skill_async,
    name="use_skill",
    description=f"""使用预置技能完成专业任务。

可用技能列表：
{_SKILL_DESC_BRIEF}

选择最匹配用户需求的技能，将用户的原始需求作为 input_text 传入。""",
)


# ================================================================
# 2. 自定义工具：search_documents（异步 RAG）
# ================================================================
async def _search_documents_async(query: str, collection_name: str = "default") -> str:
    """
    在已索引的文档知识库中搜索相关内容。

    适用场景：
    - 用户问的问题可能在之前上传的文档中有答案
    - 用户要求基于特定文档回答

    Args:
        query: 搜索查询
        collection_name: 文档集合名称，默认为 "default"
    """
    from app.langchain_utils.rag import rag_query

    logger.info("search_documents tool: query='%.60s...', collection=%s", query, collection_name)

    result_dict = await rag_query(query=query, collection_name=collection_name, top_k=4)

    answer = result_dict.get("answer", "未找到相关文档")
    sources = result_dict.get("sources", [])
    if sources:
        answer += "\n\n---\n**引用来源：**\n"
        for i, src in enumerate(sources[:3], 1):
            answer += f"{i}. {src[:200]}...\n"

    return answer


search_documents = StructuredTool.from_function(
    coroutine=_search_documents_async,
    name="search_documents",
    description="在已上传的文档知识库中搜索相关内容。当用户的问题可能在某份文档中有答案时使用。query 参数用自然语言描述要搜索的内容。",
)


# ================================================================
# 3. 全功能 Agent 的 System Prompt
# ================================================================
_FULL_AGENT_SYSTEM_PROMPT = """你是一个全功能智能助手（Full-Featured Agent），可以自主决定使用以下能力来完成任务：

## 🎯 能力一览

### Skills（技能模板）
{skill_list}

### Tools（工具函数）
- **calculator**：执行数学计算（加减乘除、开方、三角函数等）
- **get_current_time**：获取当前日期和时间
- **text_statistics**：统计文本的字符数、单词数、行数等

---

## 🧠 工作原则

1. **分析用户意图**：先理解用户想要什么
2. **选择合适的工具**：
   - 代码审阅 / 翻译 / 摘要 / SQL生成 / 邮件 / 概念解释 / 数据分析 / 面试 → 用 **use_skill**
   - 数学计算 → 用 **calculator**
   - 时间日期 → 用 **get_current_time**
   - 文本统计 → 用 **text_statistics**
   - 普通闲聊 / 简单问答 → **直接回复**，不用任何工具
3. **组合使用**：复杂任务可以先后调用多个工具
   - 例："审阅代码并计算时间复杂度" → use_skill("code-review",...) + calculator(...)
4. **利用对话历史**：结合之前的对话理解上下文
5. **RAG 上下文**：如果用户消息下方附带了「已检索到的相关文档」，请严格基于这些文档内容来回答。文档中没有的信息请诚实告知用户。
6. **最终输出**：整合所有工具返回的结果，用 Markdown 格式给出完整答案

---

## 📋 可用技能详情

{skill_details}

---

现在请处理用户的消息。结合对话历史理解上下文，自主决定使用哪些能力。"""


# ================================================================
# 4. 自动 RAG 预检索
# ================================================================
async def _auto_rag_search(message: str, top_k: int = 4) -> tuple[str | None, list]:
    """
    每条用户消息自动跑 Milvus 相似度搜索。

    有结果 → 返回格式化上下文，注入到用户消息
    无结果 / Milvus 不可用 → 返回 None，正常对话

    Returns:
        (context_string | None, source_list)
    """
    try:
        from app.langchain_utils.rag import get_vector_store

        vector_store = get_vector_store("default")
        docs = vector_store.similarity_search(message, k=top_k)

        if not docs:
            logger.debug("Auto RAG: no similar documents found")
            return None, []

        # 格式化上下文
        context_parts = []
        sources = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知")
            content = doc.page_content[:500]
            context_parts.append(f"[文档{i} | 来源: {source}]\n{content}")
            sources.append(source)

        context = "\n\n---\n\n".join(context_parts)
        logger.info("Auto RAG: found %d relevant docs, context_len=%d", len(docs), len(context))
        return context, sources

    except Exception as e:
        logger.warning("Auto RAG search failed (Milvus may not be running): %s", e)
        return None, []


# ================================================================
# 5. 构建全功能 Agent
# ================================================================
def _build_agent(temperature: float = 0.3):
    """
    构建全功能 Agent

    创建 OpenAI Functions Agent，注册所有工具：
    - use_skill（8 个子技能）
    - search_documents（RAG）
    - calculator、get_current_time、text_statistics
    """
    llm = get_llm(temperature=temperature)

    # 所有可用工具（RAG 已改为自动预检索，不再作为工具）
    all_tools = [
        use_skill,
        calculator,
        get_current_time,
        text_statistics,
    ]

    system_prompt = _FULL_AGENT_SYSTEM_PROMPT.format(
        skill_list=_SKILL_DESC_BRIEF,
        skill_details=_SKILL_DESC_DETAIL,
    )

    logger.debug("Building full agent: %d tools, system_prompt_len=%d",
                 len(all_tools), len(system_prompt))

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm, all_tools, prompt)

    from app.config import get_settings
    _settings = get_settings()

    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=_settings.DEBUG,
        max_iterations=15,           # 全功能 Agent 可能需要更多步骤
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    return agent_executor


# ================================================================
# 5. 同步执行
# ================================================================
async def run_full_agent(
    message: str,
    session_id: str = "default",
    temperature: float = 0.3,
) -> dict:
    """
    运行全功能 Agent（同步模式）

    流程：
    1. 自动 Milvus 相似度搜索 → 有结果注入上下文
    2. 构建 Agent（带 Skills + Tools）
    3. 从 Redis 加载对话历史
    4. Agent 自主决策：选择 Skills / Tools / 直接回复
    5. 返回结果（含中间步骤）

    Args:
        message: 用户消息
        session_id: 会话 ID
        temperature: LLM 温度

    Returns:
        {
            "answer": str,
            "steps": [...],
            "total_steps": int,
            "rag_used": bool,
        }
    """
    # ── 步骤 0：自动 RAG 预检索 ──
    rag_context, rag_sources = await _auto_rag_search(message)

    # 有 RAG 结果 → 注入到用户消息前面
    if rag_context:
        augmented_message = (
            f"【已检索到的相关文档 — 请严格基于以下内容回答】\n\n"
            f"{rag_context}\n\n"
            f"---\n"
            f"【用户问题】\n{message}"
        )
    else:
        augmented_message = message

    agent_executor = _build_agent(temperature=temperature)

    # 从 Redis 加载历史消息
    history = get_session_history(session_id)
    history_messages = history.messages
    logger.info("run_full_agent: session=%s, history_msgs=%d, rag=%s, message='%.80s...'",
                session_id, len(history_messages), bool(rag_context), message)

    result = await agent_executor.ainvoke({
        "input": augmented_message,
        "history": history_messages,
    })

    # 提取中间步骤
    steps = []
    for action, observation in result.get("intermediate_steps", []):
        steps.append({
            "tool": action.tool,
            "tool_input": str(action.tool_input)[:300],
            "observation": str(observation)[:500],
        })

    answer = result.get("output", "")
    logger.info("run_full_agent completed: %d steps, answer_len=%d, rag_used=%s",
                len(steps), len(answer), bool(rag_context))

    # 保存到 Redis（保存原始消息，不带 RAG 上下文）
    history.add_message(HumanMessage(content=message))
    history.add_message(AIMessage(content=answer))

    return {
        "answer": answer,
        "steps": steps,
        "total_steps": len(steps),
        "rag_used": bool(rag_context),
    }


# ================================================================
# 6. 流式执行
# ================================================================
async def run_full_agent_stream(
    message: str,
    session_id: str = "default",
    temperature: float = 0.3,
) -> AsyncIterator[dict]:
    """
    运行全功能 Agent（流式模式）

    包含：自动 RAG 预检索 + Agent 流式输出

    Yields:
        {"type": "rag_start"} / {"type": "rag_result", "count": N} / {"type": "rag_none"}
        {"type": "token", "content": "..."}
        {"type": "tool_start", "tool": "...", "input": "..."}
        {"type": "tool_end", "tool": "...", "output": "..."}
        {"type": "done", "rag_used": bool}
    """
    # ── 步骤 0：自动 RAG 预检索 ──
    yield {"type": "rag_start"}
    rag_context, rag_sources = await _auto_rag_search(message)

    if rag_context:
        yield {"type": "rag_result", "count": len(rag_sources)}
        augmented_message = (
            f"【已检索到的相关文档 — 请严格基于以下内容回答】\n\n"
            f"{rag_context}\n\n"
            f"---\n"
            f"【用户问题】\n{message}"
        )
    else:
        yield {"type": "rag_none"}
        augmented_message = message

    agent_executor = _build_agent(temperature=temperature)

    # 加载历史
    history = get_session_history(session_id)
    history_messages = history.messages
    logger.info("run_full_agent_stream: session=%s, history_msgs=%d, rag=%s, message='%.60s...'",
                session_id, len(history_messages), bool(rag_context), message)

    full_answer_parts = []
    current_tool = None

    try:
        async for event in agent_executor.astream_events(
            {"input": augmented_message, "history": history_messages},
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    content = chunk.content
                    # 跳过工具调用参数（JSON），只发送普通文本
                    if isinstance(content, str) and not (
                        content.strip().startswith('{"') or
                        content.strip().startswith('{ "')
                    ):
                        full_answer_parts.append(content)
                        yield {"type": "token", "content": content}

            elif kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                tool_input = event["data"].get("input", {})
                current_tool = tool_name
                yield {
                    "type": "tool_start",
                    "tool": tool_name,
                    "input": str(tool_input)[:500],
                }

            elif kind == "on_tool_end":
                output = str(event["data"].get("output", ""))[:500]
                yield {
                    "type": "tool_end",
                    "tool": current_tool or "unknown",
                    "output": output,
                }
                current_tool = None

    except Exception as e:
        logger.error("Full agent stream error: %s", e)
        yield {"type": "error", "message": str(e)}
        return

    full_answer = "".join(full_answer_parts)

    # 保存到 Redis
    if full_answer.strip():
        history.add_message(HumanMessage(content=message))
        history.add_message(AIMessage(content=full_answer))

    yield {"type": "done", "rag_used": bool(rag_context)}

    logger.info("run_full_agent_stream completed: answer_len=%d, rag_used=%s",
                len(full_answer), bool(rag_context))


# ================================================================
# 7. 初始化日志
# ================================================================
logger.info(
    "Full Agent module loaded: %d tools (%d skills + RAG + %d util tools)",
    len(_SKILL_NAMES) + 4,
    len(_SKILL_NAMES),
    3,
)
