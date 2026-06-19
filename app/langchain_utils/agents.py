"""
LangChain 高级篇 —— Agent（智能代理）

什么是 Agent？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent 是能够自主决策和执行任务的 AI 智能体。

Chain vs Agent 的区别：
┌──────────┬─────────────────────────┬─────────────────────────┐
│ 维度      │ Chain（链）             │ Agent（代理）           │
├──────────┼─────────────────────────┼─────────────────────────┤
│ 执行流程  │ 固定的、预定义的         │ 动态的、上下文决定的     │
│ 工具使用  │ 不支持或固定             │ 自主选择工具             │
│ 决策能力  │ 无                       │ 观察 → 思考 → 行动      │
│ 适用场景  │ 翻译、总结、格式转换     │ 复杂任务、多步推理       │
└──────────┴─────────────────────────┴─────────────────────────┘

Agent 的工作循环（ReAct 模式）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Thought（思考）: 我现在需要做什么？
2. Action（行动）: 调用哪个工具？传什么参数？
3. Observation（观察）: 工具返回了什么结果？
4. (重复) 基于观察结果，继续思考下一步...
5. Final Answer（最终答案）: 任务完成，返回最终答案

ReAct = Reasoning + Acting（推理 + 行动）
这是目前最流行的 Agent 模式。

LangChain Agent 类型：
┌──────────────────────┬──────────────────────────────────────┐
│ 类型                  │ 说明                                 │
├──────────────────────┼──────────────────────────────────────┤
│ OpenAI Functions      │ 使用 OpenAI 的 Function Calling API │
│ ReAct                 │ 经典的 思考-行动-观察 循环           │
│ Structured Chat       │ 支持结构化输入输出的 Agent           │
│ Self-Ask with Search  │ 自我提问 + 搜索                     │
└──────────────────────┴──────────────────────────────────────┘
"""
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from app.langchain_utils.llm_factory import get_llm
from app.langchain_utils.tools import get_tools_for_agent
from app.core.logging import get_logger

logger = get_logger(__name__)


# ================================================================
# 1. OpenAI Functions Agent（推荐方式）
# ================================================================
async def run_functions_agent(
    message: str,
    tool_names: list[str] | None = None,
    system_prompt: str | None = None,
):
    """
    使用 OpenAI Functions Agent 执行任务

    OpenAI Functions Agent 使用 OpenAI 的 Function Calling API：
    - LLM 直接输出结构化的函数调用（JSON 格式）
    - 不需要复杂的 Prompt 工程
    - 可靠性高，幻觉少

    工作原理：
    1. 用户输入任务描述
    2. Agent 分析任务，决定使用哪些工具
    3. Agent 调用工具并获取结果
    4. Agent 判断是否需要更多工具调用
    5. 生成最终答案

    Args:
        message: 任务描述（用自然语言告诉 Agent 要做什么）
        tool_names: 指定可用的工具名称列表，None 表示使用所有工具
        system_prompt: Agent 的系统提示词
    """
    # 获取 LLM 实例
    llm = get_llm(temperature=0.3)  # Agent 用低温度，减少幻觉

    # 获取工具列表
    tools = get_tools_for_agent(tool_names)
    tool_name_list = [t.name for t in tools]
    logger.info("run_functions_agent: message='%.80s...', tools=%s", message, tool_name_list)

    # 构建 System Prompt
    if system_prompt is None:
        system_prompt = """你是一个智能助手，可以使用各种工具来帮助用户完成任务。

工作原则：
1. 分析用户的需求，拆解为可执行的步骤
2. 选择最合适的工具来完成每个步骤
3. 基于工具返回的结果，决定下一步行动
4. 最终给出完整的答案

你可以使用的工具：
- calculator：执行数学计算
- get_current_time：获取当前时间
- text_statistics：统计文本信息"""

    # 创建 Agent
    # create_openai_functions_agent 创建一个使用 OpenAI Function Calling 的 Agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    agent = create_openai_functions_agent(llm, tools, prompt)

    # AgentExecutor 负责管理 Agent 的执行循环
    # max_iterations=10 防止 Agent 无限循环
    from app.config import get_settings
    _settings = get_settings()
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=_settings.DEBUG,           # 开发环境打印详细执行过程
        max_iterations=10,      # 最大迭代次数
        handle_parsing_errors=True,  # 处理解析错误
        return_intermediate_steps=True,  # 返回中间步骤（学习用）
    )

    # 执行 Agent
    logger.debug("run_functions_agent: executing with %d tools...", len(tools))
    result = await agent_executor.ainvoke({"input": message})

    # 提取中间步骤（展示 Agent 的思考过程）
    steps = []
    for action, observation in result.get("intermediate_steps", []):
        steps.append({
            "tool": action.tool,
            "tool_input": action.tool_input,
            "observation": str(observation)[:200],  # 截断长结果
        })

    logger.info("run_functions_agent completed: %d steps, answer='%.100s...'",
                len(steps), result["output"])

    return {
        "answer": result["output"],
        "steps": steps,
        "total_steps": len(steps),
    }


# ================================================================
# 2. 多步推理 Agent（展示思考链）
# ================================================================
async def run_reasoning_agent(message: str):
    """
    执行需要多步推理的复杂任务

    这个 Agent 展示 Agent 的核心价值：
    - 自动拆解复杂问题
    - 顺序执行多个步骤
    - 基于前一步的结果决定下一步

    示例任务：
    "帮我计算 (123 * 456) + (789 / 3) - sqrt(144)，然后计算最终结果的平方根"
    → Agent 会：
      1. 计算 123 * 456
      2. 计算 789 / 3
      3. 计算 sqrt(144)
      4. 汇总结果
      5. 计算平方根
    """
    llm = get_llm(temperature=0.2)

    logger.info("run_reasoning_agent: message='%.80s...'", message)

    # 只给计算器和统计工具
    tools = get_tools_for_agent(["calculator", "text_statistics"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个数学问题解决助手。遇到复杂的数学问题时，请：

1. 先分析问题的结构
2. 将问题拆解为小的计算步骤
3. 逐步计算每个子问题
4. 最后汇总结果

注意：每个计算步骤都使用 calculator 工具。"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm, tools, prompt)

    from app.config import get_settings
    _settings = get_settings()
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=_settings.DEBUG,
        max_iterations=10,
        handle_parsing_errors=True,
    )

    result = await agent_executor.ainvoke({"input": message})
    logger.info("run_reasoning_agent completed: answer='%.100s...'", result["output"])
    return result["output"]


# ================================================================
# 3. 获取 Agent 可以使用的工具信息
# ================================================================
def get_available_tools_info() -> list[dict]:
    """
    获取所有可用工具的信息（用于 API 文档展示）

    每个工具返回：
    - name：工具名称
    - description：工具描述
    - args：参数信息
    """
    tools = get_tools_for_agent()
    info = []
    for t in tools:
        info.append({
            "name": t.name,
            "description": t.description,
            "args_schema": str(t.args_schema.schema()) if t.args_schema else "无参数",
        })
    return info


# ================================================================
# 4. Agent 流式输出（astream_events）
# ================================================================
from typing import AsyncIterator


async def run_functions_agent_stream(
    message: str,
    tool_names: list[str] | None = None,
) -> AsyncIterator[dict]:
    """
    Agent 流式 — 展示思考过程和工具调用

    使用 astream_events() 而非 astream()，可以监听：
    - on_chat_model_stream: LLM token 输出
    - on_tool_start: Agent 开始调用工具
    - on_tool_end: 工具返回结果

    事件格式：
    - {"type": "start"}
    - {"type": "token", "content": "..."}      — LLM 逐 token 输出
    - {"type": "tool_start", "tool": "...", "input": "..."}
    - {"type": "tool_end", "tool": "...", "output": "..."}
    - {"type": "done"}

    用法：
        async for event in run_functions_agent_stream("计算 1+1"):
            if event["type"] == "token":
                print(event["content"], end="", flush=True)
    """
    llm = get_llm(temperature=0.3)
    tools = get_tools_for_agent(tool_names)

    logger.info("run_functions_agent_stream: message='%.80s...', tools=%s",
                message, [t.name for t in tools])

    system_prompt = """你是一个智能助手，可以使用各种工具来帮助用户完成任务。
工作原则：
1. 分析用户的需求，拆解为可执行的步骤
2. 选择最合适的工具来完成每个步骤
3. 基于工具返回的结果，决定下一步行动
4. 最终给出完整的答案"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm, tools, prompt)

    from app.config import get_settings
    _settings = get_settings()
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # 不用 verbose，我们自己控制事件输出
        max_iterations=10,
        handle_parsing_errors=True,
    )

    yield {"type": "start",
           "message": f"Agent 已就绪，可用工具：{', '.join(t.name for t in tools)}"}

    current_tool = None
    try:
        async for event in agent_executor.astream_events(
            {"input": message},
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    yield {"type": "token", "content": chunk.content}

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
        logger.error("Agent stream error: %s", e)
        yield {"type": "error", "message": str(e)}

    yield {"type": "done"}
    logger.info("run_functions_agent_stream completed")
