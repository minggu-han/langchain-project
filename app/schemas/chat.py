"""
聊天相关 Pydantic Schema - LangChain 对话接口的数据结构

本项目的核心 API：
1. /chat              —— 基础对话（Prompt Template + LLM Chain）
2. /chat/with-memory  —— 带记忆的对话（ConversationBufferMemory）
3. /chat/rag          —— RAG 检索增强生成
4. /chat/agent        —— Agent 智能代理（工具调用）
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# ================================================================
# 1. 基础对话请求
# ================================================================
class ChatRequest(BaseModel):
    """
    基础对话请求

    支持两种模式：
    1. 简单模式：只提供 message
    2. 高级模式：提供 system_prompt 自定义 AI 行为
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="用户的消息",
        examples=["请用 Python 写一个快速排序算法"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="会话 ID，用于关联多轮对话。不传则自动生成新的会话",
        examples=["session_abc123"],
    )
    system_prompt: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="系统提示词，用于设定 AI 的角色和行为。例如：'你是一个 Python 专家'",
        examples=["你是一个精通 Python 编程的助手，回答要包含代码示例"],
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,  # greater than or equal
        le=2.0,  # less than or equal
        description="LLM 温度参数：0=精确/保守，1=创造性，>1=更随机",
    )


# ================================================================
# 2. 对话响应
# ================================================================
class ChatResponse(BaseModel):
    """
    LLM 对话的响应

    包含完整的对话信息和元数据
    """
    session_id: str = Field(..., description="会话 ID")
    user_message: str = Field(..., description="用户的原始消息")
    ai_response: str = Field(..., description="AI 的回复")
    model_name: str = Field(..., description="使用的 LLM 模型名称")
    tokens_used: Optional[int] = Field(default=None, description="本次对话消耗的 Token 数")
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ================================================================
# 3. 聊天历史响应
# ================================================================
class ChatHistoryItem(BaseModel):
    """聊天历史中的一条消息"""
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatHistoryResponse(BaseModel):
    """某个会话的完整聊天历史"""
    session_id: str
    messages: List[ChatHistoryItem]
    total_tokens: Optional[int] = Field(default=None, description="该会话的总 Token 消耗")


class ChatSessionList(BaseModel):
    """用户的所有聊天会话列表"""
    sessions: List[str] = Field(..., description="会话 ID 列表")
    total: int = Field(..., description="会话总数")


# ================================================================
# 4. RAG（检索增强生成）请求
# ================================================================
class RAGQueryRequest(BaseModel):
    """
    RAG 查询请求

    RAG 工作流程：
    1. 用户在 query 中提一个问题
    2. 系统在已索引的文档中检索相关内容
    3. 将检索结果作为上下文，调用 LLM 生成答案
    4. 返回答案 + 引用来源
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="要查询的问题",
        examples=["什么是 LangChain 的 Chain？它和 Agent 有什么区别？"],
    )
    collection_name: Optional[str] = Field(
        default="default",
        description="Chroma 集合名称（相当于文档库的名称）",
    )
    top_k: Optional[int] = Field(
        default=4,
        ge=1,
        le=20,
        description="检索返回的相关文档片段数量",
    )


# ================================================================
# 5. Agent 查询请求
# ================================================================
class AgentQueryRequest(BaseModel):
    """
    Agent 智能代理查询请求

    Agent 与普通对话的区别：
    - Agent 可以自主决定使用哪些工具
    - Agent 可以多步推理（Thought → Action → Observation → ...）
    - Agent 会不断迭代直到找到答案或放弃
    """
    message: str = Field(
        ...,
        min_length=1,
        description="给 Agent 的任务描述",
        examples=["帮我查一下今天的天气，然后翻译成英文"],
    )
    session_id: Optional[str] = Field(default=None)
    max_iterations: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Agent 最大迭代次数（防止无限循环）",
    )
