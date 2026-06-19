"""
Pydantic 数据验证模型（Schema）

Pydantic vs SQLAlchemy：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SQLAlchemy Model  —— 数据库表结构（ORM 层）
Pydantic Schema   —— API 请求/响应结构（接口层）

为什么要分开？
- 数据库模型 ≠ API 接口结构（安全隔离）
- 密码哈希绝不能通过 API 返回
- 请求验证和响应序列化是不同职责
"""
from app.schemas.user import (
    UserCreate,
    UserLogin,
    UserResponse,
    UserUpdate,
    TokenResponse,
    TokenRefresh,
)
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatHistoryResponse,
    RAGQueryRequest,
    AgentQueryRequest,
    ChatSessionList,
)

__all__ = [
    # 用户
    "UserCreate", "UserLogin", "UserResponse", "UserUpdate",
    "TokenResponse", "TokenRefresh",
    # 聊天
    "ChatRequest", "ChatResponse", "ChatHistoryResponse",
    "RAGQueryRequest", "AgentQueryRequest", "ChatSessionList",
]
