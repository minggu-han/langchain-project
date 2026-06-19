"""
聊天记录模型 - 保存用户与 AI 的对话历史

设计思路：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 每个用户有多条聊天记录（一对多关系）
2. 每次对话是一个 session（一次连续的问答）
3. 每条记录保存角色（user / assistant / system）和内容
4. 可选保存 Token 用量，用于成本分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    String, Text, Integer, DateTime, ForeignKey, Enum as SAEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.core.database import Base


class MessageRole(str, enum.Enum):
    """
    消息角色枚举

    LangChain 对话中的三种角色：
    - system：系统提示词，设定 AI 的行为和角色
    - human / user：用户的消息
    - ai / assistant：AI 的回复
    """
    SYSTEM = "system"        # 系统指令（设定 AI 角色）
    HUMAN = "human"          # 人类消息（等同于 user）
    USER = "user"            # 用户消息
    AI = "ai"                # AI 回复（等同于 assistant）
    ASSISTANT = "assistant"  # AI 回复

    # LangChain 用 human/ai，OpenAI 用 user/assistant
    # 统一支持两种命名，方便切换


class ChatHistory(Base):
    """
    聊天记录表

    字段说明：
    ┌──────────────┬──────────────┬────────────────────────────┐
    │ 字段名        │ 类型         │ 说明                       │
    ├──────────────┼──────────────┼────────────────────────────┤
    │ id           │ int (PK)     │ 自增主键                   │
    │ user_id      │ int (FK)     │ 用户 ID，外键关联 users 表 │
    │ session_id   │ str(100)     │ 会话 ID（一次连续对话）     │
    │ role         │ enum         │ 消息角色                    │
    │ content      │ text         │ 消息内容                    │
    │ token_count  │ int          │ Token 消耗（可选）          │
    │ model_name   │ str(100)     │ 使用的 LLM 模型名（可选）   │
    │ created_at   │ datetime     │ 创建时间                    │
    └──────────────┴──────────────┴────────────────────────────┘
    """
    __tablename__ = "chat_histories"

    # ==================== 基础字段 ====================
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # ForeignKey 定义外键约束，确保数据完整性
    # user_id 必须是 users 表中存在的 id
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),  # 用户删除时级联删除聊天记录
        nullable=False,
        index=True,  # 按用户查询聊天记录，需要索引
    )

    session_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,  # 按会话查询
    )

    # ==================== 消息内容 ====================
    role: Mapped[MessageRole] = mapped_column(
        SAEnum(MessageRole),  # 使用枚举类型，数据库存储为字符串
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,  # Text 类型无长度限制，适合长消息
        nullable=False,
    )

    # ==================== 元数据字段 ====================
    token_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,  # 可能无法获取 Token 数
    )

    model_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    # ==================== 表关系 ====================
    # 反向关联到 User 模型
    user = relationship(
        "User",
        back_populates="chat_histories",
    )

    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<ChatHistory(role='{self.role}', content='{content_preview}')>"
