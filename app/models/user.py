"""
用户模型 - 用户表结构定义

SQLAlchemy 2.0 声明式模型语法（推荐）：
- Mapped[T] 替代旧的 Column(T)
- mapped_column() 定义列属性
- relationship() 定义表关系
- 类型注解提供 IDE 自动补全
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Boolean, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class User(Base):
    """
    用户表

    字段说明：
    ┌─────────────┬──────────┬────────────────────────────────┐
    │ 字段名       │ 类型     │ 说明                           │
    ├─────────────┼──────────┼────────────────────────────────┤
    │ id          │ int (PK) │ 自增主键                       │
    │ username    │ str(50)  │ 用户名，唯一索引               │
    │ email       │ str(100) │ 邮箱，唯一索引                 │
    │ hashed_pass │ str(255) │ bcrypt 哈希后的密码             │
    │ full_name   │ str(100) │ 用户全名（可选）               │
    │ is_active   │ bool     │ 账户是否启用（默认 True）      │
    │ is_superuser│ bool     │ 是否为超级管理员（默认 False）  │
    │ created_at  │ datetime │ 创建时间（自动记录）           │
    │ updated_at  │ datetime │ 更新时间（自动更新）           │
    └─────────────┴──────────┴────────────────────────────────┘
    """
    __tablename__ = "users"  # 数据库中的表名

    # ==================== 基础字段 ====================
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # mapped_column 可以指定数据库列的类型和约束

    username: Mapped[str] = mapped_column(
        String(50),          # 最多 50 个字符
        unique=True,         # 唯一索引，防止重名
        nullable=False,      # 不允许为空
        index=True,          # 创建索引（登录时按用户名查询，需要索引加速）
    )

    email: Mapped[str] = mapped_column(
        String(100),
        unique=True,         # 唯一索引，防止重复注册
        nullable=False,
        index=True,
    )

    hashed_password: Mapped[str] = mapped_column(
        String(255),         # bcrypt 哈希固定 60 字符，255 留有余地
        nullable=False,
    )

    full_name: Mapped[Optional[str]] = mapped_column(
        String(100),         # Optional[str] = 允许为 NULL
        nullable=True,
    )

    # ==================== 状态字段 ====================
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,        # 新用户默认启用
        nullable=False,
    )

    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,       # 新用户默认不是管理员
        nullable=False,
    )

    # ==================== 时间戳字段 ====================
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,  # 创建时自动设置为当前 UTC 时间
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,  # 更新时自动刷新时间
        nullable=False,
    )

    # ==================== 表关系（Relationship） ====================
    # relationship 建立 ORM 层面的关联，不是数据库外键
    # back_populates 双向关联：User.chat_histories <-> ChatHistory.user
    # lazy="dynamic" 延迟加载，返回 Query 对象，可以继续过滤
    # cascade="all, delete-orphan" 级联删除用户时，同时删除其聊天记录
    chat_histories = relationship(
        "ChatHistory",
        back_populates="user",
        lazy="dynamic",         # 延迟加载，避免查询所有聊天记录
        cascade="all, delete-orphan",  # 级联操作
    )

    def __repr__(self) -> str:
        """对象的字符串表示，调试用"""
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"
