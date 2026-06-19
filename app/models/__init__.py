"""
ORM 数据模型 - 数据库表结构定义

SQLAlchemy 2.0 新特性：
- 使用 Mapped[T] 类型注解，比旧的 Column() 更清晰
- 支持 dataclass-style 定义
- IDE 类型提示更准确
"""
from app.models.user import User
from app.models.chat import ChatHistory

__all__ = ["User", "ChatHistory"]
