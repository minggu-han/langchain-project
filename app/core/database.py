"""
数据库引擎与会话管理 - SQLAlchemy 2.0 异步模式

核心概念：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Engine（引擎）      —— 数据库连接池，整个应用只创建一个
2. Session（会话）     —— 一次数据库操作的上下文，每次请求创建新的
3. Base（模型基类）    —— 所有 ORM 模型的父类，定义表结构
4. AsyncSession（异步会话）—— 非阻塞的数据库会话，配合 async/await 使用
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

为什么用异步？
- FastAPI 是异步框架，同步数据库操作会阻塞事件循环
- asyncpg 是 PostgreSQL 的异步驱动，性能远高于同步驱动
- 在高并发场景下，异步可以同时处理多个请求
"""
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase
from app.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

# ================================================================
# 1. 创建异步引擎（Engine）
# ================================================================
# create_async_engine 创建一个异步数据库引擎
# - echo=True：打印所有 SQL 语句（调试用）
# - pool_size=10：连接池大小，最多同时保持 10 个连接
# - max_overflow=20：连接池溢出时额外创建的连接数（最多 30 个并发连接）
# - pool_pre_ping=True：每次从连接池取出连接时先 ping 一下，确保连接有效
#   （避免因为数据库重启导致的 "connection closed" 错误）
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,                 # SQL 日志通过 logging 模块输出
    pool_size=10,               # 连接池基础大小
    max_overflow=20,            # 额外连接数
    pool_pre_ping=True,         # 连接健康检查
)

# ================================================================
# 2. 创建异步会话工厂（Session Factory）
# ================================================================
# async_sessionmaker 是一个工厂函数，每次调用返回一个新的 AsyncSession
# - expire_on_commit=False：提交后不使对象过期
#   （这样在 commit 后还可以访问对象的属性，而不需要重新查询）
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,        # 指定使用 AsyncSession
    expire_on_commit=False,     # 提交后不自动过期
)


# ================================================================
# 3. 声明式基类（Declarative Base）
# ================================================================
# 所有 ORM 模型都继承这个 Base
# Base.metadata 包含所有表的信息，Alembic 用它来生成迁移
class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类 - 所有数据表模型都继承它"""
    pass


# ================================================================
# 4. FastAPI 依赖：获取数据库会话
# ================================================================
async def get_db() -> AsyncSession:
    """
    FastAPI 依赖注入函数 - 为每个请求创建一个新的数据库会话

    用法：
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(User))
            return result.scalars().all()

    工作原理：
    1. FastAPI 调用 get_db()，通过 AsyncSessionLocal() 创建新的 AsyncSession
    2. 请求处理完成后（无论成功或失败），自动关闭会话
    3. yield 之前的代码在请求开始时执行
    4. yield 之后的代码在请求结束时执行（清理资源）

    这就是 FastAPI 依赖注入系统的威力：
    - 路由函数不需要知道数据库连接的细节
    - 连接管理集中在一处，修改方便
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            # 请求正常结束时，提交事务
            await session.commit()
        except Exception:
            # 发生异常时回滚，避免脏数据写入
            await session.rollback()
            raise
        finally:
            # 无论成功与否，关闭会话（归还连接到连接池）
            await session.close()


# ================================================================
# 5. 数据库初始化（创建所有表）
# ================================================================
async def init_db():
    """
    创建所有 ORM 模型对应的数据库表

    注意：这只适合开发环境和学习用途！
    生产环境应该使用 Alembic 进行数据库迁移。

    Alembic 的作用：
    - 版本化管理数据库结构变更
    - 支持升级（upgrade）和降级（downgrade）
    - 记录每次变更的历史，方便团队协作
    """
    # 导入所有模型，确保它们被注册到 Base.metadata
    from app.models.user import User  # noqa: F401
    from app.models.chat import ChatHistory  # noqa: F401

    logger.info("Creating database tables...")
    async with async_engine.begin() as conn:
        # create_all 创建所有未存在的表
        # 如果表已存在，则跳过（不会删除已有数据）
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready")
