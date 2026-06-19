"""
配置管理模块 - 使用 pydantic-settings 从环境变量和 .env 文件加载配置

为什么用 pydantic-settings？
- 自动从 .env 文件加载环境变量
- 类型验证（端口号必须是 int，URL 必须是 str）
- IDE 自动补全和类型检查
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    应用配置类

    每个属性对应一个环境变量，pydantic-settings 自动完成：
    1. 查找同名的环境变量（不区分大小写）
    2. 从 .env 文件加载（如果存在）
    3. 类型转换和验证
    """

    # ==================== 应用基础配置 ====================
    APP_NAME: str = "LangChain学习项目"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # ==================== 安全密钥 ====================
    # SECRET_KEY 用于签名 JWT Token，生产环境必须更换为随机字符串
    # 生成方式：openssl rand -hex 32
    SECRET_KEY: str = "dev-secret-key-change-in-production-abc123xyz"

    # ==================== 数据库配置 ====================
    # asyncpg 是 PostgreSQL 的高性能异步驱动
    # 格式：postgresql+asyncpg://用户名:密码@主机:端口/数据库名
    DATABASE_URL: str = (
        "postgresql+asyncpg://snowhan:Msi60067710@localhost:5432/security-db"
    )
    # 同步版本用于 Alembic（数据库迁移工具）
    DATABASE_URL_SYNC: str = (
        "postgresql://snowhan:Msi60067710@localhost:5432/security-db"
    )

    # ==================== Redis 配置 ====================
    # Redis 用于缓存和会话管理
    # 格式：redis://:密码@主机:端口/数据库编号
    REDIS_URL: str = "redis://:YourStrongRedisPassword123@localhost:6379/0"
    REDIS_PASSWORD: str = "YourStrongRedisPassword123"

    # ==================== JWT Token 配置 ====================
    # ACCESS_TOKEN：短期令牌，用于 API 请求认证
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30  # 30 分钟后过期
    # REFRESH_TOKEN：长期令牌，用于获取新的 ACCESS_TOKEN
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7  # 7 天后过期
    JWT_ALGORITHM: str = "HS256"  # HMAC-SHA256 签名算法

    # ==================== OpenAI 配置 ====================
    # LangChain 默认使用 OpenAI 模型，需要提供 API Key
    OPENAI_API_KEY: str = "sk-your-openai-api-key-here"
    # 模型选择：gpt-4o（最强）/ gpt-4o-mini（性价比）/ gpt-3.5-turbo（最快）
    OPENAI_MODEL: str = "gpt-4o-mini"
    # 如果你使用 OpenAI 兼容 API（如 Azure、本地模型），修改此 URL
    OPENAI_BASE_URL: str | None = None

    # ==================== Milvus 向量数据库 ====================
    # Milvus 是高性能分布式向量数据库，通过 Docker 运行
    # 连接配置（本地 standalone 模式，无需认证）
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    # Milvus 连接 URI（LangChain 集成用）
    MILVUS_URI: str = "http://localhost:19530"

    # ==================== 日志配置 ====================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = (
        "%(asctime)s.%(msecs)03d  %(levelname)-8s  %(name)-40s  %(message)s"
    )

    # pydantic-settings 配置：从 .env 文件加载
    model_config = {
        "env_file": ".env",         # 环境变量文件名
        "env_file_encoding": "utf-8",  # 文件编码
        "case_sensitive": True,     # 区分大小写
    }


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例

    @lru_cache() 是什么意思？
    - LRU = Least Recently Used（最近最少使用）缓存
    - 函数只会执行一次，后续调用直接返回缓存的结果
    - 这样 Settings() 只会实例化一次，避免重复读取 .env 文件
    - 这就是单例模式（Singleton）的 Pythonic 实现

    用法：
        settings = get_settings()
        print(settings.APP_NAME)
    """
    return Settings()
