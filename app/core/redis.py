"""
Redis 客户端模块 - 异步缓存与数据存储

Redis 在本项目中的用途：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Token 黑名单    —— 用户登出后将 JWT 加入黑名单，防止被重复使用
2. 速率限制        —— 限制 API 调用频率，防止滥用
3. 缓存 LLM 响应   —— 缓存相同的问答，节省 API 费用
4. 会话缓存        —— 缓存用户会话信息，减少数据库查询
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

为什么用 Redis？
- 内存存储，读写速度极快（微秒级）
- 支持多种数据结构（字符串、哈希、列表、集合、有序集合）
- 支持过期时间（TTL），适合缓存和临时数据
- 支持发布/订阅模式
"""
import redis.asyncio as aioredis
from app.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

# ================================================================
# 1. 创建 Redis 连接池
# ================================================================
# ConnectionPool 管理一组可复用的 Redis 连接
# 为什么用连接池？
# - 避免每次操作都建立新连接（TCP 握手开销很大）
# - 限制最大连接数，防止连接数爆炸
# - 自动处理断线重连
redis_pool = aioredis.ConnectionPool.from_url(
    settings.REDIS_URL,
    max_connections=20,         # 最大连接数
    decode_responses=True,      # 自动将字节解码为字符串
)

# ================================================================
# 2. 创建 Redis 客户端
# ================================================================
# 每次调用 get_redis() 返回一个新的 Redis 客户端实例
# 但底层共享同一个连接池
async def get_redis() -> aioredis.Redis:
    """
    FastAPI 依赖注入函数 - 获取 Redis 客户端

    用法：
        @app.get("/cache")
        async def get_cache(redis: Redis = Depends(get_redis)):
            value = await redis.get("my-key")
            return {"value": value}

    注意：
    - 使用 decode_responses=True，所以 get() 返回 str 而不是 bytes
    - 连接池在应用启动时创建，关闭时销毁
    """
    client = aioredis.Redis(connection_pool=redis_pool)
    try:
        yield client
    finally:
        # Redis 客户端会自动归还连接到连接池，不需要手动 close
        pass


# ================================================================
# 3. 便捷函数：获取原始 Redis 客户端（非依赖注入场景）
# ================================================================
async def get_redis_client() -> aioredis.Redis:
    """
    获取 Redis 客户端（用于非 FastAPI 依赖注入的场景）

    用法：
        redis = await get_redis_client()
        await redis.setex("key", 3600, "value")
    """
    return aioredis.Redis(connection_pool=redis_pool)


# ================================================================
# 4. 应用启动/关闭时管理 Redis 连接池
# ================================================================
async def init_redis():
    """
    初始化 Redis 连接池（应用启动时调用）

    实际上连接池在模块加载时已创建，
    这个函数用于验证 Redis 连接是否正常。
    """
    client = aioredis.Redis(connection_pool=redis_pool)
    try:
        await client.ping()
        logger.info("Redis connected: %s", settings.REDIS_URL)
    except Exception as e:
        logger.error("Redis connection failed: %s", e)
        raise


async def close_redis():
    """
    关闭 Redis 连接池（应用关闭时调用）

    释放所有连接，确保优雅关闭。
    """
    await redis_pool.disconnect()
    logger.info("Redis connection pool closed")
