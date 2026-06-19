"""
速率限制中间件 — 基于 Redis 滑动窗口算法

原理：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
使用 Redis 的 sorted set 实现滑动窗口限流：
1. 每个请求以当前时间戳为 score 加入 sorted set
2. 删除窗口外的旧记录
3. 统计窗口内的请求数
4. 超过限制则返回 429 Too Many Requests

限制策略：
┌──────────────┬──────────────────────────────┐
│ 路由前缀      │ 限制                          │
├──────────────┼──────────────────────────────┤
│ /api/v1/auth │ 每分钟 10 次（防暴力破解）     │
│ /api/v1/chat │ 每分钟 30 次（防滥用 LLM）     │
│ 其他          │ 每分钟 60 次                  │
└──────────────┴──────────────────────────────┘
"""
import time
import redis.asyncio as aioredis
from fastapi import Request, HTTPException
from app.core.logging import get_logger
from app.config import get_settings

settings = get_settings()
logger = get_logger(__name__)

# 速率限制配置
RATE_LIMITS = {
    "/api/v1/auth": {"max_requests": 10, "window_seconds": 60},
    "/api/v1/chat": {"max_requests": 30, "window_seconds": 60},
    "/api/v1/documents": {"max_requests": 20, "window_seconds": 60},
    "default": {"max_requests": 60, "window_seconds": 60},
}


def _get_limit_config(path: str) -> dict:
    """根据请求路径获取限流配置"""
    for prefix, config in RATE_LIMITS.items():
        if prefix != "default" and path.startswith(prefix):
            return config
    return RATE_LIMITS["default"]


async def check_rate_limit(request: Request) -> None:
    """
    检查请求是否超过速率限制

    使用 Redis sorted set 实现滑动窗口：
    - key: rate_limit:{user_or_ip}:{path_prefix}
    - member: 请求唯一标识（时间戳 + 随机后缀）
    - score: 请求时间戳（用于窗口过滤）

    Raises:
        HTTPException 429: 超过速率限制
    """
    path = request.url.path

    # 跳过非 API 路由
    if not path.startswith("/api/"):
        return

    # 标识符：优先用已认证用户，否则用 IP
    user = getattr(request.state, "user", None)
    identifier = user.username if user else request.client.host

    config = _get_limit_config(path)
    max_req = config["max_requests"]
    window = config["window_seconds"]

    redis_key = f"rate_limit:{identifier}:{path.split('/')[3] if len(path.split('/')) > 3 else 'other'}"

    try:
        import redis.asyncio as aioredis
        redis = aioredis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        now = time.time()
        window_start = now - window

        async with redis.pipeline(transaction=True) as pipe:
            # 1. 删除窗口外的旧记录
            pipe.zremrangebyscore(redis_key, 0, window_start)
            # 2. 添加当前请求
            pipe.zadd(redis_key, {f"{now}-{id(request)}": now})
            # 3. 设置 key 过期时间
            pipe.expire(redis_key, window + 1)
            # 4. 统计窗口内请求数
            pipe.zcard(redis_key)
            _, _, _, count = await pipe.execute()

        await redis.aclose()

        if count > max_req:
            logger.warning("Rate limit exceeded: %s %s (count=%d, limit=%d)",
                           identifier, path, count, max_req)
            raise HTTPException(
                status_code=429,
                detail=f"请求过于频繁，请稍后重试（每分钟最多 {max_req} 次）",
            )

        logger.debug("Rate limit: %s %s (%d/%d)", identifier, path, count, max_req)

    except HTTPException:
        raise
    except Exception as e:
        # Redis 不可用时放行，避免影响正常请求
        logger.debug("Rate limit check skipped (Redis unavailable): %s", e)
        return
