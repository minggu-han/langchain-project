"""
安全模块 - JWT Token 生成/验证 + 密码哈希

核心概念：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. JWT（JSON Web Token）  —— 无状态认证令牌，由三部分组成：
   - Header（头部）  ：算法 + Token 类型
   - Payload（载荷） ：用户信息 + 过期时间
   - Signature（签名）：防篡改的哈希值

2. OAuth2 Password Flow —— FastAPI 内置的认证流程：
   用户发送用户名+密码 → 服务器验证 → 返回 JWT Token
   后续请求在 Authorization 头中携带 Token

3. bcrypt —— 密码哈希算法：
   - 单向哈希，无法逆向破解
   - 自动加盐（salt），防止彩虹表攻击
   - 计算成本可调（rounds 参数）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import hashlib
from datetime import datetime, timedelta
from typing import Any

import bcrypt
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis

from app.config import get_settings
from app.core.database import get_db
from app.core.redis import get_redis
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ================================================================
# 1. bcrypt 密码哈希
# ================================================================
# 为什么直接用 bcrypt 而不是 passlib？
# - passlib 是旧库（2020 年后未更新），与新版 bcrypt（5.0+）不兼容
# - bcrypt 是官方维护的库，简单直接
# - 直接用 bcrypt 避免中间层的不兼容问题

def hash_password(password: str) -> str:
    """
    对明文密码进行 bcrypt 哈希

    为什么不用 MD5/SHA？
    - MD5/SHA 是快速哈希，GPU 可以每秒尝试数十亿次
    - bcrypt 是慢速哈希，每次哈希故意耗时 ~0.3 秒
    - 对于正常登录，0.3 秒可以忽略
    - 对于暴力破解，0.3 秒意味着每秒只能试 3 次

    Args:
        password: 明文密码
    Returns:
        哈希后的密码字符串（包含算法标识和盐值），如 $2b$12$LJ3m4ys3GZ...

    注意：bcrypt 要求密码先编码为 UTF-8 字节
    """
    # bcrypt.hashpw 需要 bytes 输入
    password_bytes = password.encode("utf-8")
    # 生成盐值（rounds=12 表示 2^12 次迭代，约 0.3 秒）
    salt = bcrypt.gensalt(rounds=12)
    # 哈希密码
    hashed = bcrypt.hashpw(password_bytes, salt)
    # 返回字符串形式（用于存入数据库）
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证明文密码是否匹配哈希

    Args:
        plain_password: 用户输入的明文密码
        hashed_password: 数据库中存储的哈希密码
    Returns:
        是否匹配

    工作原理：
    1. 从 hashed_password 中提取盐值（存在哈希字符串前缀中）
    2. 用相同的盐值对 plain_password 进行哈希
    3. 比较两个哈希值
    """
    password_bytes = plain_password.encode("utf-8")
    hashed_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(password_bytes, hashed_bytes)


# ================================================================
# 2. JWT Token 生成与验证
# ================================================================
# ✅ HTTPBearer：简单的 Bearer Token 输入框（推荐！直接粘贴 Token 即可）
# ─────────────────────────────────────────────────────────────
# 在 Swagger UI 中点击 Authorize → 会看到一个输入框，直接粘贴 access_token
http_bearer_scheme = HTTPBearer(
    scheme_name="JWT Token",  # Swagger UI 中显示的名称
    description="输入登录后获取的 access_token（不需要 'Bearer ' 前缀）",
)

# OAuth2PasswordBearer：完整的 OAuth2 密码流程（有4个输入框）
# 保留它仅用于 /login/oauth 接口的 Swagger 文档引用
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login/oauth",
)


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """
    创建 JWT 访问令牌（Access Token）

    Access Token 特点：
    - 短期有效（默认 30 分钟）
    - 每次 API 请求都需要携带
    - 过期后使用 Refresh Token 刷新

    Args:
        data: 要编码到 Token 中的数据（通常包含 user_id, username）
        expires_delta: 自定义过期时间，None 则使用默认值
    Returns:
        编码后的 JWT 字符串

    JWT 内部结构：
        eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIiwiZXhwIjoxNzAwMDAwMDAwfQ.signature
        ^                    ^                                            ^
        Header               Payload                                      Signature
        (算法信息)           (用户数据+过期时间)                           (防篡改签名)
    """
    to_encode = data.copy()
    # 计算过期时间
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    # 添加 JWT 标准字段
    to_encode.update({
        "exp": expire,      # 过期时间（Expiration Time）
        "iat": datetime.utcnow(),  # 签发时间（Issued At）
        "type": "access",   # Token 类型标识
    })

    # 使用 SECRET_KEY 签名
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt


def create_refresh_token(data: dict[str, Any]) -> str:
    """
    创建 JWT 刷新令牌（Refresh Token）

    Refresh Token 特点：
    - 长期有效（默认 7 天）
    - 只用于获取新的 Access Token
    - 不用于 API 请求认证

    为什么需要两个 Token？
    - 如果 Access Token 被盗，攻击者只有 30 分钟窗口期
    - Refresh Token 使用频率低，暴露风险小
    - 可以通过撤销 Refresh Token 来强制用户重新登录
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
    })
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    """
    解码并验证 JWT Token

    Args:
        token: JWT 字符串
    Returns:
        Token 中包含的数据字典

    Raises:
        JWTError: Token 无效或已过期

    验证步骤：
    1. 检查签名是否有效（使用相同的 SECRET_KEY 重新签名并比较）
    2. 检查是否过期（exp 字段）
    3. 解码 Payload 并返回
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except JWTError as e:
        # Token 无效：可能过期、签名不对、格式错误
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"无效的认证凭证: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ================================================================
# 3. Token 黑名单（Redis）
# ================================================================

def _token_key(token: str) -> str:
    """生成 token 对应的 Redis key（SHA256 哈希，不存原始 token）"""
    return f"blacklist:token:{hashlib.sha256(token.encode()).hexdigest()}"


async def add_to_blacklist(token: str, redis: aioredis.Redis) -> None:
    """将 token 加入 Redis 黑名单，TTL 匹配 token 剩余有效期"""
    try:
        payload = decode_token(token)
        exp = payload.get("exp")
        now = datetime.utcnow().timestamp()
        ttl = int(exp - now) if exp else 1800  # 默认 30 分钟
        if ttl > 0:
            await redis.setex(_token_key(token), ttl, "1")
            logger.info("Token blacklisted: ttl=%ds", ttl)
    except Exception:
        # token 已过期或无效，无需加入黑名单
        pass


async def is_blacklisted(token: str, redis: aioredis.Redis) -> bool:
    """检查 token 是否在黑名单中"""
    try:
        return await redis.exists(_token_key(token)) > 0
    except Exception:
        return False  # Redis 不可用时放行


# ================================================================
# 4. 获取当前用户（FastAPI 依赖）
# ================================================================
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer_scheme),
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
):
    """
    从 JWT Token 中解析当前登录用户

    这是最常用的认证依赖，在需要登录的路由中使用：

        @app.get("/me")
        async def my_profile(current_user = Depends(get_current_user)):
            return {"username": current_user.username}

    工作流程：
    1. 从请求头 Authorization: Bearer <token> 中提取 Token
    2. 解码并验证 Token
    3. 从数据库中查询用户
    4. 如果用户不存在或 Token 无效，返回 401

    注意：
    - 使用 Depends 实现依赖注入，FastAPI 自动处理缓存和生命周期
    - 可以在任何路由中复用这个依赖
    """
    from sqlalchemy import select
    from app.models.user import User

    # HTTPBearer 自动从 Authorization 头提取 Token
    # credentials.credentials 就是 Bearer 后面的字符串
    token = credentials.credentials
    logger.debug("Authenticating token (prefix: %s...)", token[:10])

    # 解码 Token
    payload = decode_token(token)

    # 检查 Token 类型（防止用 Refresh Token 访问 API）
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请使用访问令牌（Access Token），而不是刷新令牌",
        )

    # 检查 Redis 黑名单（登出后的 Token 不可再用）
    if await is_blacklisted(token, redis):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 已失效（已登出），请重新登录",
        )

    # 从 Token 中获取用户名
    username: str | None = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 中缺少用户标识",
        )

    # 从数据库查询用户
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在或已被删除",
        )

    # 检查用户是否被禁用
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账户已被禁用，请联系管理员",
        )

    logger.debug("User authenticated: username='%s', id=%d", user.username, user.id)
    return user
