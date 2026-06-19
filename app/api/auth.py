"""
认证 API 路由 - 用户注册、登录、Token 管理

OAuth2 Password Flow（密码模式）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 用户提交用户名+密码 → 服务器验证 → 返回 Access Token + Refresh Token
2. 后续请求携带 Access Token（Authorization: Bearer <token>）
3. Access Token 过期后，用 Refresh Token 获取新的 Access Token

为什么需要两种 Token？
- Access Token：短期（30分钟），用于 API 认证
  → 即使被盗，攻击窗口只有 30 分钟
- Refresh Token：长期（7天），只用于刷新 Access Token
  → 使用频率低，暴露风险小
"""
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import redis.asyncio as aioredis

from app.config import get_settings
from app.core.database import get_db
from app.core.redis import get_redis
from app.core.logging import get_logger
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    add_to_blacklist,
    http_bearer_scheme,
)
from app.models.user import User
from app.schemas.user import (
    UserCreate,
    UserLogin,
    UserResponse,
    TokenResponse,
    TokenRefresh,
)

settings = get_settings()
logger = get_logger(__name__)

# 创建路由器
# prefix="/auth" 所有路由都是 /api/v1/auth/...
# tags=["认证"] 在 Swagger UI 中分组显示
router = APIRouter(prefix="/auth", tags=["认证"])


# ================================================================
# 1. 用户注册
# ================================================================
@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="用户注册",
    description="创建新用户账户。用户名和邮箱必须唯一。密码需要至少8个字符，包含字母和数字。",
)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    注册新用户

    流程：
    1. 检查用户名是否已存在
    2. 检查邮箱是否已存在
    3. 对密码进行 bcrypt 哈希
    4. 将用户信息存入数据库

    注意：
    - 密码永远不会以明文形式存储
    - bcrypt 哈希是单向的，无法逆向还原
    - 即使数据库泄露，攻击者也无法获知原始密码
    """
    logger.info("POST /auth/register - username='%s'", user_data.username)

    # 检查用户名是否已存在
    result = await db.execute(
        select(User).where(User.username == user_data.username)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"用户名 '{user_data.username}' 已被占用",
        )

    # 检查邮箱是否已存在
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"邮箱 '{user_data.email}' 已被注册",
        )

    # 创建用户对象
    # 注意：密码在存入数据库之前必须先哈希！
    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name,
    )

    db.add(user)
    await db.flush()   # 刷新到数据库以获取生成的 ID
    await db.refresh(user)  # 刷新对象状态

    return user


# ================================================================
# 2. 用户登录（JSON 格式）
# ================================================================
@router.post(
    "/login",
    response_model=TokenResponse,
    summary="用户登录（JSON格式）",
    description="使用用户名和密码登录，返回 Access Token 和 Refresh Token。",
)
async def login_json(
    login_data: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    """
    JSON 格式的登录接口

    流程：
    1. 根据用户名查找用户
    2. 验证明文密码是否匹配数据库中的哈希
    3. 生成 Access Token + Refresh Token
    4. 返回 Token 对
    """
    logger.info("POST /auth/login - username='%s'", login_data.username)
    return await _authenticate_user(db, login_data.username, login_data.password)


# ================================================================
# 3. 用户登录（OAuth2 表单格式）
# ================================================================
@router.post(
    "/login/oauth",
    response_model=TokenResponse,
    summary="用户登录（OAuth2表单格式）",
    description="""标准的 OAuth2 Password Flow 登录接口。

Swagger UI 中的 "Authorize" 按钮使用的就是这个接口。
使用 `application/x-www-form-urlencoded` 格式提交数据。
""",
)
async def login_oauth(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """
    OAuth2 标准格式的登录接口

    OAuth2PasswordRequestForm 要求：
    - Content-Type: application/x-www-form-urlencoded
    - 字段：username, password, grant_type, scope

    这个接口是 FastAPI 自动生成的 Swagger UI 认证弹窗使用的。
    """
    return await _authenticate_user(db, form_data.username, form_data.password)


# ================================================================
# 4. 刷新 Token
# ================================================================
@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="刷新 Access Token",
    description="使用 Refresh Token 获取新的 Access Token。Access Token 过期后无需重新登录。",
)
async def refresh_token(
    refresh_data: TokenRefresh,
):
    """
    用 Refresh Token 获取新的 Access Token

    使用场景：
    1. Access Token 即将过期（30分钟后）
    2. 前端自动调用此接口
    3. 获取新的 Access Token，用户无需重新登录

    安全考虑：
    - Refresh Token 的有效期是 7 天
    - 7 天后需要重新登录
    - 不要将 Refresh Token 存储在 localStorage（XSS 风险）
    - 推荐使用 httpOnly cookie 存储
    """
    # 解码并验证 Refresh Token
    payload = decode_token(refresh_data.refresh_token)

    # 确保这是 Refresh Token 而不是 Access Token
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请使用刷新令牌（Refresh Token），而不是访问令牌",
        )

    # 从 Token 中提取用户信息
    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 格式无效",
        )

    # 生成新的 Token 对
    access_token = create_access_token(data={"sub": username})
    new_refresh_token = create_refresh_token(data={"sub": username})

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # 转换为秒
    )


# ================================================================
# 5. 登出（Token 黑名单）
# ================================================================
@router.post(
    "/logout",
    summary="登出（使当前 Access Token 失效）",
    description="将当前 Access Token 加入 Redis 黑名单，使之立即失效。",
)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer_scheme),
    redis: aioredis.Redis = Depends(get_redis),
):
    """
    登出操作

    原理：
    - JWT 是无状态的，无法主动撤销
    - 将 token 加入 Redis 黑名单，设置 TTL 匹配剩余有效时间
    - 后续请求在黑名单中检查，命中则拒绝
    """
    token = credentials.credentials
    await add_to_blacklist(token, redis)
    return {"message": "已登出，Token 已失效", "status": "success"}


@router.post(
    "/logout-all",
    summary="登出所有设备（使 Refresh Token 也失效）",
    description="将当前 Access Token 加入黑名单后，Refresh Token 也将失效。重新登录获取新 Token。",
)
async def logout_all(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer_scheme),
    redis: aioredis.Redis = Depends(get_redis),
):
    """
    登出所有设备

    额外将 refresh token 也加入黑名单，彻底阻断 token 刷新。
    """
    token = credentials.credentials
    await add_to_blacklist(token, redis)
    logger.info("Logout all: access token blacklisted")
    return {"message": "已从所有设备登出，请重新登录", "status": "success"}


# ================================================================
# 6. 获取当前用户信息
# ================================================================
@router.get(
    "/me",
    response_model=UserResponse,
    summary="获取当前用户信息",
    description="返回当前登录用户的详细信息。需要在请求头中携带有效的 Access Token。",
)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    获取当前登录用户信息

    使用方式：
    1. 在 Swagger UI 中点击 "Authorize" 按钮
    2. 输入 Access Token
    3. 调用此接口即可获取用户信息

    这是验证认证系统是否正常工作的最佳测试接口。
    """
    return current_user


# ================================================================
# 辅助函数：用户认证逻辑
# ================================================================
async def _authenticate_user(
    db: AsyncSession,
    username: str,
    password: str,
) -> TokenResponse:
    """
    验证用户凭证并生成 Token

    这个函数被 JSON 登录和 OAuth2 登录两个接口共用，
    体现了 DRY（Don't Repeat Yourself）原则。
    """
    # 1. 查找用户
    result = await db.execute(
        select(User).where(User.username == username)
    )
    user = result.scalar_one_or_none()

    # 2. 验证用户存在且密码正确
    # 注意：错误消息故意模糊，不透露是用户名还是密码错了
    # 这是安全最佳实践——防止用户枚举攻击
    if not user or not verify_password(password, user.hashed_password):
        logger.warning("Auth failed: username='%s' (invalid credentials)", username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. 检查账户状态
    if not user.is_active:
        logger.warning("Auth failed: username='%s' (account disabled)", username)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账户已被禁用",
        )

    logger.info("Auth success: username='%s', user_id=%d", user.username, user.id)

    # 4. 生成 Token
    token_data = {
        "sub": user.username,   # sub = Subject（JWT 标准字段）
        "user_id": user.id,
    }

    access_token = create_access_token(data=token_data)
    refresh_token = create_refresh_token(data=token_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
