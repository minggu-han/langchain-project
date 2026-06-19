"""
用户相关 Pydantic Schema - API 请求/响应的数据结构

Pydantic 的核心功能：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 数据验证      —— 自动检查类型、长度、格式
2. 数据转换      —— 字符串自动转为 int、datetime 等
3. JSON 序列化   —— model_dump() / model_dump_json()
4. OpenAPI 文档  —— 自动生成 API 文档的请求/响应示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import re
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


# ================================================================
# 1. 用户注册请求
# ================================================================
class UserCreate(BaseModel):
    """
    注册新用户的请求体

    Field() 参数说明：
    - min_length/max_length：字符串长度限制
    - examples：OpenAPI 文档中的示例值
    - description：字段说明（会显示在文档中）
    """
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_]+$",  # 只允许字母、数字、下划线
        description="用户名，3-50 字符，只能包含字母、数字和下划线",
        examples=["zhangsan"],
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="密码，最少 8 个字符，必须包含字母和数字",
        examples=["MyPass123"],
    )
    email: EmailStr = Field(
        ...,
        description="邮箱地址",
        examples=["user@example.com"],
    )
    full_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="用户全名（可选）",
        examples=["张三"],
    )

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """
        验证密码强度

        @field_validator 是 Pydantic v2 的自定义验证器
        命名规则：validate_<字段名>
        每个带这个装饰器的方法都是一个验证步骤
        """
        if not re.search(r"[A-Za-z]", v):
            raise ValueError("密码必须包含至少一个字母")
        if not re.search(r"\d", v):
            raise ValueError("密码必须包含至少一个数字")
        return v


# ================================================================
# 2. 用户登录请求
# ================================================================
class UserLogin(BaseModel):
    """
    登录请求体

    OAuth2PasswordRequestForm 要求 username 和 password 字段
    这个 Schema 也兼容 JSON 格式的登录请求
    """
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


# ================================================================
# 3. 用户信息响应
# ================================================================
class UserResponse(BaseModel):
    """
    返回给客户端的用户信息

    注意：这个响应不包含 hashed_password！
    - 密码哈希绝不能通过 API 返回
    - 即使哈希了也不能返回（可能在别处被暴力破解）
    """
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime

    # model_config 配置 Pydantic 行为
    model_config = {
        "from_attributes": True,  # 允许从 ORM 对象创建（替代 v1 的 orm_mode）
    }


# ================================================================
# 4. 用户更新请求
# ================================================================
class UserUpdate(BaseModel):
    """更新用户信息（所有字段可选）"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(default=None, max_length=100)
    password: Optional[str] = Field(default=None, min_length=8, max_length=128)


# ================================================================
# 5. Token 响应
# ================================================================
class TokenResponse(BaseModel):
    """
    登录成功返回的 Token 响应

    包含：
    - access_token：用于 API 认证（短期）
    - refresh_token：用于刷新 access_token（长期）
    - token_type：认证类型，固定为 "bearer"
    - expires_in：access_token 的有效期（秒）
    """
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token 有效期（秒）")


class TokenRefresh(BaseModel):
    """刷新 Token 的请求体"""
    refresh_token: str = Field(..., description="Refresh Token 字符串")
