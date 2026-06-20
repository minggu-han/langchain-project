"""
FastAPI 应用入口 - LangChain 学习项目

启动方式：
    uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

项目结构：
    app/
    ├── core/              ← 基础设施（数据库、Redis、安全）
    ├── models/             ← ORM 数据模型（SQLAlchemy）
    ├── schemas/            ← Pydantic 数据验证模型
    ├── api/                ← API 路由（认证、对话、文档）
    ├── langchain_utils/    ← LangChain 功能模块
    └── main.py             ← 应用入口（启动、中间件、生命周期）

API 文档：
    启动后访问 http://localhost:8001/docs     (Swagger UI)
    或           http://localhost:8001/redoc   (ReDoc)

    {
  "username": "zhangsan",
  "password": "MyPass123",
  "email": "user@example.com",
  "full_name": "张三"
}

"""
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.core.database import init_db
from app.core.redis import init_redis, close_redis
from app.core.rate_limit import check_rate_limit
from app.langchain_utils.llm_factory import setup_llm_cache
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.skills import router as skills_router
from app.api.agent import router as agent_router

settings = get_settings()

# ═══════════════════════════════════════════════════════════════
# 日志初始化
# ═══════════════════════════════════════════════════════════════
# 注意：setup_logging() 在 lifespan 启动阶段调用，而不是在这里（模块级别）
# 原因：uvicorn 会在导入模块后重新配置 root logger，覆盖模块级别的日志配置
# 所以必须等 uvicorn 初始化完成后，在 lifespan 中重新配置
logger = get_logger(__name__)


# ================================================================
# 应用生命周期（Startup / Shutdown）
# ================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理

    lifespan 是 FastAPI 推荐的方式，替代旧的 on_event("startup") / on_event("shutdown")

    启动阶段：
    1. 初始化数据库连接池 + 创建表
    2. 验证 Redis 连接
    3. 创建 Chroma 数据目录

    关闭阶段：
    1. 关闭数据库连接池
    2. 关闭 Redis 连接池
    """
    # ═══════════════════════════════════════════════════════════
    # 日志初始化（必须在 lifespan 中执行，此时 uvicorn 已完成初始化）
    # ═══════════════════════════════════════════════════════════
    setup_logging(settings)

    logger.info("=" * 60)
    logger.info("  %s v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info("=" * 60)

    # ── 启动阶段 ──
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database ready")

    try:
        logger.info("Connecting to Redis...")
        await init_redis()
        logger.info("Redis ready")
        setup_llm_cache()
    except Exception as e:
        logger.warning("Redis connection failed (partial functionality unavailable): %s", e)

    logger.info("Application ready — http://localhost:8001/docs")
    logger.info("=" * 60)

    # yield 之前的代码在启动时执行
    yield
    # yield 之后的代码在关闭时执行

    # ── 关闭阶段 ──
    logger.info("Shutting down...")
    try:
        await close_redis()
    except Exception:
        pass
    logger.info("Application stopped")


# ================================================================
# 创建 FastAPI 应用
# ================================================================
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
# 🦜🔗 LangChain 学习项目

基于 **FastAPI** + **PostgreSQL** + **Redis** 的 LangChain 学习平台。

## 功能模块

### 🔐 认证模块 `/api/v1/auth`
- 用户注册/登录（JWT Token 认证）
- OAuth2 Password Flow
- Token 刷新机制

### 💬 LangChain 对话 `/api/v1/chat`
- **基础对话** —— Prompt Template + LLM Chain
- **角色扮演** —— System Prompt 控制 AI 行为
- **带记忆对话** —— ConversationBufferMemory
- **翻译链** —— 专用 Prompt 模板
- **代码审阅** —— 结构化分析
- **Agent 代理** —— 工具调用和自主决策
- **RAG 检索** —— 基于文档的问答

### 📄 文档管理 `/api/v1/documents`
- 文本/文件/URL 索引
- Milvus 向量数据库管理
- RAG 文档查询

### 🎯 Skills 技能 `/api/v1/skills`
- 8 个预置可复用技能模板
- 代码审阅、翻译、摘要、SQL 生成、邮件撰写...
- 支持同步调用和 SSE 流式输出

### 🤖 Full Agent 全功能代理 `/api/v1/agent`
- 整合 Memory + Skills + Tools + RAG + Chain
- Agent 自主决策：用什么能力、什么顺序
- 支持同步和 SSE 流式，展示思考步骤

## 学习路径

1. 注册账户 → 获取 Token
2. 体验基础对话 → 理解 Chain
3. 体验带记忆对话 → 理解 Memory
4. 上传文档 → 体验 RAG
5. 尝试 Agent → 理解工具调用

## 技术栈

| 组件 | 技术 |
|------|------|
| Web 框架 | FastAPI (异步) |
| ORM | SQLAlchemy 2.0 (异步) |
| 数据库 | PostgreSQL + asyncpg |
| 缓存 | Redis |
| 认证 | JWT + bcrypt |
| LLM | LangChain + OpenAI |
| 向量数据库 | Milvus (Docker) |
""",
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI 地址
    redoc_url="/redoc",     # ReDoc 地址
    openapi_url="/openapi.json",
)


# ================================================================
# CORS 中间件（跨域资源共享）
# ================================================================
# 允许前端从不同域名访问 API（开发时有用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # 开发环境允许所有来源
    allow_credentials=True,         # 允许携带 Cookie
    allow_methods=["*"],            # 允许所有 HTTP 方法
    allow_headers=["*"],            # 允许所有请求头
)


# ================================================================
# 速率限制中间件
# ================================================================
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Redis 滑动窗口限流 — 检查请求频率"""
    await check_rate_limit(request)
    return await call_next(request)


# ================================================================
# HTTP 请求/响应日志中间件
# ================================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每个 HTTP 请求的方法、路径、状态码和耗时"""
    start_time = time.time()
    logger.info("--> %s %s", request.method, request.url.path)
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        "<-- %s %s  %d  %.0fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ================================================================
# 注册路由
# ================================================================
app.include_router(auth_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(skills_router, prefix="/api/v1")
app.include_router(agent_router, prefix="/api/v1")


# ================================================================
# 根路径 - 健康检查
# ================================================================
@app.get("/", tags=["系统"], summary="健康检查")
async def root():
    """
    健康检查接口

    返回应用基本信息，可用于：
    - 确认应用正在运行
    - 负载均衡器健康检查
    - 监控系统探测
    """
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


# ================================================================
# 直接在命令行运行时的入口
# ================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
