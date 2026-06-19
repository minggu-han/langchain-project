"""
LLM 工厂模块 - 创建和管理 LLM 实例

为什么要用工厂模式？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 集中管理 —— 所有 LLM 配置在一处，修改方便
2. 延迟加载 —— 只在需要时创建实例（节省资源）
3. 缓存复用 —— 同一个实例多次使用（避免重复初始化）
4. 切换方便 —— 从 OpenAI 切换到其他模型只需改一处
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LangChain 支持的模型提供商：
- OpenAI (GPT-4o, GPT-4o-mini, GPT-3.5)
- Anthropic (Claude Opus, Sonnet, Haiku)
- Google (Gemini)
- 本地模型 (Ollama, llama.cpp)
"""
from functools import lru_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.globals import set_llm_cache
from langchain_community.cache import RedisCache
from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def setup_llm_cache() -> None:
    """
    启用 Redis 作为 LangChain LLM 响应缓存

    效果：
    - 相同 prompt 的 LLM 调用直接从 Redis 返回缓存结果
    - 节省 API 费用（OpenAI 每 1000 token 收费）
    - Redis 缓存 TTL 默认为 None（永久），适合学习项目

    调用时机：在 app/main.py 的 lifespan 启动阶段调用
    """
    try:
        settings = get_settings()
        import redis
        redis_client = redis.Redis.from_url(settings.REDIS_URL)
        # 测试连接
        redis_client.ping()
        set_llm_cache(RedisCache(redis_client))
        logger.info("LLM cache enabled (Redis backend)")
    except Exception as e:
        logger.warning("LLM cache disabled (Redis unavailable): %s", e)


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.7, model_name: str | None = None):
    """
    获取 ChatOpenAI LLM 实例（单例缓存）

    参数说明：
    ┌─────────────┬────────────────────────────────────────────────┐
    │ 参数         │ 说明                                           │
    ├─────────────┼────────────────────────────────────────────────┤
    │ temperature │ 温度（0-2），控制输出随机性：                    │
    │             │  - 0.0 = 确定性的（代码生成、数学）               │
    │             │  - 0.7 = 平衡的（通用对话）                      │
    │             │  - 1.0+ = 创造性的（写作、头脑风暴）              │
    │ model_name  │ 模型名称，默认从配置读取                         │
    │ max_tokens  │ 最大输出 Token 数，None = 模型自动决定            │
    │ streaming   │ 是否启用流式输出（打字机效果）                    │
    └─────────────┴────────────────────────────────────────────────┘

    ChatOpenAI vs OpenAI 的区别：
    - ChatOpenAI：用于对话模型（gpt-4o, gpt-3.5-turbo）
      - 输入/输出都是 Message 对象列表
    - OpenAI：用于补全模型（gpt-3.5-turbo-instruct）
      - 输入/输出都是纯文本字符串
    - 现在推荐使用 ChatOpenAI，因为 OpenAI 已转向 Chat API
    """
    settings = get_settings()

    model = model_name or settings.OPENAI_MODEL
    logger.debug("Creating LLM: model=%s, temperature=%.2f", model, temperature)

    # 构建参数
    kwargs = {
        "temperature": temperature,
        "model": model,
        "api_key": settings.OPENAI_API_KEY,
        "streaming": True,  # 启用流式输出，支持 astream()
        # verbose 由全局 LANGCHAIN_VERBOSE + logging 控制，不在此硬编码
    }

    # 如果配置了自定义 base_url（如 API 代理），则使用它
    if settings.OPENAI_BASE_URL:
        kwargs["base_url"] = settings.OPENAI_BASE_URL
        logger.debug("Using custom base_url: %s", settings.OPENAI_BASE_URL)

    return ChatOpenAI(**kwargs)


@lru_cache(maxsize=1)
def get_embeddings():
    """
    获取 OpenAI Embeddings 实例（单例缓存）

    Embeddings 是什么？
    - 将文本转换为高维向量（一组数字）
    - 语义相似的文本，向量在空间中距离更近
    - 例如：
      - "我爱吃苹果" → [0.01, 0.23, -0.15, ..., 0.08]（1536 维向量）
      - "我喜欢吃水果" → [0.02, 0.25, -0.13, ..., 0.07]（向量距离很近）
      - "今天天气不错" → [-0.12, 0.31, 0.18, ..., -0.05]（向量距离很远）

    Embeddings 在 RAG 中的作用：
    1. 将文档切块后，每块生成一个向量
    2. 将用户问题也生成一个向量
    3. 通过向量相似度找到最相关的文档块
    4. 将相关文档块作为上下文传给 LLM 生成答案
    """
    settings = get_settings()
    logger.debug("Creating Embeddings: base_url=%s", settings.OPENAI_BASE_URL or "default")
    kwargs = {
        "api_key": settings.OPENAI_API_KEY,
    }
    if settings.OPENAI_BASE_URL:
        kwargs["base_url"] = settings.OPENAI_BASE_URL

    return OpenAIEmbeddings(**kwargs)


def clear_llm_cache():
    """清除 LLM 缓存（配置更新后调用）"""
    get_llm.cache_clear()
    get_embeddings.cache_clear()
