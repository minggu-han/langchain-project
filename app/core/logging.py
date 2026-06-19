"""
日志系统初始化模块 — 集中配置所有日志输出

设计原则：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 单一 Handler — 只添加一个 StreamHandler 到 root logger
   所有应用和库的 logger 都通过 propagate 传到 root，一处控制
2. 全局级别控制 — LOG_LEVEL 配置项同时控制应用日志和框架日志
3. LangChain 内部日志 — 启用 langchain/langchain_core 等内部 logger
4. 第三方库降噪 — httpx/httpcore/urllib3 等设为 WARNING，避免刷屏
5. 零外部依赖 — 纯 Python logging 模块，不需要 loguru 等第三方库
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

日志格式说明：
  2026-06-19 14:32:05.123  INFO     app.api.chat           消息内容...
  ^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^  ^^^^^^^^^^^^^^^        ^^^^^^^^
  时间（含毫秒）           级别       logger 名称             消息
"""
import logging
import os
import sys
from typing import Any


def setup_logging(settings: Any) -> None:
    """
    初始化日志系统（在应用启动时调用一次）

    调用时机：在 app/main.py 的最开始（在任何 logger 使用之前）

    做了什么：
    1. 将 LOG_LEVEL 配置映射为 Python logging 级别
    2. 清空 root logger 已有 handler，添加一个格式化的 StreamHandler
    3. 启用 LangChain 框架内部 logger（langchain、langchain_core 等）
    4. 压制第三方库的噪音日志
    5. DEBUG 模式下设置 LANGCHAIN_VERBOSE 环境变量
    """
    # ── 1. 解析日志级别 ──
    level_name = settings.LOG_LEVEL.upper() if hasattr(settings, 'LOG_LEVEL') else "INFO"
    log_level = getattr(logging, level_name, logging.INFO)

    # ── 2. 日志格式 ──
    log_format = getattr(
        settings,
        'LOG_FORMAT',
        "%(asctime)s.%(msecs)03d  %(levelname)-8s  %(name)-40s  %(message)s",
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # ── 3. 创建 Handler ──
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # ── 4. 配置 Root Logger ──
    root_logger = logging.getLogger()
    # 清空已有 handler（避免 uvicorn 预装的 handler 造成重复输出）
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)

    # ── 5. 启用 LangChain 框架内部日志 ──
    # LangChain 内部使用 Python logging，启用后可看到：
    # - LLM 请求/响应详情
    # - Chain 执行步骤
    # - Tool 调用过程
    # - Retriever 检索细节
    langchain_loggers = [
        "langchain",
        "langchain_core",
        "langchain_openai",
        "langchain_milvus",
        "langchain_community",
        "langchain_classic",
        "langchain.callbacks",
        "langchain.agents",
        "langchain.chains",
        "langchain.retrievers",
        "langchain.tools",
        "langchain.embeddings",
    ]

    # DEBUG 模式：LangChain 日志也开 DEBUG（非常详细）
    # 非 DEBUG 模式：INFO（只看关键步骤）
    lc_level = logging.DEBUG if (hasattr(settings, 'DEBUG') and settings.DEBUG) else logging.INFO

    for name in langchain_loggers:
        lc_logger = logging.getLogger(name)
        lc_logger.setLevel(lc_level)
        lc_logger.propagate = True  # 传到 root logger，由统一 handler 输出

    # ── 6. 设置 LANGCHAIN_VERBOSE 环境变量 ──
    # 这个变量控制 LangChain 的 stdout verbose 回调（verbose=True 时的输出）
    # 我们通过 logging 模块来控制，不需要这个，但设置它可以让某些
    # LangChain 内部模块也通过 logging 输出更多信息
    if hasattr(settings, 'DEBUG') and settings.DEBUG:
        os.environ["LANGCHAIN_VERBOSE"] = "true"
    else:
        os.environ.pop("LANGCHAIN_VERBOSE", None)

    # ── 7. 压制第三方库噪音 ──
    # 这些库在 DEBUG 模式下会产生大量底层网络日志，影响阅读
    noisy_loggers = [
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "pymilvus",
        "asyncio",
        "sqlalchemy",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    # ── 8. SQLAlchemy 日志（始终 WARNING，太吵） ──
    # 如需查看 SQL 查询，手动设为 INFO
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的 logger（便捷函数）

    用法：
        from app.core.logging import get_logger
        logger = get_logger(__name__)

    注意：
    - 必须在 setup_logging() 调用之后才能使用
    - 不需要给 logger 添加 handler，所有日志自动传播到 root logger
    - 使用 logger.info("message %s", value) 而不是 f-string，实现延迟求值
    """
    return logging.getLogger(name)
