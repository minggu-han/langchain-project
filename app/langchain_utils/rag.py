"""
LangChain 实战篇 —— RAG（检索增强生成）

什么是 RAG（Retrieval-Augmented Generation）？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAG 是让 LLM 基于私有文档回答问题的技术。

为什么需要 RAG？
1. LLM 训练数据有截止日期（知识过时）
2. LLM 不知道你的私有文档内容
3. LLM 会 "幻觉"（编造不存在的事实）
4. RAG 让 LLM 的回答可追溯（能引用来源）

RAG 的完整工作流程：
┌──────────────────────────────────────────────────────────┐
│               索引阶段（离线，一次性）                     │
│                                                          │
│  文档 → 加载 → 切分 → 嵌入 → 向量数据库（Milvus）         │
│                                                          │
│  PDF/TXT   Document  Chunks   Vectors  VectorStore       │
│                 ↓                                         │
│               检索阶段（在线，每次查询）                    │
│                                                          │
│  用户问题 → 嵌入 → 向量检索 → 相关文档 → LLM → 答案       │
└──────────────────────────────────────────────────────────┘

Milvus vs Chroma：
┌──────────┬──────────────────────┬──────────────────────┐
│ 特性      │ Milvus               │ Chroma               │
├──────────┼──────────────────────┼──────────────────────┤
│ 架构      │ 分布式（支持集群）    │ 单机嵌入式            │
│ 性能      │ 十亿级向量毫秒检索    │ 百万级向量            │
│ 部署      │ Docker / K8s         │ pip install 即可      │
│ 索引类型  │ IVF_FLAT, HNSW 等    │ HNSW                  │
│ 适用场景  │ 生产环境              │ 开发/原型验证         │
└──────────┴──────────────────────┴──────────────────────┘

核心组件说明：
┌──────────────────┬─────────────────────────────────────────┐
│ DocumentLoader   │ 加载各种格式的文档（PDF, TXT, Web...）  │
│ TextSplitter     │ 将长文档切分成小块（chunk）              │
│ Embeddings       │ 将文本转换为向量（数字数组）            │
│ VectorStore      │ 存储和检索向量（Milvus）                │
│ Retriever        │ 检索器接口（从 VectorStore 检索文档）    │
└──────────────────┴─────────────────────────────────────────┘
"""
import uuid
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_milvus import Milvus
from pymilvus import connections, utility

from app.langchain_utils.llm_factory import get_llm, get_embeddings
from app.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ================================================================
# 1. Milvus 连接管理
# ================================================================
# Milvus 连接（懒加载，首次使用时建立连接）
_connection_established = False


def _ensure_connection():
    """
    确保与 Milvus 的连接已建立

    Milvus 连接的特点：
    - 连接是全局的（进程级别），不需要每次操作都连接
    - 支持连接池
    - alias="default" 是默认连接别名
    """
    global _connection_established
    if not _connection_established:
        logger.debug("Connecting to Milvus at %s:%s...", settings.MILVUS_HOST, settings.MILVUS_PORT)
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
        )
        _connection_established = True
        logger.info("Milvus connected to %s:%s", settings.MILVUS_HOST, settings.MILVUS_PORT)


# ================================================================
# 2. 获取或创建 Milvus 向量存储
# ================================================================
def get_vector_store(collection_name: str = "default") -> Milvus:
    """
    获取或创建 Milvus 向量存储

    Milvus 的核心概念：
    ┌──────────────┬──────────────────────────────────────┐
    │ 概念          │ 说明                                 │
    ├──────────────┼──────────────────────────────────────┤
    │ Collection   │ 集合 = 一张表，存储一类文档的向量     │
    │              │ 类比：关系数据库中的 Table            │
    │ Partition    │ 分区 = 集合内的逻辑分区（可选）       │
    │ Field        │ 字段 = 集合中的列（id, vector, text） │
    │ Index        │ 索引 = 加速向量检索的结构            │
    │              │ 类型：IVF_FLAT, HNSW, IVF_SQ8 等     │
    └──────────────┴──────────────────────────────────────┘

    在这个学习项目中：
    - 每个 collection_name 对应一个 Milvus Collection
    - 例如 "knowledge_base"、"product_docs"、"faqs"

    Args:
        collection_name: 集合名称

    Returns:
        LangChain Milvus 向量存储实例
    """
    _ensure_connection()
    embeddings = get_embeddings()

    logger.debug("Getting vector store: collection='%s'", collection_name)

    # LangChain 的 Milvus 封装会自动处理：
    # 1. 如果 Collection 不存在 → 创建
    # 2. 如果 Collection 已存在 → 直接使用
    # 3. 自动创建索引（默认 IVF_FLAT）
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={
            "host": settings.MILVUS_HOST,
            "port": settings.MILVUS_PORT,
        },
        # 向量维度由 Embedding 模型决定（OpenAI text-embedding-ada-002 = 1536）
        # drop_old=False 表示不删除已有的同名 Collection
        drop_old=False,
        # 自动创建主键字段
        auto_id=True,
    )
    return vector_store


# ================================================================
# 3. 文档加载器
# ================================================================
async def load_documents_from_text(text: str, source_name: str = "用户输入") -> List[Document]:
    """
    从纯文本创建文档对象

    Document 是 LangChain 的核心数据结构：
    - page_content：文档的文本内容
    - metadata：文档的元数据（来源、页码、作者等）
    """
    return [Document(
        page_content=text,
        metadata={
            "source": source_name,
            "doc_id": str(uuid.uuid4()),
        }
    )]


async def load_documents_from_file(file_path: str) -> List[Document]:
    """
    从文件加载文档

    支持格式：
    - .txt：纯文本文件
    - .pdf：PDF 文件
    """
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}（仅支持 .txt 和 .pdf）")

    return loader.load()


async def load_documents_from_url(url: str) -> List[Document]:
    """
    从网页加载文档

    使用 WebBaseLoader 抓取网页内容
    """
    loader = WebBaseLoader(url)
    return loader.load()


# ================================================================
# 4. 文本分割器
# ================================================================
def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    将长文档切分成小块（Chunking）

    为什么要切分？
    1. LLM 有上下文长度限制（Token 限制）
    2. 小块检索精度更高（大块包含太多无关信息）
    3. Embedding 模型对短文本效果更好

    为什么要重叠（chunk_overlap）？
    1. 防止关键信息被切断（正好在 chunk 边界）
    2. 保持语义完整性

    参数选择建议：
    ┌────────────┬──────────┬─────────────────────────────────┐
    │ 文档类型    │ chunk    │ 说明                             │
    ├────────────┼──────────┼─────────────────────────────────┤
    │ 代码        │ 500-800  │ 函数/类通常在这个范围            │
    │ 文档/文章   │ 800-1200 │ 包含 2-3 个段落                  │
    │ 长文/书籍   │ 1500+    │ 需要更多上下文                   │
    │ 问答/FAQ    │ 200-500  │ 每个问答对较短                   │
    └────────────┴──────────┴─────────────────────────────────┘
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    return splitter.split_documents(docs)


# ================================================================
# 5. 文档索引（离线阶段）
# ================================================================
async def index_text(
    text: str,
    collection_name: str = "default",
    source_name: str = "手动输入",
) -> int:
    """
    将文本索引到 Milvus 向量数据库

    完整的索引流程：
    1. 创建 Document 对象
    2. 切分成 chunks
    3. 生成 Embeddings（向量）
    4. 存入 Milvus Collection

    Args:
        text: 要索引的文本
        collection_name: Milvus 集合名称
        source_name: 来源标识

    Returns:
        索引的 chunk 数量
    """
    logger.info("index_text: source='%s', collection='%s', text_len=%d", source_name, collection_name, len(text))

    # 1. 加载
    docs = await load_documents_from_text(text, source_name)

    # 2. 切分
    chunks = split_documents(docs)
    logger.debug("index_text: created %d chunks", len(chunks))

    # 3. 存入 Milvus（Milvus 会自动调用 Embeddings 生成向量）
    vector_store = get_vector_store(collection_name)
    # add_documents 会：生成向量 → 插入 Milvus Collection
    vector_store.add_documents(chunks)

    logger.info("index_text completed: %d chunks indexed to '%s'", len(chunks), collection_name)
    return len(chunks)


async def index_file(
    file_path: str,
    collection_name: str = "default",
) -> int:
    """
    将文件索引到 Milvus 向量数据库
    """
    logger.info("index_file: file='%s', collection='%s'", file_path, collection_name)
    docs = await load_documents_from_file(file_path)
    chunks = split_documents(docs)
    logger.debug("index_file: created %d chunks", len(chunks))
    vector_store = get_vector_store(collection_name)
    vector_store.add_documents(chunks)
    logger.info("index_file completed: %d chunks indexed to '%s'", len(chunks), collection_name)
    return len(chunks)


# ================================================================
# 6. RAG 查询（在线阶段）
# ================================================================
async def rag_query(
    query: str,
    collection_name: str = "default",
    top_k: int = 4,
) -> dict:
    """
    执行 RAG 查询 —— 检索 + 生成

    完整流程：
    1. 将用户问题 Embedding 为向量
    2. 在 Milvus 中检索 top_k 个最相关的文档块
       （Milvus 使用 ANN 近似最近邻算法，毫秒级检索百万向量）
    3. 将检索结果作为上下文，拼接成 Prompt
    4. LLM 基于上下文生成答案

    Args:
        query: 用户的问题
        collection_name: Milvus 集合名称
        top_k: 返回的相关文档块数量（建议 3-5）

    Returns:
        包含答案和引用来源的字典
    """
    logger.info("rag_query: query='%.60s...', collection='%s', top_k=%d", query, collection_name, top_k)

    vector_store = get_vector_store(collection_name)

    # 检查 Collection 是否存在且有数据
    _ensure_connection()
    if not utility.has_collection(collection_name):
        logger.warning("rag_query: collection '%s' not found", collection_name)
        return {
            "answer": f"集合 '{collection_name}' 不存在，请先索引一些文档。可以通过 /api/v1/documents/index 接口上传文档。",
            "sources": [],
        }

    # Milvus Collection 中的实体数量
    try:
        collection = utility.get_collection_stats(collection_name)
        row_count = collection.get("row_count", 0)
        if row_count == 0:
            return {
                "answer": f"集合 '{collection_name}' 为空，请先索引文档。",
                "sources": [],
            }
    except Exception:
        # get_collection_stats 在新版 Milvus 中可能返回不同的结构
        pass

    # 第1步：向量相似度检索
    # similarity_search 内部流程：
    # 1. 将 query 用 Embedding 模型转换为向量
    # 2. 在 Milvus 中执行 ANN（近似最近邻）搜索
    # 3. 返回相似度最高的 top_k 个 Document
    logger.debug("rag_query: executing similarity search (top_k=%d)...", top_k)
    relevant_docs = vector_store.similarity_search(query, k=top_k)

    if not relevant_docs:
        logger.info("rag_query: no relevant documents found")
        return {
            "answer": "未找到与问题相关的文档内容。",
            "sources": [],
        }

    logger.debug("rag_query: retrieved %d documents", len(relevant_docs))

    # 第2步：构建 RAG Prompt
    context = "\n\n---\n\n".join([
        f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
        for doc in relevant_docs
    ])

    # RAG 专用 Prompt：要求 LLM 严格基于上下文回答
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个基于文档的问答助手。请严格根据以下提供的上下文来回答问题。

规则：
1. 如果在上下文中找到答案，请准确回答并引用来源
2. 如果上下文不足以回答问题，请明确说 "根据提供的文档，无法回答这个问题"
3. 不要编造不存在于上下文中的信息
4. 回答要简洁明了，使用 Markdown 格式"""),
        ("human", "上下文资料：\n\n{context}\n\n---\n\n问题：{question}\n\n请根据以上上下文回答问题。"),
    ])

    # 第3步：创建 RAG Chain 并生成答案
    llm = get_llm(temperature=0.3)  # RAG 用低温度，减少幻觉
    chain = prompt | llm | StrOutputParser()

    answer = await chain.ainvoke({
        "context": context,
        "question": query,
    })

    # 收集引用来源
    sources = []
    for doc in relevant_docs:
        source = doc.metadata.get("source", "未知来源")
        if source not in sources:
            sources.append(source)

    logger.info("rag_query completed: answer_len=%d, sources=%d, chunks=%d",
                len(answer), len(sources), len(relevant_docs))

    return {
        "answer": answer,
        "sources": sources,
        "relevant_chunks": len(relevant_docs),
    }


# ================================================================
# 7. 管理功能
# ================================================================
def get_collection_stats(collection_name: str = "default") -> dict:
    """
    获取 Milvus 集合的统计信息

    返回：
    - collection_name: 集合名称
    - entity_count: 文档（向量实体）数量
    - exists: 集合是否存在
    """
    _ensure_connection()
    exists = utility.has_collection(collection_name)

    if not exists:
        return {
            "collection_name": collection_name,
            "entity_count": 0,
            "exists": False,
        }

    try:
        stats = utility.get_collection_stats(collection_name)
        row_count = stats.get("row_count", 0)
    except Exception:
        row_count = "无法获取（请使用 Milvus Attu 管理界面查看）"

    return {
        "collection_name": collection_name,
        "entity_count": row_count,
        "exists": True,
        "milvus_host": f"{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
    }


def list_collections() -> list[str]:
    """列出 Milvus 中所有的 Collection"""
    _ensure_connection()
    return utility.list_collections()


def drop_collection(collection_name: str):
    """
    删除 Milvus 集合

    注意：这会永久删除集合中的所有数据！
    """
    _ensure_connection()
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        logger.info("Milvus collection '%s' dropped", collection_name)


def clear_collection(collection_name: str = "default"):
    """
    清空集合中的所有向量数据

    策略：删除旧集合再重建（因为 Milvus 不支持 TRUNCATE）
    注意：更优雅的方式是调用 collection.delete() 但 LangChain 封装层不暴露 entity 删除 API
          所以这里采用 drop + 重新触发创建的方式
    """
    _ensure_connection()
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        logger.info("Milvus collection '%s' cleared", collection_name)


# ================================================================
# 8. RAG 流式查询
# ================================================================
from typing import AsyncIterator


async def rag_query_stream(
    query: str,
    collection_name: str = "default",
    top_k: int = 4,
) -> AsyncIterator[dict]:
    """
    RAG 流式查询 — 检索后逐 token 生成，最后返回引用来源

    与 rag_query 的区别：
    - rag_query 返回 dict（answer + sources + chunks）
    - rag_query_stream yield dict 事件（token / metadata / done）

    事件类型：
    - {"type": "start"}
    - {"type": "token", "content": "..."}
    - {"type": "metadata", "sources": [...], "relevant_chunks": N}
    - {"type": "done"}

    用法：
        async for event in rag_query_stream(...):
            print(event)  # {"type": "token", "content": "向量..."}
    """
    # 1. 检查集合
    _ensure_connection()
    if not utility.has_collection(collection_name):
        yield {"type": "error", "message": f"集合 '{collection_name}' 不存在"}
        yield {"type": "done"}
        return

    vector_store = get_vector_store(collection_name)

    # 2. 向量检索
    logger.info("rag_query_stream: query='%.60s...', collection='%s', top_k=%d",
                query, collection_name, top_k)
    docs = vector_store.similarity_search(query, k=top_k)

    if not docs:
        yield {"type": "token", "content": "未找到与问题相关的文档内容。"}
        yield {"type": "done"}
        return

    # 3. 构建上下文
    context = "\n\n---\n\n".join([
        f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
        for d in docs
    ])

    # 4. 流式生成
    llm = get_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个基于文档的问答助手。请严格根据以下提供的上下文来回答问题。

规则：
1. 如果在上下文中找到答案，请准确回答并引用来源
2. 如果上下文不足以回答问题，请明确说 "根据提供的文档，无法回答这个问题"
3. 不要编造不存在于上下文中的信息"""),
        ("human", "上下文资料：\n\n{context}\n\n---\n\n问题：{question}\n\n请根据以上上下文回答问题。"),
    ])
    chain = prompt | llm | StrOutputParser()

    yield {"type": "start"}

    async for chunk in chain.astream({"context": context, "question": query}):
        if chunk:
            yield {"type": "token", "content": chunk}

    # 5. 返回元数据
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "未知来源")
        if source not in sources:
            sources.append(source)

    logger.info("rag_query_stream completed: sources=%d, chunks=%d", len(sources), len(docs))
    yield {"type": "metadata", "sources": sources, "relevant_chunks": len(docs)}
    yield {"type": "done"}
