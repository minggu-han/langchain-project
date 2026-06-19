"""
文档管理 API 路由 - RAG 的文档上传、索引和查询（Milvus 版）

RAG 工作流程回顾：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
准备阶段（本模块）：
1. 上传文档（文本/文件/URL）
2. 切分文档为 chunks
3. 生成 Embeddings 向量
4. 存入 Milvus 向量数据库

查询阶段（chat.py 的 /chat/rag 接口）：
1. 用户提问
2. 检索相关 chunks（Milvus ANN 近似最近邻搜索）
3. LLM 基于检索结果生成答案
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from app.core.security import get_current_user
from app.core.logging import get_logger
from app.models.user import User
from app.langchain_utils.rag import (
    index_text,
    index_file,
    rag_query,
    get_collection_stats,
    list_collections,
    drop_collection,
    clear_collection,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["RAG 文档管理 Milvus"])


class IndexTextRequest(BaseModel):
    """文本索引请求"""
    text: str = Field(..., min_length=1, description="要索引的文本内容")
    collection_name: str = Field(default="default", description="Milvus 集合名称")
    source_name: str = Field(default="手动输入", description="来源标识")


# ================================================================
# 1. 索引纯文本
# ================================================================
@router.post(
    "/index/text",
    summary="[RAG] 索引纯文本到 Milvus",
    description="""
将文本内容索引到 Milvus 向量数据库，之后可以通过 RAG 查询。

完整的索引流程：
1. 创建 Document 对象
2. 使用 RecursiveCharacterTextSplitter 切分
3. 生成 OpenAI Embeddings（1536 维向量）
4. 存入 Milvus Collection

学习要点：
- RecursiveCharacterTextSplitter 的切分策略
- chunk_size 和 chunk_overlap 的作用
- Milvus Collection 就是向量表
""",
)
async def index_text_endpoint(
    request: IndexTextRequest,
    current_user: User = Depends(get_current_user),
):
    """索引文本到 Milvus"""
    logger.info("POST /documents/index/text - user=%s, collection=%s, text_len=%d",
                current_user.username, request.collection_name, len(request.text))
    try:
        chunk_count = await index_text(
            text=request.text,
            collection_name=request.collection_name,
            source_name=request.source_name,
        )
        return {
            "message": "文本索引成功",
            "chunks_created": chunk_count,
            "collection": request.collection_name,
            "vector_db": "Milvus",
        }
    except Exception as e:
        logger.error("POST /documents/index/text - failed: %s", e)
        raise HTTPException(status_code=500, detail=f"索引失败: {str(e)}")


# ================================================================
# 2. 上传文件索引
# ================================================================
@router.post(
    "/index/file",
    summary="[RAG] 上传并索引文件到 Milvus",
    description="""
上传文件（支持 .txt 和 .pdf），自动索引到 Milvus。

支持格式：.txt、.pdf
""",
)
async def index_file_endpoint(
    file: UploadFile = File(..., description="要上传的文件（.txt 或 .pdf）"),
    collection_name: str = Form(default="default", description="Milvus 集合名称"),
    current_user: User = Depends(get_current_user),
):
    """上传文件并索引到 Milvus"""
    import tempfile
    import os

    logger.info("POST /documents/index/file - user=%s, file=%s, collection=%s",
                current_user.username, file.filename, collection_name)

    allowed_extensions = [".txt", ".pdf"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式 '{file_ext}'，仅支持 {', '.join(allowed_extensions)}",
        )

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
            prefix="rag_upload_",
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        chunk_count = await index_file(
            file_path=tmp_path,
            collection_name=collection_name,
        )

        os.unlink(tmp_path)

        return {
            "message": f"文件 '{file.filename}' 索引成功",
            "chunks_created": chunk_count,
            "collection": collection_name,
            "vector_db": "Milvus",
        }
    except Exception as e:
        logger.error("POST /documents/index/file - failed: %s", e)
        raise HTTPException(status_code=500, detail=f"文件索引失败: {str(e)}")


# ================================================================
# 3. 查看所有 Collections
# ================================================================
@router.get(
    "/collections",
    summary="[RAG] 列出所有 Milvus 集合",
    description="查看 Milvus 中所有的向量集合。",
)
async def list_all_collections(
    current_user: User = Depends(get_current_user),
):
    """列出所有 Milvus 集合"""
    collections = list_collections()
    return {
        "collections": collections,
        "total": len(collections),
        "vector_db": "Milvus",
    }


# ================================================================
# 4. 查询集合统计信息
# ================================================================
@router.get(
    "/collections/{collection_name}/stats",
    summary="[RAG] Milvus 集合统计",
    description="查看 Milvus 集合的文档数量等统计信息。",
)
async def collection_stats(
    collection_name: str = "default",
    current_user: User = Depends(get_current_user),
):
    """获取 Milvus 集合统计"""
    stats = get_collection_stats(collection_name)
    return stats


# ================================================================
# 5. 删除集合
# ================================================================
@router.delete(
    "/collections/{collection_name}",
    summary="[RAG] 删除 Milvus 集合",
    description="永久删除指定的 Milvus 集合及其所有数据。",
)
async def delete_collection_endpoint(
    collection_name: str,
    current_user: User = Depends(get_current_user),
):
    """删除 Milvus 集合"""
    try:
        drop_collection(collection_name)
        return {
            "message": f"Milvus 集合 '{collection_name}' 已删除",
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


# ================================================================
# 6. 清空集合
# ================================================================
@router.delete(
    "/collections/{collection_name}/clear",
    summary="[RAG] 清空 Milvus 集合数据",
    description="清空集合中的所有向量数据（删除并重建）。",
)
async def clear_collection_endpoint(
    collection_name: str,
    current_user: User = Depends(get_current_user),
):
    """清空集合数据"""
    try:
        clear_collection(collection_name)
        return {
            "message": f"Milvus 集合 '{collection_name}' 数据已清空",
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空失败: {str(e)}")


# ================================================================
# 7. RAG 查询（便捷接口，带认证）
# ================================================================
@router.post(
    "/query",
    summary="[RAG] 文档查询（Milvus）",
    description="基于 Milvus 中已索引的文档回答问题（需要先通过 /index 接口索引文档）。",
)
async def query_documents(
    query: str = Form(..., description="要查询的问题"),
    collection_name: str = Form(default="default", description="Milvus 集合名称"),
    top_k: int = Form(default=4, ge=1, le=20, description="返回的相关文档数量"),
    current_user: User = Depends(get_current_user),
):
    """基于 Milvus 的 RAG 文档查询"""
    result = await rag_query(
        query=query,
        collection_name=collection_name,
        top_k=top_k,
    )
    return result
