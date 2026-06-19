"""
API 路由包 - 定义所有 HTTP 接口

路由结构：
├── /api/v1/auth/*     —— 认证相关（注册、登录、Token 刷新）
├── /api/v1/chat/*     —— LangChain 对话（基础对话、记忆、RAG、Agent）
└── /api/v1/documents/*—— 文档管理（上传、索引、查询）
"""
from fastapi import APIRouter

# 创建版本化路由
# prefix="/api/v1" 确保所有路由都有统一的版本前缀
api_router = APIRouter(prefix="/api/v1")
