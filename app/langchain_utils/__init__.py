"""
LangChain 工具模块 - 演示各种 LangChain 核心功能

本项目的学习路径：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. chains.py   —— 基础篇：Prompt Template + LLM Chain
   理解 LangChain 的核心概念：Prompt、Model、Chain
   ↓
2. memory.py   —— 进阶篇：对话记忆
   学习如何管理对话上下文，让 AI 记住之前的对话
   ↓
3. rag.py      —— 实战篇：检索增强生成（RAG）
   学习如何让 LLM 基于私有文档回答问题
   ↓
4. tools.py    —— 扩展篇：自定义工具
   学习如何给 LLM 添加外部能力（搜索、计算、数据库查询等）
   ↓
5. agents.py   —— 高级篇：智能代理
   学习如何让 LLM 自主决策、调用工具、完成复杂任务
   ↓
6. skills.py   —— 实战篇：可复用技能模板
   学习如何封装完整任务场景为可调用技能
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LangChain 核心概念速查：
┌─────────────────┬──────────────────────────────────────────────┐
│ 概念             │ 说明                                         │
├─────────────────┼──────────────────────────────────────────────┤
│ PromptTemplate   │ 提示词模板：定义如何与 LLM 交互的模板         │
│ LLM/ChatModel    │ 语言模型：OpenAI GPT、Anthropic Claude 等    │
│ Chain            │ 链：将多个步骤串联起来（LCEL 表达式语言）      │
│ Memory           │ 记忆：在多次对话中保持上下文                  │
│ Retriever        │ 检索器：从文档库中检索相关内容                │
│ DocumentLoader   │ 文档加载器：加载 PDF、TXT、网页等            │
│ TextSplitter     │ 文本分割器：将文档切分成小块                 │
│ VectorStore      │ 向量数据库：存储和检索文档向量（Milvus）     │
│ Tool             │ 工具：LLM 可以调用的外部函数                 │
│ Agent            │ 代理：自主决策并使用工具的智能体              │
│ Skill            │ 技能：任务场景封装，面向用户的完整功能      │
│ Embeddings       │ 嵌入：将文本转换为向量表示                   │
└─────────────────┴──────────────────────────────────────────────┘
"""
from app.langchain_utils.llm_factory import get_llm, get_embeddings

__all__ = ["get_llm", "get_embeddings"]
