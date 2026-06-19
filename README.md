# 🦜🔗 LangChain 学习项目

基于 **FastAPI** + **PostgreSQL** + **Redis** + **JWT认证** 的 LangChain 学习平台。

## 🚀 快速开始

### 1. 环境准备

```bash
# 确保 PostgreSQL 和 Redis 已启动

# 进入项目目录
cd langchain-project
```

### 2. 配置环境变量

```bash
# 复制配置文件
cp .env.example .env

# 编辑 .env，填入你的 OpenAI API Key
# OPENAI_API_KEY=sk-your-key-here
```

### 3. 安装依赖（使用 uv）

```bash
# uv 会自动读取 pyproject.toml 并安装所有依赖
uv sync

# 如果需要开发依赖（pytest 等）
uv sync --dev
```

### 4. 启动应用

```bash
# 方式1：使用 uv run
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

# 方式2：直接使用 uvicorn（需要先激活虚拟环境）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

### 5. 访问 API 文档

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## 📚 学习路径

### 第一步：注册和认证

在 Swagger UI 中：
1. 调用 `POST /api/v1/auth/register` 注册用户
2. 调用 `POST /api/v1/auth/login` 获取 Token
3. 点击页面右上角 "Authorize" 按钮，输入 Token
4. 认证后，所有需要登录的接口都可以使用了

### 第二步：基础对话 —— 理解 Chain

| 接口 | LangChain 概念 | 学习要点 |
|------|---------------|---------|
| `POST /api/v1/chat/simple` | Chain (链) | Prompt → LLM → OutputParser |
| `POST /api/v1/chat/role` | Role Prompting | System Prompt 控制 AI 行为 |
| `POST /api/v1/chat/translate` | Prompt 模板 | 专用提示词设计 |
| `POST /api/v1/chat/code-review` | 结构化输出 | Markdown 格式输出 |

**重点文件**：`app/langchain_utils/chains.py`

### 第三步：带记忆对话 —— 理解 Memory

| 接口 | LangChain 概念 | 学习要点 |
|------|---------------|---------|
| `POST /api/v1/chat/with-memory` | ConversationMemory | 对话历史管理 |

**重点文件**：`app/langchain_utils/memory.py`

**关键代码**：
```python
# RunnableWithMessageHistory 包装链，自动管理历史
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,          # 获取历史的函数
    input_messages_key="input",   # 输入键名
    history_messages_key="history", # 历史键名
)
```

### 第四步：RAG —— 理解检索增强生成

| 接口 | LangChain 概念 | 学习要点 |
|------|---------------|---------|
| `POST /api/v1/documents/index/text` | DocumentLoader | 创建文档对象 |
| `POST /api/v1/documents/index/file` | TextSplitter | 文档切分策略 |
| `POST /api/v1/chat/rag` | Retriever + VectorStore | 向量检索 + 生成 |

**重点文件**：`app/langchain_utils/rag.py`

**RAG 工作流程**：
```
上传文档 → 切分 → Embedding → Milvus 存储
    ↓
用户提问 → Embedding → 检索相似文档 → LLM 生成答案
```

### 第五步：Agent —— 理解智能代理

| 接口 | LangChain 概念 | 学习要点 |
|------|---------------|---------|
| `POST /api/v1/chat/agent` | Agent + Tools | ReAct 循环 |
| `GET /api/v1/chat/tools` | Tool 定义 | 工具注册和描述 |

**重点文件**：`app/langchain_utils/agents.py` 和 `app/langchain_utils/tools.py`

**Agent 的 ReAct 循环**：
```
Thought（思考）→ Action（行动）→ Observation（观察）→ 重复...
```

## 🏗️ 项目结构

```
langchain-project/
├── pyproject.toml           # 项目配置和依赖
├── .env                     # 环境变量（不提交到 Git）
├── .env.example             # 环境变量模板
├── README.md                # 项目文档
│
├── app/
│   ├── __init__.py
│   ├── main.py              # 应用入口 + 生命周期管理
│   ├── config.py            # 配置管理（pydantic-settings）
│   │
│   ├── core/                # 基础设施层
│   │   ├── __init__.py
│   │   ├── database.py      # SQLAlchemy 异步引擎 + 会话管理
│   │   ├── redis.py         # Redis 客户端 + 连接池
│   │   └── security.py      # JWT Token + 密码哈希
│   │
│   ├── models/              # ORM 数据模型
│   │   ├── __init__.py
│   │   ├── user.py          # 用户表（SQLAlchemy 2.0 Mapped）
│   │   └── chat.py          # 聊天记录表
│   │
│   ├── schemas/             # Pydantic 数据验证
│   │   ├── __init__.py
│   │   ├── user.py          # 用户请求/响应模型
│   │   └── chat.py          # 聊天请求/响应模型
│   │
│   ├── api/                 # API 路由
│   │   ├── __init__.py
│   │   ├── auth.py          # 认证接口（注册/登录/Token）
│   │   ├── chat.py          # LangChain 对话接口
│   │   └── documents.py     # 文档管理接口（RAG）
│   │
│   └── langchain_utils/     # LangChain 功能模块
│       ├── __init__.py      # LangChain 概念速查表
│       ├── llm_factory.py   # LLM 工厂（单例模式）
│       ├── chains.py        # 基础链（对话/翻译/审阅）
│       ├── memory.py        # 对话记忆（Buffer/Summary）
│       ├── tools.py         # 自定义工具（计算器/时间/统计）
│       ├── rag.py           # RAG（文档加载/切分/检索/生成）
│       └── agents.py        # Agent（OpenAI Functions Agent）
```

## 🔑 核心概念对照表

| LangChain 概念 | 对应模块 | 一句话解释 |
|---------------|---------|-----------|
| **PromptTemplate** | chains.py | 用模板格式化给 LLM 的提示词 |
| **LCEL** | chains.py | 用 `|` 管道符串联组件 |
| **ChatModel** | llm_factory.py | 对话式语言模型（GPT-4, Claude...） |
| **Memory** | memory.py | 让 LLM 记住之前的对话 |
| **DocumentLoader** | rag.py | 加载各种格式的文档 |
| **TextSplitter** | rag.py | 将长文档切分成小块 |
| **Embeddings** | rag.py | 把文本变成数字向量 |
| **VectorStore** | rag.py | 存储和检索向量 |
| **Retriever** | rag.py | 从向量库中找相关内容 |
| **Tool** | tools.py | 给 LLM 装上手，能调用外部函数 |
| **Agent** | agents.py | 能自主决策的智能代理 |
| **Chain** | chains.py | 将多个步骤串起来执行 |

## 🛠️ 常用命令

```bash
# 安装依赖
uv sync

# 启动开发服务器
uv run uvicorn app.main:app --reload

# 代码检查（如果安装 ruff）
uv run ruff check .

# 运行测试（如果有）
uv run pytest
```

## ⚙️ 依赖说明

| 包名 | 用途 |
|------|------|
| `fastapi` | 高性能异步 Web 框架 |
| `uvicorn` | ASGI 服务器 |
| `sqlalchemy[asyncio]` | ORM 框架（异步模式） |
| `asyncpg` | PostgreSQL 异步驱动 |
| `redis` | Redis 客户端 |
| `python-jose` | JWT 生成与验证 |
| `passlib[bcrypt]` | 密码哈希 |
| `pydantic-settings` | 配置管理 |
| `langchain` | LangChain 核心 |
| `langchain-openai` | OpenAI 集成 |
| `langchain-milvus` | Milvus 向量数据库集成 |
