"""
Skills 技能相关 Pydantic Schema

Skills 是什么？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Skill 是比 Chain / Tool 更高一层的抽象，面向用户的真实任务场景。

每个 Skill 封装了：
- 名称 + 描述 + 分类     → 帮助用户发现和理解技能
- 输入参数 Schema        → 告诉用户需要提供哪些参数
- 执行函数              → 内部调用 LangChain 链（Prompt + LLM + ...）

Skills vs Chain vs Tool：
┌──────────┬──────────────────────────────────────────────────────┐
│ 概念      │ 关注点                                               │
├──────────┼──────────────────────────────────────────────────────┤
│ Tool      │ 原子操作：计算、查时间、文本统计                      │
│ Chain     │ 技术串联：Prompt → LLM → OutputParser                │
│ Skill     │ 任务场景：代码审阅、翻译、邮件撰写（用户视角）         │
│ Agent     │ 自主决策：自动选择 Skill/Tool 完成任务                │
└──────────┴──────────────────────────────────────────────────────┘
"""
from typing import Optional, List
from pydantic import BaseModel, Field


# ================================================================
# 1. 技能基本信息（列表展示用）
# ================================================================
class SkillInfo(BaseModel):
    """
    技能基本信息 — 用于列表展示

    返回给前端的技能卡片信息，包含分类、标签帮助用户筛选
    """
    name: str = Field(
        ...,
        description="技能唯一标识（kebab-case 命名）",
        examples=["code-review", "translate", "summarize"],
    )
    display_name: str = Field(
        ...,
        description="技能显示名称（中文）",
        examples=["代码审阅", "多语言翻译", "文本摘要"],
    )
    description: str = Field(
        ...,
        description="技能简要描述",
        examples=["对代码进行全面审阅，包括质量、安全、性能分析"],
    )
    category: str = Field(
        ...,
        description="技能分类",
        examples=["code", "text", "productivity", "education"],
    )
    tags: List[str] = Field(
        default_factory=list,
        description="技能标签",
        examples=[["代码", "审阅", "重构"]],
    )
    has_stream: bool = Field(
        default=False,
        description="是否支持流式输出",
    )


# ================================================================
# 2. 技能详情（含参数 Schema，单个技能查看用）
# ================================================================
class SkillParamField(BaseModel):
    """技能参数中的单个字段"""
    name: str = Field(..., description="参数名")
    type: str = Field(..., description="参数类型", examples=["string", "integer", "number"])
    required: bool = Field(default=True, description="是否必填")
    default: Optional[str] = Field(default=None, description="默认值")
    description: str = Field(default="", description="参数说明")


class SkillDetail(BaseModel):
    """
    技能详情 — 包含完整的参数说明

    与 SkillInfo 相比，多了 input_fields，告诉调用者需要传什么参数
    """
    name: str
    display_name: str
    description: str
    category: str
    tags: List[str] = Field(default_factory=list)
    has_stream: bool = False
    input_fields: List[SkillParamField] = Field(
        default_factory=list,
        description="技能需要的输入参数列表",
    )


# ================================================================
# 3. 技能调用请求
# ================================================================
class SkillInvokeRequest(BaseModel):
    """
    技能调用请求

    参数说明：
    - params: 传递给技能的参数，key 与技能定义的 input_fields 对应
    - 例如 code-review 技能：{"code": "...", "language": "Python"}

    示例：
    ```json
    {
        "params": {
            "code": "def hello(): print('world')",
            "language": "Python"
        }
    }
    ```
    """
    params: dict = Field(
        default_factory=dict,
        description="传递给技能的参数（key-value 对象）",
        examples=[{"code": "def hello(): print('world')", "language": "Python"}],
    )


# ================================================================
# 4. 技能调用响应
# ================================================================
class SkillInvokeResponse(BaseModel):
    """
    技能调用响应
    """
    skill_name: str = Field(..., description="调用的技能名称")
    result: str = Field(..., description="技能执行结果")
    model_name: str = Field(..., description="使用的 LLM 模型名称")
    params_used: dict = Field(
        default_factory=dict,
        description="实际使用的参数（含默认值填充后的完整参数）",
    )


# ================================================================
# 5. 技能列表响应
# ================================================================
class SkillListResponse(BaseModel):
    """
    技能列表响应
    """
    skills: List[SkillInfo] = Field(..., description="技能列表")
    total: int = Field(..., description="技能总数")
    categories: List[str] = Field(
        default_factory=list,
        description="所有分类（用于筛选）",
    )


# ================================================================
# 6. 智能 Skill 对话请求
# ================================================================
class SkillChatRequest(BaseModel):
    """
    智能 Skill 对话请求 — 用户只需发送自然语言消息

    系统会自动识别意图 → 匹配 Skill → 按步骤执行。

    路由模式（mode）：
    - auto（默认）：关键词优先匹配 → 低分时 LLM 二次判断
    - keyword：仅关键词匹配，匹配不到返回错误
    - llm：将所有 Skill 定义注入 prompt，LLM 自主判断并执行
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="用户自然语言消息，系统自动识别要用哪个 Skill",
        examples=["帮我审阅这段 Python 代码：def hello(): print('world')"],
    )
    mode: str = Field(
        default="auto",
        pattern="^(auto|llm|keyword)$",
        description="路由模式：auto（双重路由）/ llm（LLM 自主判断）/ keyword（纯关键词）",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="会话 ID，可选用于多轮对话",
    )


# ================================================================
# 7. 智能 Skill 对话响应
# ================================================================
class SkillChatResponse(BaseModel):
    """
    智能 Skill 对话响应
    """
    skill_used: Optional[str] = Field(
        default=None,
        description="系统自动选择的技能名称，None 表示未匹配到任何技能",
        examples=["code-review"],
    )
    route_method: str = Field(
        ...,
        description="路由方式：keyword（关键词匹配）/ llm（LLM 分类）/ none（未匹配）",
    )
    result: str = Field(
        ...,
        description="技能执行结果（AI 的回复）",
    )
    model_name: str = Field(
        ...,
        description="使用的 LLM 模型名称",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="会话 ID",
    )


# ================================================================
# 8. 全功能 Agent 对话请求
# ================================================================
class AgentChatRequest(BaseModel):
    """
    全功能 Agent 对话请求

    Agent 自主决定使用以下能力：
    - Skills：8 个预置技能（代码审阅、翻译、摘要...）
    - Tools：calculator、get_current_time、text_statistics
    - RAG：search_documents 文档检索
    - Chain：通用对话
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="用户消息（自然语言），Agent 自主判断该用什么能力",
        examples=["帮我审阅这段代码，然后计算它的时间复杂度"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="会话 ID，传相同 ID 保持多轮对话上下文",
    )
    temperature: Optional[float] = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="LLM 温度参数",
    )


# ================================================================
# 9. 全功能 Agent 对话响应
# ================================================================
class AgentStepInfo(BaseModel):
    """Agent 执行的一个中间步骤"""
    tool: str = Field(..., description="使用的工具名称")
    tool_input: str = Field(..., description="传给工具的输入")
    observation: str = Field(..., description="工具返回的结果（截断）")


class AgentChatResponse(BaseModel):
    """
    全功能 Agent 对话响应
    """
    answer: str = Field(..., description="Agent 的最终答案")
    session_id: str = Field(..., description="会话 ID")
    model_name: str = Field(..., description="使用的 LLM 模型名称")
    steps: list[AgentStepInfo] = Field(
        default_factory=list,
        description="Agent 的中间步骤（工具调用过程）",
    )
    total_steps: int = Field(default=0, description="总步骤数")
    rag_used: bool = Field(
        default=False,
        description="是否使用了 RAG（Milvus 检索到了相关文档）",
    )
