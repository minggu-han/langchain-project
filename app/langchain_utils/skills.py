"""
LangChain 实战篇 —— Skills（智能路由技能系统）

什么是 Skill（技能）？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Skill 是一个面向任务场景的、带步骤的 System Prompt 模板。

核心理念：
  用户说一句话 → 系统自动识别意图 → 匹配 Skill → 按 Skill 步骤执行

每个 Skill 包含：
├── name              → 唯一标识（kebab-case）
├── display_name      → 显示名称（中文）
├── description       → 技能描述（用于 LLM 意图分类）
├── trigger_keywords  → 触发关键词（用于快速关键词匹配）
├── category          → 分类
├── tags              → 标签
├── input_fields      → 输入参数定义
├── system_prompt     → 带步骤的 System Prompt（核心！）
├── execute()         → 同步执行
└── execute_stream()  → 流式执行

意图路由模式（两种）：
┌──────────────────┬──────────────────────────────────────────────┐
│ 模式              │ 说明                                         │
├──────────────────┼──────────────────────────────────────────────┤
│ keyword（快速）    │ 关键词匹配 → 高分直接用，无需 LLM 调用        │
│ llm（智能）        │ 将所有 Skill 定义发给 LLM，让 LLM 自己选      │
│ auto（默认）       │ keyword 优先 → 低分时 fallback 到 llm        │
└──────────────────┴──────────────────────────────────────────────┘

LLM 自主判断模式（mode=llm）：
  将所有 Skill 的 name + description 列在 system prompt 中，
  LLM 读取用户消息后自己决定用哪个 Skill 然后按步骤执行。
  一个 LLM 调用完成识别 + 执行。
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, AsyncIterator
import re

from app.langchain_utils.chains import basic_chat, basic_chat_stream
from app.langchain_utils.llm_factory import get_llm
from app.core.logging import get_logger

logger = get_logger(__name__)


# ================================================================
# 1. Skill 数据结构
# ================================================================
@dataclass
class Skill:
    """
    技能数据结构

    system_prompt 是核心 —— 它同时包含角色设定 + 执行步骤，
    LLM 拿到后按步骤输出结果。
    """
    name: str
    display_name: str
    description: str
    category: str
    trigger_keywords: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    input_fields: list = field(default_factory=list)
    system_prompt: str = ""
    execute: Optional[Callable] = None
    execute_stream: Optional[Callable] = None

    @property
    def has_stream(self) -> bool:
        return self.execute_stream is not None


# ================================================================
# 2. 全局注册表
# ================================================================
SKILL_REGISTRY: dict[str, Skill] = {}


def register(skill: Skill) -> Skill:
    SKILL_REGISTRY[skill.name] = skill
    logger.debug("Registered skill: %s (%s)", skill.name, skill.display_name)
    return skill


def get_skill(name: str) -> Skill | None:
    return SKILL_REGISTRY.get(name)


def get_all_skills() -> list[Skill]:
    return list(SKILL_REGISTRY.values())


def get_skills_by_category(category: str) -> list[Skill]:
    return [s for s in SKILL_REGISTRY.values() if s.category == category]


# ================================================================
# 3. 辅助函数
# ================================================================
def _field(name: str, type_: str, description: str,
           required: bool = True, default: str | None = None) -> dict:
    return {
        "name": name, "type": type_,
        "required": required, "default": default,
        "description": description,
    }


# ================================================================
# 4. 预置技能定义（每个 skill 的 system_prompt 包含执行步骤）
# ================================================================

# ─── 4.1 代码审阅 ───
_CODEREVIEW_PROMPT = """你是一位资深代码审阅专家。请严格按以下步骤执行代码审阅：

## 执行步骤

### 第一步：代码理解
- 读取用户提供的代码
- 判断编程语言
- 概括代码的用途和核心逻辑

### 第二步：质量分析
- 检查代码风格是否规范
- 评估命名是否清晰有意义
- 检查注释是否充分
- 评估代码结构是否合理

### 第三步：Bug 和安全检查
- 寻找潜在的逻辑错误
- 检查边界条件处理
- 审查安全漏洞（注入攻击、权限问题等）
- 检查异常处理是否完善

### 第四步：性能评估
- 识别性能瓶颈
- 检查是否有不必要的重复计算
- 评估内存使用和算法复杂度

### 第五步：改进建议
- 给出具体的改进方案
- 提供优化后的代码示例
- 推荐相关的最佳实践

请用 Markdown 格式输出完整的审阅报告。"""

register(Skill(
    name="code-review",
    display_name="代码审阅",
    description="对代码进行全面审阅：代码质量、安全漏洞、性能优化、最佳实践建议",
    category="code",
    trigger_keywords=[
        "审阅", "review", "代码检查", "代码审查", "code review",
        "审查代码", "检查代码", "看下代码", "看看代码", "代码有问题",
        "bug", "重构", "refactor", "优化代码", "代码质量",
    ],
    tags=["代码", "审阅", "重构", "安全"],
    input_fields=[
        _field("code", "string", "要审阅的源代码"),
        _field("language", "string", "编程语言", required=False, default="Python"),
    ],
    system_prompt=_CODEREVIEW_PROMPT,
))


# ─── 4.2 多语言翻译 ───
_TRANSLATE_PROMPT = """你是一位专业的多语言翻译专家。请严格按以下步骤执行翻译：

## 执行步骤

### 第一步：语言识别
- 识别源文本的语言
- 确认目标语言

### 第二步：内容理解
- 理解文本的语境和主题
- 识别专业术语和文化特定表达
- 判断文本的语气（正式/口语/技术）

### 第三步：翻译执行
- 逐句翻译，保持原文结构
- 专业术语使用标准译法
- 文化特定表达寻找等价表达

### 第四步：质量检查
- 检查语法是否正确
- 确认术语一致性
- 验证语气是否匹配原文
- 确保没有漏译

输出格式：
**原文**：...
**译文**：...
**翻译说明**：（如有特殊处理，简要说明）"""

register(Skill(
    name="translate",
    display_name="多语言翻译",
    description="将文本翻译为目标语言，保持原文语气和风格，专业术语翻译准确",
    category="text",
    trigger_keywords=[
        "翻译", "translate", "译成", "翻成", "用英语", "用英文",
        "用日语", "用法语", "用韩语", "用德语", "翻译成",
        "translation", "中译", "英译",
    ],
    tags=["翻译", "多语言", "文本"],
    input_fields=[
        _field("text", "string", "要翻译的文本"),
        _field("target_lang", "string", "目标语言", required=False, default="英文"),
    ],
    system_prompt=_TRANSLATE_PROMPT,
))


# ─── 4.3 文本摘要 ───
_SUMMARIZE_PROMPT = """你是一位专业的文本摘要专家。请严格按以下步骤执行摘要：

## 执行步骤

### 第一步：快速浏览
- 阅读全文，把握整体主题
- 识别文章类型（新闻/论文/博客/报告）

### 第二步：提取关键信息
- 找出核心观点和结论
- 识别支撑性论据
- 标注重要数据和事实

### 第三步：组织摘要
- 按逻辑顺序排列要点
- 用简洁清晰的语言重述
- 保持原文的核心信息不丢失

### 第四步：精简优化
- 删除冗余表达
- 确保摘要长度适中（控制在原文 20%-30%）
- 检查是否保留了最重要的信息

输出格式：
## 摘要
（简洁的摘要内容）

## 关键要点
- 要点 1
- 要点 2
- ..."""

register(Skill(
    name="summarize",
    display_name="文本摘要",
    description="将长文本压缩为简洁摘要，提取核心观点和关键信息",
    category="text",
    trigger_keywords=[
        "摘要", "总结", "概括", "归纳", "summarize", "summary",
        "简述", "精简", "压缩", "太长", "缩写", "提炼",
    ],
    tags=["摘要", "总结", "文本处理", "效率"],
    input_fields=[
        _field("text", "string", "要摘要的文本"),
        _field("max_length", "integer", "摘要最大字数", required=False, default="300"),
    ],
    system_prompt=_SUMMARIZE_PROMPT,
))


# ─── 4.4 SQL 生成 ───
_SQL_PROMPT = """你是一位资深数据库工程师，精通 SQL 编写。请严格按以下步骤执行：

## 执行步骤

### 第一步：需求分析
- 理解用户的查询意图
- 识别涉及的数据表和字段
- 确认查询类型（SELECT/INSERT/UPDATE/DELETE）

### 第二步：表结构推导
- 根据用户描述推断表结构
- 确定表之间的关联关系
- 识别主键和外键

### 第三步：SQL 编写
- 编写标准 SQL（兼容 PostgreSQL/MySQL）
- 使用合适的 JOIN 类型
- 添加必要的 WHERE 条件
- 考虑聚合和排序需求

### 第四步：优化检查
- 检查索引使用情况
- 避免 SELECT *
- 考虑查询性能
- 添加注释说明

输出格式：
```sql
-- 你的 SQL 语句
```

**说明**：（解释 SQL 的逻辑和注意事项）"""

register(Skill(
    name="sql-generator",
    display_name="SQL 生成器",
    description="用自然语言描述需求，自动生成 SQL 查询语句",
    category="code",
    trigger_keywords=[
        "sql", "SQL", "查询", "数据库", "建表", "select", "SELECT",
        "join", "JOIN", "表", "字段", "写入数据库", "更新数据",
        "删除数据", "数据表", "查询语句",
    ],
    tags=["SQL", "数据库", "代码生成"],
    input_fields=[
        _field("query", "string", "用自然语言描述查询需求"),
        _field("schema_info", "string", "表结构信息（可选）", required=False, default="未提供"),
    ],
    system_prompt=_SQL_PROMPT,
))


# ─── 4.5 邮件撰写 ───
_EMAIL_PROMPT = """你是一位资深商务沟通专家。请严格按以下步骤撰写邮件：

## 执行步骤

### 第一步：需求理解
- 明确邮件类型（商务/求职/感谢/通知）
- 了解收件人身份和关系
- 确定邮件的核心目的

### 第二步：结构设计
- 撰写得体的称呼
- 设计开篇（说明来意）
- 组织正文（逻辑清晰、重点突出）
- 撰写结语（行动呼吁或礼貌收尾）

### 第三步：语言润色
- 确保语气恰当（正式/半正式/友好）
- 检查语法和拼写
- 优化表达，简洁有力

### 第四步：最终检查
- 主题行是否清晰
- 是否有遗漏的关键信息
- 附件提醒（如有需要）
- 联系方式是否正确

输出完整的邮件内容（包含主题行、称呼、正文、结语、签名）。"""

register(Skill(
    name="email-writer",
    display_name="邮件撰写",
    description="根据提供的信息自动撰写专业邮件，语言得体、结构清晰",
    category="productivity",
    trigger_keywords=[
        "邮件", "email", "写信", "发邮件", "回复邮件", "商务邮件",
        "求职信", "求职邮件", "感谢信", "通知", "邀请函", "email",
        "写邮件", "帮我写", "起草", "撰写邮件",
    ],
    tags=["邮件", "写作", "商务", "沟通"],
    input_fields=[
        _field("recipient", "string", "收件人"),
        _field("subject", "string", "邮件主题"),
        _field("key_points", "string", "邮件要点"),
        _field("email_type", "string", "邮件类型", required=False, default="商务邮件"),
    ],
    system_prompt=_EMAIL_PROMPT,
))


# ─── 4.6 概念解释 ───
_EXPLAIN_PROMPT = """你是一位耐心的技术导师，擅长把复杂概念讲得通俗易懂。请严格按以下步骤执行：

## 执行步骤

### 第一步：概念定位
- 明确要解释的概念
- 判断概念的难度级别
- 确定目标受众的知识水平

### 第二步：类比设计
- 找一个生活中的类比
- 确保类比准确且易懂
- 用类比引入概念

### 第三步：分层讲解
- 从最基础的定义开始
- 逐步深入核心原理
- 补充关键细节和注意事项

### 第四步：实例演示
- 提供实际的代码示例（如果适用）
- 展示概念的实际应用场景
- 列举常见误区

输出格式：
## 概念解释：{概念名}
### 一句话理解
### 生活中的类比
### 详细讲解
### 代码示例（如适用）
### 常见误区
### 延伸学习"""

register(Skill(
    name="explain-concept",
    display_name="概念解释",
    description="用通俗易懂的方式解释技术概念，包含类比和代码示例",
    category="education",
    trigger_keywords=[
        "解释", "什么是", "是什么意思", "不懂", "理解", "讲讲",
        "讲一下", "讲一讲", "说一下", "说说",
        "概念", "介绍", "科普", "说明", "explain", "什么是",
        "不太懂", "帮我理解", "通俗", "举个例子", "教我",
    ],
    tags=["学习", "教育", "概念", "教程"],
    input_fields=[
        _field("concept", "string", "要解释的概念"),
        _field("level", "string", "难度级别", required=False, default="中级"),
    ],
    system_prompt=_EXPLAIN_PROMPT,
))


# ─── 4.7 数据分析建议 ───
_DATA_PROMPT = """你是一位资深数据分析师。请严格按以下步骤给出分析建议：

## 执行步骤

### 第一步：数据理解
- 分析用户提供的数据描述
- 判断数据类型（结构化/非结构化/时序）
- 评估数据规模和质量

### 第二步：探索性分析建议
- 推荐合适的可视化方法
- 建议关键的统计指标
- 指出需要关注的数据特征

### 第三步：建模方案
- 根据分析目标推荐算法
- 说明每种算法的适用场景
- 提出特征工程的思路

### 第四步：行动计划
- 给出分步骤的分析计划
- 推荐可用的工具和库
- 提醒常见陷阱和注意事项

请用 Markdown 格式输出完整的分析建议报告。"""

register(Skill(
    name="data-analyzer",
    display_name="数据分析建议",
    description="根据数据描述给出分析建议：可视化方法、统计指标、建模方案",
    category="productivity",
    trigger_keywords=[
        "数据", "分析", "统计", "可视化", "建模", "数据挖掘",
        "data", "analysis", "图表", "趋势", "报表", "指标",
        "数据分析", "数据集", "特征工程",
    ],
    tags=["数据", "分析", "可视化", "建模"],
    input_fields=[
        _field("data_description", "string", "数据的简要描述"),
        _field("goal", "string", "分析目标", required=False, default="了解数据特征和规律"),
    ],
    system_prompt=_DATA_PROMPT,
))


# ─── 4.8 面试模拟 ───
_INTERVIEW_PROMPT = """你是一位资深技术面试官，在各大科技公司有 10 年以上面试经验。请严格按以下步骤进行面试模拟：

## 执行步骤

### 第一步：场景设定
- 确认面试岗位和轮次
- 简要说明面试形式和时长
- 营造专业但不压迫的氛围

### 第二步：提问环节
- 提出第一个技术问题（与岗位相关）
- 问题由浅入深，考察思维过程
- 给候选人足够的思考空间

### 第三步：回答评估
- 评价回答的正确性和深度
- 指出亮点和不足
- 给出参考答案或改进方向

### 第四步：追问深入
- 基于回答提出更深层的追问
- 考察知识广度和系统思维
- 模拟真实面试的压力和节奏

### 第五步：总结建议
- 整体评价面试表现
- 给出具体的改进建议
- 推荐复习方向和学习资源

面试风格：关注思考过程而非标准答案，给出建设性反馈。"""

register(Skill(
    name="interview-coach",
    display_name="面试模拟",
    description="模拟技术面试场景：出题、评估回答、给出改进建议",
    category="education",
    trigger_keywords=[
        "面试", "interview", "求职", "面经", "模拟面试",
        "面试题", "面试准备", "技术面", "算法题", "系统设计",
    ],
    tags=["面试", "求职", "练习", "技术"],
    input_fields=[
        _field("position", "string", "目标岗位", required=False, default="后端开发工程师"),
        _field("round_name", "string", "面试轮次", required=False, default="技术一面"),
    ],
    system_prompt=_INTERVIEW_PROMPT,
))


# ================================================================
# 5. 关键词匹配器
# ================================================================
def match_by_keywords(message: str, skills: list[Skill] | None = None) -> tuple[Skill | None, float]:
    """
    用关键词匹配找到最合适的 Skill

    算法：
    1. 将用户消息分词（按中文和英文规律）
    2. 对每个 Skill，计算 trigger_keywords 与消息词的交集
    3. 匹配分 = 命中关键词数 / skill 关键词总数
    4. 返回得分最高的 Skill（需超过阈值）

    Args:
        message: 用户消息
        skills: 候选技能列表，None 表示使用全部

    Returns:
        (匹配的 Skill, 置信度分数 0-1)，未匹配返回 (None, 0)
    """
    if skills is None:
        skills = get_all_skills()

    message_lower = message.lower()

    best_skill = None
    best_score = 0.0
    best_hits = 0

    for skill in skills:
        if not skill.trigger_keywords:
            continue

        # 计算命中的关键词数
        hits = sum(1 for kw in skill.trigger_keywords if kw.lower() in message_lower)

        if hits > 0:
            # 得分 = 命中数 / 关键词总数
            score = hits / len(skill.trigger_keywords)
            # 额外加分：命中多个关键词
            if hits >= 3:
                score = min(score + 0.15, 1.0)
            elif hits >= 2:
                score = min(score + 0.05, 1.0)

            # 同分时选命中数多的
            if score > best_score or (abs(score - best_score) < 0.001 and hits > best_hits):
                best_score = score
                best_skill = skill
                best_hits = hits

    # 返回阈值以上的结果
    # 单关键词命中需要高置信度（至少命中 2 个或分数 > 0.05 即命中率超 5%）
    KEYWORD_THRESHOLD = 0.03
    if best_score >= KEYWORD_THRESHOLD:
        logger.debug("Keyword match: skill=%s score=%.3f", best_skill.name, best_score)
        return best_skill, min(best_score, 1.0)

    return None, 0.0


# ================================================================
# 6. LLM 意图分类器
# ================================================================
# Intent classification prompt — 让 LLM 从技能列表中选择
_INTENT_CLASSIFY_PROMPT = """你是一个意图分类器。根据用户的消息，判断用户想使用以下哪个技能。

可用技能列表：
{skill_list}

用户消息：{message}

请只回复技能名称（kebab-case），不要回复其他内容。
如果用户的消息与以上技能都不匹配，请回复 "none"。

技能名称："""


async def classify_by_llm(message: str, skills: list[Skill] | None = None) -> str | None:
    """
    使用 LLM 对用户消息进行意图分类

    将技能列表 + 用户消息发给 LLM，让它选出最匹配的技能。

    Args:
        message: 用户消息
        skills: 候选技能列表，None 表示使用全部

    Returns:
        技能名称，未匹配返回 None
    """
    if skills is None:
        skills = get_all_skills()

    if not skills:
        return None

    # 构建技能列表文本
    skill_list = "\n".join(
        f"- {s.name}: {s.description}"
        for s in skills
    )

    classify_prompt = _INTENT_CLASSIFY_PROMPT.format(
        skill_list=skill_list,
        message=message,
    )

    logger.debug("LLM intent classification: %d skills, message='%.60s...'",
                 len(skills), message)

    llm = get_llm(temperature=0.1)  # 低温度，确保分类稳定
    result = await llm.ainvoke(classify_prompt)

    # 提取技能名称
    skill_name = result.content.strip().lower() if hasattr(result, 'content') else str(result).strip().lower()
    # 清理可能的额外字符
    skill_name = skill_name.split("\n")[0].strip().strip("'\"")

    logger.info("LLM classified intent: '%s' -> skill='%s'", message[:60], skill_name)

    if skill_name == "none" or skill_name not in SKILL_REGISTRY:
        return None

    return skill_name


# ================================================================
# 7. 路由函数（整合关键词 + LLM）
# ================================================================
async def route_skill(
    message: str,
    mode: str = "auto",
) -> tuple[Skill | None, str]:
    """
    意图路由：根据用户消息选择最合适的 Skill

    Args:
        message: 用户自然语言消息
        mode: 路由模式
            - "keyword": 仅关键词匹配
            - "llm": 仅 LLM 分类
            - "auto": 关键词优先 → 低分时 LLM fallback（默认）

    Returns:
        (Skill | None, route_method)
        - route_method 可选值: "keyword", "llm", "none"
    """
    logger.info("Routing intent: mode=%s, message='%.80s...'", mode, message)

    # ── 模式 1: 纯关键词 ──
    if mode == "keyword":
        skill, score = match_by_keywords(message)
        if skill:
            return skill, "keyword"
        return None, "none"

    # ── 模式 2: 纯 LLM ──
    if mode == "llm":
        skill_name = await classify_by_llm(message)
        if skill_name:
            return SKILL_REGISTRY[skill_name], "llm"
        return None, "none"

    # ── 模式 3: auto（关键词 + LLM 双重）──
    # 第一步：关键词匹配
    skill, score = match_by_keywords(message)

    # 高分直接使用
    HIGH_THRESHOLD = 0.08  # 命中足够多的关键词
    if skill and score >= HIGH_THRESHOLD:
        logger.info("Auto route: keyword high-confidence match (score=%.3f)", score)
        return skill, "keyword"

    # 低分或未匹配 → LLM 二次判断
    logger.info("Auto route: keyword low-confidence (score=%.3f), falling back to LLM", score)
    skill_name = await classify_by_llm(message)
    if skill_name:
        return SKILL_REGISTRY[skill_name], "llm"

    # LLM 也匹配不到
    return None, "none"


# ================================================================
# 8. Skill 执行函数
# ================================================================
async def execute_skill(
    skill: Skill,
    message: str,
    temperature: float = 0.3,
) -> str:
    """
    执行指定的 Skill：注入 system_prompt → LLM 按步骤生成

    Args:
        skill: 要执行的技能
        message: 用户原始消息
        temperature: LLM 温度
    """
    logger.info("Executing skill: %s, message='%.60s...'", skill.name, message)

    result = await basic_chat(
        message=message,
        system_prompt=skill.system_prompt,
        temperature=temperature,
    )

    logger.info("Skill '%s' completed: result_len=%d", skill.name, len(result))
    return result


async def execute_skill_stream(
    skill: Skill,
    message: str,
    temperature: float = 0.3,
) -> AsyncIterator[str]:
    """流式执行指定的 Skill"""
    logger.info("Executing skill stream: %s, message='%.60s...'", skill.name, message)

    async for chunk in basic_chat_stream(
        message=message,
        system_prompt=skill.system_prompt,
        temperature=temperature,
    ):
        yield chunk

    logger.info("Skill '%s' stream completed", skill.name)


# ================================================================
# 9. LLM 自主判断模式
# ================================================================
_LLM_ROUTING_SYSTEM_PROMPT = """你是一个智能助手，可以根据用户的需求自动选择合适的技能来完成任务。

## 可用技能

{skill_descriptions}

## 工作方式

当用户提出请求时：
1. 判断用户的意图
2. 选择最匹配的技能
3. 严格按照该技能的步骤执行
4. 用 Markdown 格式输出结果

如果用户的请求与以上任何技能都不匹配，请以通用助手的方式直接回答。

现在请处理用户的消息。"""


async def execute_with_llm_routing(
    message: str,
    temperature: float = 0.3,
) -> tuple[str, str | None]:
    """
    LLM 自主判断模式：将所有 Skill 注入 prompt，LLM 自己选 + 执行

    一次 LLM 调用完成意图识别 + 任务执行。

    Args:
        message: 用户消息
        temperature: LLM 温度

    Returns:
        (result, detected_skill_name | None)
    """
    skills = get_all_skills()

    # 构建所有技能描述（名称 + 描述 + 核心步骤摘要）
    skill_descriptions_parts = []
    for s in skills:
        # 提取 system_prompt 的前几行作为技能摘要
        steps_preview = s.system_prompt[:200].replace("\n", " ").strip()
        skill_descriptions_parts.append(
            f"### {s.display_name}（`{s.name}`）\n"
            f"{s.description}\n"
            f"执行方式：{steps_preview}...\n"
        )

    skill_descriptions = "\n".join(skill_descriptions_parts)

    system_prompt = _LLM_ROUTING_SYSTEM_PROMPT.format(
        skill_descriptions=skill_descriptions,
    )

    logger.info("LLM routing mode: %d skills in prompt, message='%.60s...'",
                len(skills), message)

    result = await basic_chat(
        message=message,
        system_prompt=system_prompt,
        temperature=temperature,
    )

    # 尝试检测 LLM 用了哪个技能（简单启发式 — 检查结果中是否提及技能名）
    detected_skill = None
    result_lower = result.lower()
    for s in skills:
        if s.display_name in result or s.name in result_lower:
            detected_skill = s.name
            break

    logger.info("LLM routing completed: result_len=%d, detected_skill=%s",
                len(result), detected_skill or "none")

    return result, detected_skill


async def execute_with_llm_routing_stream(
    message: str,
    temperature: float = 0.3,
) -> AsyncIterator[str]:
    """
    LLM 自主判断模式（流式版）
    """
    skills = get_all_skills()

    skill_descriptions_parts = []
    for s in skills:
        steps_preview = s.system_prompt[:200].replace("\n", " ").strip()
        skill_descriptions_parts.append(
            f"### {s.display_name}（`{s.name}`）\n"
            f"{s.description}\n"
            f"执行方式：{steps_preview}...\n"
        )

    system_prompt = _LLM_ROUTING_SYSTEM_PROMPT.format(
        skill_descriptions="\n".join(skill_descriptions_parts),
    )

    logger.info("LLM routing stream: %d skills, message='%.60s...'",
                len(skills), message)

    async for chunk in basic_chat_stream(
        message=message,
        system_prompt=system_prompt,
        temperature=temperature,
    ):
        yield chunk


# ================================================================
# 10. 带记忆的 Skill 对话（RunnableWithMessageHistory）
# ================================================================
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from app.langchain_utils.memory import get_session_history

# LLM 自主判断 + 记忆模式的 System Prompt
_LLM_ROUTING_MEMORY_PROMPT = """你是一个智能助手，可以根据用户的需求自主选择是否使用技能。

## 可用技能

{skill_descriptions}

## 对话规则

1. **自主判断**：根据用户的消息，判断是否需要使用某个技能
   - 如果消息与某个技能匹配 → 严格按照该技能的步骤执行
   - 如果消息不匹配任何技能 → 以通用助手的方式自然回复
   - 如果消息是追问/延续上一轮 → 保持上下文，不要重复执行技能

2. **利用对话历史**：结合之前的对话内容理解用户的意图
   - 如果用户说"再优化一下"，回顾上文知道他在说什么
   - 如果用户在面试模拟中说"我不太理解"，针对当前问题解释

3. **输出格式**：用 Markdown 格式，结构清晰

现在请处理用户的消息。"""

# 通用对话 + 记忆模式的 System Prompt（未匹配到 Skill 时使用）
_GENERIC_MEMORY_PROMPT = """你是一个有帮助的AI助手。请结合对话历史理解用户的意图。

如果用户之前的对话涉及特定任务（如代码审阅、翻译等），请保持上下文连贯。
如果用户开启了新话题，请自然跟随。

请用友好、专业的方式回复。"""


async def chat_with_skill_memory(
    message: str,
    session_id: str,
    mode: str = "auto",
    temperature: float = 0.3,
) -> tuple[str, str | None, str]:
    """
    带记忆的 Skill 对话 —— 利用 Redis 存储对话历史

    工作流程：
    1. 从 Redis 加载该会话的历史消息
    2. 路由选择 Skill（keyword / llm / auto）
    3. 将历史消息 + Skill system_prompt 注入 LLM
    4. LLM 生成回复
    5. 自动将本轮对话存入 Redis（RunnableWithMessageHistory 自动处理）

    Args:
        message: 用户消息
        session_id: 会话 ID（用于关联多轮对话）
        mode: 路由模式 (auto / llm / keyword)
        temperature: LLM 温度

    Returns:
        (result, skill_used | None, route_method)
    """
    llm = get_llm(temperature=temperature)

    # ── 步骤 1：路由选择 Skill ──
    if mode == "llm":
        # LLM 自主判断模式：所有 Skill 注入 prompt + 对话历史
        skills = get_all_skills()
        skill_descriptions_parts = []
        for s in skills:
            steps_preview = s.system_prompt[:200].replace("\n", " ").strip()
            skill_descriptions_parts.append(
                f"### {s.display_name}（`{s.name}`）\n"
                f"{s.description}\n"
                f"执行步骤：{steps_preview}...\n"
            )

        system_prompt = _LLM_ROUTING_MEMORY_PROMPT.format(
            skill_descriptions="\n".join(skill_descriptions_parts),
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        chain = prompt | llm | StrOutputParser()
        chain_with_history = RunnableWithMessageHistory(
            chain, get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        logger.info("chat_with_skill_memory[llm]: session=%s, message='%.60s...'",
                    session_id, message)

        result = await chain_with_history.ainvoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}},
        )

        # 启发式检测技能
        detected_skill = None
        result_lower = result.lower()
        for s in skills:
            if s.display_name in result or s.name in result_lower:
                detected_skill = s.name
                break

        return result, detected_skill, "llm"

    # ── keyword / auto 模式：先路由再执行 ──
    skill, route_method = await route_skill(message, mode=mode)

    if skill is None:
        # 未匹配 → 通用对话 + 记忆
        if mode == "keyword":
            return None, None, "none"

        prompt = ChatPromptTemplate.from_messages([
            ("system", _GENERIC_MEMORY_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        chain = prompt | llm | StrOutputParser()
        chain_with_history = RunnableWithMessageHistory(
            chain, get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        logger.info("chat_with_skill_memory[%s]: no match, session=%s", route_method, session_id)
        result = await chain_with_history.ainvoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}},
        )
        return result, None, "none"

    # 匹配到 Skill → 注入 skill.system_prompt + 记忆中已有的对话上下文
    memory_aware_prompt = skill.system_prompt + """\n\n## 对话上下文\n请结合上面的对话历史理解用户意图。如果用户的问题是上一轮的延续（如"再优化一下"、"讲详细点"），请基于历史上下文回答。"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", memory_aware_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain, get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    logger.info("chat_with_skill_memory[%s]: skill=%s, session=%s, message='%.60s...'",
                route_method, skill.name, session_id, message)

    result = await chain_with_history.ainvoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}},
    )

    logger.info("chat_with_skill_memory[%s]: skill=%s, result_len=%d",
                route_method, skill.name, len(result))

    return result, skill.name, route_method


async def chat_with_skill_memory_stream(
    message: str,
    session_id: str,
    mode: str = "auto",
    temperature: float = 0.3,
) -> AsyncIterator[str]:
    """
    带记忆的 Skill 对话（流式版） — SSE 逐 token 输出

    使用 RunnableWithMessageHistory + astream()，
    历史在流开始前加载，流结束后保存。
    """
    llm = get_llm(temperature=temperature)

    # ── LLM 自主判断模式 ──
    if mode == "llm":
        skills = get_all_skills()
        skill_descriptions_parts = []
        for s in skills:
            steps_preview = s.system_prompt[:200].replace("\n", " ").strip()
            skill_descriptions_parts.append(
                f"### {s.display_name}（`{s.name}`）\n"
                f"{s.description}\n"
                f"执行步骤：{steps_preview}...\n"
            )

        system_prompt = _LLM_ROUTING_MEMORY_PROMPT.format(
            skill_descriptions="\n".join(skill_descriptions_parts),
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        chain = prompt | llm | StrOutputParser()
        chain_with_history = RunnableWithMessageHistory(
            chain, get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        logger.info("chat_with_skill_memory_stream[llm]: session=%s", session_id)
        async for chunk in chain_with_history.astream(
            {"input": message},
            config={"configurable": {"session_id": session_id}},
        ):
            if chunk:
                yield chunk
        logger.info("chat_with_skill_memory_stream[llm] completed")
        return

    # ── keyword / auto 模式 ──
    skill, route_method = await route_skill(message, mode=mode)

    if skill is None:
        if mode == "keyword":
            yield ""  # signal no match
            return

        system_prompt = _GENERIC_MEMORY_PROMPT
        skill_name = None
    else:
        system_prompt = skill.system_prompt + """\n\n## 对话上下文\n请结合上面的对话历史理解用户意图。如果用户的问题是上一轮的延续，请基于历史上下文回答。"""
        skill_name = skill.name

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain, get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    logger.info("chat_with_skill_memory_stream[%s]: skill=%s, session=%s",
                route_method, skill_name or "none", session_id)

    async for chunk in chain_with_history.astream(
        {"input": message},
        config={"configurable": {"session_id": session_id}},
    ):
        if chunk:
            yield chunk

    logger.info("chat_with_skill_memory_stream[%s]: skill=%s, completed",
                route_method, skill_name or "none")


# ================================================================
# 11. 初始化日志
# ================================================================
logger.info(
    "Skills module loaded: %d skills — %s",
    len(SKILL_REGISTRY),
    ", ".join(SKILL_REGISTRY.keys()),
)
