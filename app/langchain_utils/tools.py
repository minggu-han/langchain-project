"""
LangChain 扩展篇 —— Tools（自定义工具）

什么是 Tool（工具）？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LLM 本身只能生成文本，不能：
- 搜索互联网
- 查询数据库
- 执行计算
- 读取文件
- 调用 API

Tool 就是给 LLM 装上 "手脚"，让它能执行这些操作。

工具的定义需要：
1. name：工具名称（LLM 用来识别要调用哪个工具）
2. description：工具描述（LLM 用来判断何时使用该工具）
3. func：实际执行的函数
4. args_schema：参数的 JSON Schema（LLM 用来生成正确的参数）

本模块演示：
├── 计算器工具       —— 安全的数学表达式求值
├── 时间日期工具     —— 获取当前时间
├── 单词计数工具     —— 文本统计
└── 示例：如何创建自定义工具
"""
import math
import datetime
from typing import List, Optional
from langchain_core.tools import tool


# ================================================================
# 1. 计算器工具
# ================================================================
@tool
def calculator(expression: str) -> str:
    """
    执行数学计算。支持基本运算 (+, -, *, /, **, %) 和数学函数 (sqrt, sin, cos, log 等)。

    使用场景：
    - 用户问 "123 * 456 等于多少？"
    - LLM 识别出需要计算，调用 calculator("123 * 456")
    - 计算器返回 "56088"
    - LLM 将结果整合到回答中

    Args:
        expression: 数学表达式，例如 "2 + 2", "sqrt(16)", "sin(pi/2)"
    """
    # 安全限制：只允许安全的数学运算符和函数
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("__")
    }
    allowed_names.update({
        "abs": abs,
        "round": round,
        "int": int,
        "float": float,
        "pow": pow,
        "max": max,
        "min": min,
        "sum": sum,
        "pi": math.pi,
        "e": math.e,
    })

    try:
        # 使用 eval 在受限环境中执行表达式
        # 注意：生产环境应使用专门的表达式求值库（如 numexpr）
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# ================================================================
# 2. 时间日期工具
# ================================================================
@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """
    获取当前日期和时间。

    使用场景：
    - 用户问 "现在几点了？"
    - 用户问 "今天是星期几？"

    Args:
        timezone: 时区，如 "Asia/Shanghai"（默认）, "America/New_York", "Europe/London"
    """
    now = datetime.datetime.utcnow()
    return f"当前 UTC 时间: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n星期: {now.strftime('%A')}\n这是今年的第 {now.timetuple().tm_yday} 天"


# ================================================================
# 3. 文本统计工具
# ================================================================
@tool
def text_statistics(text: str) -> str:
    """
    统计文本的字符数、单词数、行数等信息。

    使用场景：
    - 用户问 "这段文字有多少字？"
    - 分析文本长度

    Args:
        text: 要统计的文本
    """
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", "").replace("\n", ""))
    words = len(text.split())
    lines = text.count("\n") + 1
    paragraphs = len([p for p in text.split("\n\n") if p.strip()])

    return f"""文本统计结果：
- 字符数（含空格）: {chars}
- 字符数（不含空格）: {chars_no_spaces}
- 单词数: {words}
- 行数: {lines}
- 段落数: {paragraphs}"""


# ================================================================
# 4. 所有可用工具的列表
# ================================================================
# Agent 会从这个列表中选择合适的工具
# 工具越多，Agent 能力越强，但选择也越困难
ALL_TOOLS = [
    calculator,
    get_current_time,
    text_statistics,
]


def get_tools_for_agent(tool_names: Optional[List[str]] = None):
    """
    获取指定名称的工具列表

    Args:
        tool_names: 需要的工具名称列表，None 表示使用全部工具
                   例如：["calculator", "get_current_time"]

    Returns:
        工具列表
    """
    if tool_names is None:
        return ALL_TOOLS

    tool_map = {t.name: t for t in ALL_TOOLS}
    return [tool_map[name] for name in tool_names if name in tool_map]
