"""
搜尋工具
========
這裡提供兩種示範：
1. MockSearchTool：用假資料示範架構 (不需 API Key)
2. AnthropicWebSearchTool：使用 Anthropic API 內建的 web_search (生產環境推薦)

擴充方向：
- 串接 Tavily / Serper / Brave Search API
- 加入向量資料庫做語意檢索
"""
from src.tools.base import Tool


class MockSearchTool(Tool):
    """
    假的搜尋工具，示範架構用。
    實際專案應替換為真實的搜尋 API。
    """

    name = "web_search"
    description = "在網路上搜尋資訊，回傳相關結果的摘要。"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜尋關鍵字",
            },
            "max_results": {
                "type": "integer",
                "description": "最多回傳幾筆結果",
                "default": 3,
            },
        },
        "required": ["query"],
    }

    def execute(self, query: str, max_results: int = 3) -> str:
        # 這裡只是示範，實際應串接 Tavily / Serper / Google Custom Search
        results = [
            f"[模擬結果 {i+1}] 關於 '{query}' 的資訊：這是第 {i+1} 個搜尋結果的摘要。"
            for i in range(max_results)
        ]
        return "\n\n".join(results) + "\n\n⚠️ 注意：這是 MockSearchTool，請替換為真實 API"


class CalculatorTool(Tool):
    """一個簡單的計算工具，示範無風險工具。"""

    name = "calculator"
    description = "執行數學計算。支援 +、-、*、/、** 等基本運算。"
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "要計算的數學表達式，例如 '2 + 3 * 4'",
            },
        },
        "required": ["expression"],
    }

    def execute(self, expression: str) -> str:
        # 安全的 eval：只允許數字和運算子
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "❌ 表達式含有不允許的字元"
        try:
            result = eval(expression)  # noqa: S307 - 已做字元限制
            return f"{expression} = {result}"
        except Exception as e:
            return f"❌ 計算錯誤：{e}"
