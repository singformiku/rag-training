"""
Tool 基礎類別
==============
定義工具的統一介面。所有的工具都必須：
1. 有 name, description, input_schema (供模型理解怎麼呼叫)
2. 實作 execute() 方法 (實際執行)

模型如何知道要用這個工具？
- API 呼叫時我們會把 tools 列表 (OpenAI function-calling 格式) 送過去
- 模型看到 description 判斷是否適合用
- 模型生成符合 parameters (= input_schema) 的參數
- 我們收到 ``tool_calls`` 後呼叫 execute()
"""
from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """所有工具的抽象基類。"""

    # 子類別必須覆寫這三個屬性
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {}

    # 是否為「危險操作」，需要人工確認 (例如刪檔、執行任意 bash)
    requires_confirmation: bool = False

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """
        實際執行工具。

        Args:
            **kwargs: 由模型生成的參數，結構符合 input_schema

        Returns:
            str: 執行結果。會作為 ``role: tool`` 訊息送回模型。
        """
        raise NotImplementedError

    def to_api_format(self) -> dict[str, Any]:
        """
        轉成 OpenAI function-calling 格式。

        格式範例::

            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"


class ToolRegistry:
    """
    工具註冊表。

    負責管理所有可用的工具，並把它們轉成 API 格式。
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """註冊一個工具。"""
        if not tool.name:
            raise ValueError(f"Tool {tool.__class__.__name__} 缺少 name 屬性")
        self._tools[tool.name] = tool

    def register_all(self, tools: list[Tool]) -> None:
        """批次註冊。"""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        """依名稱取得工具。"""
        return self._tools.get(name)

    def to_api_format(self) -> list[dict[str, Any]]:
        """把所有工具轉成 API 需要的格式。"""
        return [tool.to_api_format() for tool in self._tools.values()]

    def list_tools(self) -> list[str]:
        """列出所有已註冊的工具名稱。"""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)
