"""
Memory 模組
===========
負責管理對話歷史。Claude 本身沒有記憶，每次 API 呼叫都要把
完整歷史送過去，這個模組就是把歷史存起來並格式化。

設計重點：
- 分離「user/assistant 訊息」和「tool_use/tool_result」
- 避免 token 無限增長：提供 trim() 方法
- 可擴充成長期記憶 (向量庫、檔案持久化)
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Memory:
    """
    Agent 的對話記憶。

    訊息格式遵循 Anthropic API 規範：
    - user / assistant 角色
    - content 可以是字串，或由 text / tool_use / tool_result blocks 組成的列表
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    max_messages: int = 100  # 保留的最多訊息數，防止爆 context

    def add_user_message(self, content: str | list) -> None:
        """新增使用者訊息。"""
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str | list) -> None:
        """新增 Claude 的回應訊息。"""
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def add_tool_result(self, tool_use_id: str, result: str, is_error: bool = False) -> None:
        """
        新增工具執行結果。

        在 Anthropic API 中，tool_result 是包在 user role 訊息裡的特殊 block。
        這是因為 API 把「工具的輸出」視為「給 Claude 的新輸入」。
        """
        self.messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result,
                "is_error": is_error,
            }],
        })
        self._trim()

    def _trim(self) -> None:
        """簡單的記憶修剪：超出 max_messages 時砍掉最舊的一般訊息。"""
        if len(self.messages) > self.max_messages:
            # 保留最近的訊息，但要小心 tool_use 和 tool_result 必須配對
            # 這裡做簡化處理：直接砍最前面
            self.messages = self.messages[-self.max_messages:]

    def clear(self) -> None:
        """清空記憶。"""
        self.messages = []

    def get_messages(self) -> list[dict[str, Any]]:
        """取得完整的訊息列表，用於 API 呼叫。"""
        return self.messages

    def __len__(self) -> int:
        return len(self.messages)
