"""
Memory 模組
===========
負責管理對話歷史。模型本身沒有記憶，每次 API 呼叫都要把
完整歷史送過去，這個模組就是把歷史存起來並格式化。

訊息格式遵循 OpenAI Chat Completions 規範：
- ``system`` / ``user`` / ``assistant`` / ``tool`` 角色
- ``assistant`` 訊息若含 function-calling 會帶 ``tool_calls`` 欄位
- 工具執行結果以 ``role: tool`` + ``tool_call_id`` 的訊息回覆

設計重點：
- ``system`` 訊息獨立儲存，trim 時不會被砍掉
- 支援傳入 OpenAI SDK 的 message 物件 (會自動序列化)
- 避免 token 無限增長：提供 trim() 機制
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _serialize_assistant_message(message: Any) -> Dict[str, Any]:
    """
    把 OpenAI SDK 的 ``ChatCompletionMessage`` (或 dict) 轉成可再送回
    API 的 dict。只保留 role / content / tool_calls 三個欄位。
    """
    if isinstance(message, dict):
        out: Dict[str, Any] = {"role": "assistant"}
        if message.get("content") is not None:
            out["content"] = message["content"]
        if message.get("tool_calls"):
            out["tool_calls"] = message["tool_calls"]
        return out

    # OpenAI SDK pydantic model
    out = {"role": "assistant", "content": getattr(message, "content", None)}

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "",
                },
            }
            for tc in tool_calls
        ]

    # 有些 server 會附加 reasoning_content；目前不送回下一輪 (多數 server 不接受)
    return out


@dataclass
class Memory:
    """
    Agent 的對話記憶。
    """

    messages: List[Dict[str, Any]] = field(default_factory=list)
    system_message: Optional[Dict[str, Any]] = None
    max_messages: int = 100  # 保留的最多非 system 訊息數，防止爆 context

    # ------------------------------------------------------------------
    # 寫入
    # ------------------------------------------------------------------
    def set_system_message(self, content: str) -> None:
        """設定 (或覆寫) system message。"""
        self.system_message = {"role": "system", "content": content}

    def add_user_message(self, content: str) -> None:
        """新增使用者訊息。"""
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, message: Any) -> None:
        """
        新增 assistant 回應。

        ``message`` 可以是:
        - OpenAI SDK 的 ``ChatCompletionMessage`` (含 .content / .tool_calls)
        - dict
        - 純字串 (只有 content，無 tool_calls)
        """
        if isinstance(message, str):
            self.messages.append({"role": "assistant", "content": message})
        else:
            self.messages.append(_serialize_assistant_message(message))
        self._trim()

    def add_tool_result(
        self,
        tool_call_id: str,
        result: str,
        is_error: bool = False,
    ) -> None:
        """
        新增工具執行結果 (OpenAI 格式)。

        OpenAI 把工具輸出當作一則 ``role: tool`` 的訊息，
        並用 ``tool_call_id`` 關聯到 assistant 先前發出的 tool_calls。
        ``is_error`` 不是 OpenAI 原生欄位，會以前綴形式混進 content 以告知模型。
        """
        content = f"[ERROR] {result}" if is_error else result
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )
        self._trim()

    # ------------------------------------------------------------------
    # 讀取 / 維護
    # ------------------------------------------------------------------
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        取得完整的訊息列表，用於 API 呼叫。
        system message 會自動放在最前面。
        """
        if self.system_message is not None:
            return [self.system_message, *self.messages]
        return list(self.messages)

    def _trim(self) -> None:
        """
        簡單的記憶修剪：超出 max_messages 時砍掉最舊的訊息。
        注意：直接砍最前面有可能把 tool 訊息孤立，實務上可做更細的配對修剪。
        """
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def clear(self, keep_system: bool = False) -> None:
        """清空記憶。"""
        self.messages = []
        if not keep_system:
            self.system_message = None

    def __len__(self) -> int:
        # 與 get_messages 一致：算上 system
        return len(self.messages) + (1 if self.system_message else 0)
