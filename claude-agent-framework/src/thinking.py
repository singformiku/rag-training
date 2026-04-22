"""
Extended Thinking 模組
======================
這個模組主要示範如何使用 Claude 的 Extended Thinking 功能。

實際上 thinking 是透過 API 參數開啟的 (見 agent.py 中的 _call_claude)，
這裡提供幾個輔助函式用於觀察和分析 thinking 內容。

為什麼要有 Extended Thinking？
- 讓模型在輸出最終答案前先「想清楚」
- 增強複雜推理任務的品質
- 使用者可以看到推理過程，方便除錯和建立信任
"""
from typing import Any


def extract_thinking(content_blocks: list) -> list[str]:
    """從 API 回應中抽出所有 thinking block 的內容。"""
    return [b.thinking for b in content_blocks if b.type == "thinking"]


def extract_text(content_blocks: list) -> list[str]:
    """從 API 回應中抽出所有 text block 的內容。"""
    return [b.text for b in content_blocks if b.type == "text"]


def extract_tool_uses(content_blocks: list) -> list[dict[str, Any]]:
    """從 API 回應中抽出所有工具呼叫。"""
    return [
        {"id": b.id, "name": b.name, "input": b.input}
        for b in content_blocks
        if b.type == "tool_use"
    ]


def summarize_response(response: Any) -> dict[str, Any]:
    """把一次 API 回應整理成結構化的 dict，方便 log 或分析。"""
    return {
        "stop_reason": response.stop_reason,
        "thinking": extract_thinking(response.content),
        "text": extract_text(response.content),
        "tool_uses": extract_tool_uses(response.content),
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }
