"""
Reasoning / Thinking 觀察模組
==============================
這個模組提供輔助函式，用於從 OpenAI 風格的 ``ChatCompletion`` 回應
中擷取 reasoning、content、tool_calls 等資訊，方便 log、除錯或分析。

為什麼要觀察 reasoning？
- 讓模型在輸出最終答案前先「想清楚」 (gpt-oss-120b 的 reasoning_effort 機制)
- 增強複雜推理任務的品質
- 使用者可以看到推理過程，方便除錯和建立信任

注意：
- 並非所有 OpenAI-compatible endpoint 都會回傳 reasoning。
  gpt-oss 透過 vLLM 時通常會有 ``message.reasoning_content``。
"""
from typing import Any


def extract_reasoning(message: Any) -> str:
    """從 assistant message 中取出 reasoning (若伺服器有回傳)。"""
    if message is None:
        return ""
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(message, attr, None)
        if val:
            return str(val)
    if isinstance(message, dict):
        for attr in ("reasoning_content", "reasoning"):
            val = message.get(attr)
            if val:
                return str(val)
    return ""


def extract_text(message: Any) -> str:
    """從 assistant message 中取出主要文字內容。"""
    if message is None:
        return ""
    if isinstance(message, dict):
        return message.get("content") or ""
    return getattr(message, "content", None) or ""


def extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    """從 assistant message 中取出 tool_calls 的輕量表示。"""
    if message is None:
        return []
    tool_calls = (
        message.get("tool_calls") if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    if not tool_calls:
        return []

    out = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            out.append({
                "id": tc.get("id"),
                "name": fn.get("name"),
                "arguments": fn.get("arguments"),
            })
        else:
            out.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            })
    return out


def summarize_response(response: Any) -> dict[str, Any]:
    """
    把一次 ChatCompletion 回應整理成結構化的 dict，方便 log / 分析。
    """
    choice = response.choices[0]
    message = choice.message
    usage = getattr(response, "usage", None)
    return {
        "finish_reason": choice.finish_reason,
        "reasoning": extract_reasoning(message),
        "text": extract_text(message),
        "tool_calls": extract_tool_calls(message),
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
    }
