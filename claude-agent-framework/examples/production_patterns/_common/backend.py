"""
Unified backend shim used by every demo under ``examples/production_patterns/``.

Why this module exists
----------------------
The article "Production AI agent engineering, beyond tutorials" shows Anthropic-
only features (``cache_control``, ``tool_use`` content blocks, extended
thinking).  In this repo, most developers only have access to an internal
OpenAI-compatible endpoint (``llm_service.py`` + ``gpt-oss-120b``).  Each demo
should still run end-to-end with ``python demo.py`` regardless of which
credentials are present.

Backend selection (per call)
----------------------------
* **chat / stream** — uses Anthropic SDK if ``ANTHROPIC_API_KEY`` is set,
  otherwise falls back to the internal ``llm_service`` (``gpt-oss-120b``
  through an OpenAI-compatible endpoint).
* **embed** — prefers ``voyageai`` (if ``VOYAGE_API_KEY``), then
  ``sentence-transformers`` (always installed in this repo), and finally a
  deterministic hash-based fallback so demos run even on air-gapped boxes.

Tier aliases
------------
Several demos need "cheap / medium / expensive" model tiers.  On Anthropic we
map to haiku / sonnet / opus.  On the internal endpoint we only have one
model, so tiers collapse to the same model with different ``reasoning_effort``
values (``low`` / ``medium`` / ``high``) — enough to illustrate the routing
patterns even when true tier diversity isn't available.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterable, Iterator, List, Optional

# Make ``llm_service`` and ``src.*`` importable when a demo is launched directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # claude-agent-framework/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Eagerly load ``.env`` so that later ``os.getenv`` calls (for ANTHROPIC_API_KEY,
# LANGFUSE_*, VOYAGE_API_KEY, etc.) see the project credentials.  We do this
# here so every demo that imports ``_common.backend`` gets a consistent env.
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except Exception:
    pass  # dotenv is in requirements.txt, but absence should not crash imports


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _has_anthropic() -> bool:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except Exception:
        return False


BACKEND: str = "anthropic" if _has_anthropic() else "llm_service"


# Model tier aliases. Override per-demo with env vars when you want a specific
# model on Anthropic, e.g. ANTHROPIC_MODEL_MEDIUM=claude-3-5-sonnet-latest.
if BACKEND == "anthropic":
    MODELS: Dict[str, str] = {
        "cheap":     os.getenv("ANTHROPIC_MODEL_CHEAP",  "claude-haiku-4-5"),
        "medium":    os.getenv("ANTHROPIC_MODEL_MEDIUM", "claude-sonnet-4-5"),
        "expensive": os.getenv("ANTHROPIC_MODEL_BIG",    "claude-opus-4-5"),
    }
else:
    # Internal endpoint: single model; tiers are simulated via reasoning_effort.
    _llm_model = os.getenv("LLM_MODEL", "gpt-oss-120b")
    MODELS = {"cheap": _llm_model, "medium": _llm_model, "expensive": _llm_model}


# Pricing for illustrative $ estimates — USD per 1M tokens.
# Numbers match the article's April-2026 snapshot for Anthropic; the internal
# endpoint is billed differently, so we report 0 and focus on token deltas.
PRICING: Dict[str, Dict[str, float]] = {
    "claude-haiku-4-5":   {"in": 1.00, "out": 5.00,  "cache_read": 0.10, "cache_write": 1.25},
    "claude-sonnet-4-5":  {"in": 3.00, "out": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "claude-opus-4-5":    {"in": 5.00, "out": 25.00, "cache_read": 0.50, "cache_write": 6.25},
    "gpt-oss-120b":       {"in": 0.0,  "out": 0.0},
}


# ---------------------------------------------------------------------------
# Normalized result type
# ---------------------------------------------------------------------------
@dataclass
class ChatResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str = ""
    # Normalised tool calls the demo can iterate over. Each dict is:
    #   {"id": str, "name": str, "arguments": dict}
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # Raw SDK response for demos that need provider-specific fields.
    raw: Any = None

    @property
    def cost_usd(self) -> float:
        return price_usd(self.model, self)


def price_usd(model: str, usage: Any) -> float:
    """Return $ cost estimate for a ChatResult / dict-like usage object."""
    p = PRICING.get(model, {"in": 0.0, "out": 0.0})
    def _g(key: str) -> int:
        if hasattr(usage, key):
            return int(getattr(usage, key) or 0)
        if isinstance(usage, dict):
            return int(usage.get(key) or 0)
        return 0
    cost = _g("input_tokens") * p.get("in", 0.0) / 1e6
    cost += _g("output_tokens") * p.get("out", 0.0) / 1e6
    cost += _g("cache_read_tokens") * p.get("cache_read", p.get("in", 0.0)) / 1e6
    cost += _g("cache_create_tokens") * p.get("cache_write", p.get("in", 0.0)) / 1e6
    return cost


# ---------------------------------------------------------------------------
# Tokenizer helper (approximate)
# ---------------------------------------------------------------------------
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken is in requirements.txt
    _ENC = None


def count_tokens(text: str | List[Dict[str, Any]]) -> int:
    """Approximate token count. Good enough for demo-level cost math."""
    if isinstance(text, list):
        text = json.dumps(text, ensure_ascii=False)
    if _ENC is not None:
        return len(_ENC.encode(text))
    # Rough fallback: 4 characters per token on average English / CJK mix.
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Tool-format translation
# ---------------------------------------------------------------------------
def _tools_to_anthropic(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Accept OpenAI-style or Anthropic-style tool dicts and emit Anthropic."""
    out = []
    for t in tools:
        if "input_schema" in t:                # already Anthropic
            out.append(t)
        elif t.get("type") == "function" and "function" in t:
            fn = t["function"]
            out.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        else:                                   # assume simple {name, description, parameters}
            out.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", t.get("input_schema", {"type": "object", "properties": {}})),
            })
    return out


def _tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Accept OpenAI-style or Anthropic-style tool dicts and emit OpenAI."""
    out = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            out.append(t)
        elif "input_schema" in t:               # Anthropic → OpenAI
            out.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["input_schema"],
                },
            })
        else:
            out.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            })
    return out


# ---------------------------------------------------------------------------
# Message-format translation
# ---------------------------------------------------------------------------
def _split_system(messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    """Anthropic takes ``system`` as a top-level arg; split it out."""
    sys_parts: List[str] = []
    rest: List[Dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system":
            content = m["content"]
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        sys_parts.append(c.get("text", ""))
                    elif isinstance(c, str):
                        sys_parts.append(c)
            else:
                sys_parts.append(str(content))
        else:
            rest.append(m)
    return "\n\n".join(sys_parts), rest


def _reasoning_tier_to_effort(tier: str) -> str:
    return {"cheap": "low", "medium": "medium", "expensive": "high"}.get(tier, "medium")


# ---------------------------------------------------------------------------
# Public API: chat, chat_async, stream, stream_async
# ---------------------------------------------------------------------------
def chat(
    messages: List[Dict[str, Any]],
    *,
    tier: str = "medium",
    model: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Any = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    system_cache: bool = False,
    response_format: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> ChatResult:
    """Single-shot chat completion. Normalised across backends.

    Args:
        messages: OpenAI-style ``[{role, content}]`` list.
        tier: "cheap" | "medium" | "expensive" — mapped to a real model.
        model: Explicit model name that overrides ``tier``.
        tools: Tool list in OpenAI or Anthropic format.
        tool_choice: Provider-specific.
        system_cache: When True and backend is Anthropic, place a
            ``cache_control: ephemeral`` breakpoint on the system prompt.
        response_format: Passed to OpenAI Chat Completions (e.g. Pydantic-
            compatible ``{"type": "json_schema", ...}``). Ignored on Anthropic.
    """
    model = model or MODELS.get(tier, MODELS["medium"])
    t0 = time.perf_counter()

    if BACKEND == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        system_text, rest = _split_system(messages)
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": rest,
        }
        if system_text:
            if system_cache:
                kwargs["system"] = [{
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                kwargs["system"] = system_text
        if tools:
            kwargs["tools"] = _tools_to_anthropic(tools)
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        if extra:
            kwargs.update(extra)
        resp = client.messages.create(**kwargs)
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "arguments": block.input})
        u = resp.usage
        return ChatResult(
            text="".join(text_parts),
            model=model,
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_create_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            latency_s=time.perf_counter() - t0,
            finish_reason=resp.stop_reason or "",
            tool_calls=tool_calls,
            raw=resp,
        )

    # ---- llm_service backend ----
    from llm_service import llm_service
    openai_tools = _tools_to_openai(tools) if tools else None
    tc = tool_choice
    if openai_tools and tc is None:
        tc = "auto"
    reasoning_effort = (extra or {}).get("reasoning_effort") or _reasoning_tier_to_effort(tier)
    kwargs = {
        "messages": messages,
        "tools": openai_tools,
        "tool_choice": tc,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
    }
    # OpenAI Chat Completions supports response_format via the client directly
    # — route through llm_service.client when it's specified so we keep the
    # strict-schema features the article demos.
    if response_format is not None:
        c = llm_service.client
        api_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": response_format,
        }
        if openai_tools:
            api_kwargs["tools"] = openai_tools
            api_kwargs["tool_choice"] = tc
        completion = c.chat.completions.create(**api_kwargs)
    else:
        completion = llm_service.complete_with_tools(**kwargs)

    choice = completion.choices[0]
    
    msg = choice.message

    tool_calls = []
    for tc_obj in (getattr(msg, "tool_calls", None) or []):
        try:
            args = json.loads(tc_obj.function.arguments or "{}")
        except Exception:
            args = {}
        tool_calls.append({"id": tc_obj.id, "name": tc_obj.function.name, "arguments": args})
    usage = completion.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return ChatResult(
        text=msg.content or "",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cached,
        cache_create_tokens=0,
        latency_s=time.perf_counter() - t0,
        finish_reason=choice.finish_reason or "",
        tool_calls=tool_calls,
        raw=completion,
    )


async def chat_async(*args, **kwargs) -> ChatResult:
    """Thin asyncio wrapper for demos that want to run calls in parallel."""
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: chat(*args, **kwargs))


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
def stream(
    messages: List[Dict[str, Any]],
    *,
    tier: str = "medium",
    model: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Iterator[Dict[str, Any]]:
    """Yield ``{"type": "text"|"done", ...}`` dicts with deltas + final usage.

    Each text event has ``{"type": "text", "delta": str}``.
    The final event is ``{"type": "done", "result": ChatResult}``.
    """
    model = model or MODELS.get(tier, MODELS["medium"])
    t0 = time.perf_counter()

    if BACKEND == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        system_text, rest = _split_system(messages)
        buf: List[str] = []
        usage_in = usage_out = cache_read = cache_create = 0
        finish = ""
        kwargs: Dict[str, Any] = {
            "model": model, "max_tokens": max_tokens, "temperature": temperature,
            "messages": rest,
        }
        if system_text:
            kwargs["system"] = system_text
        with client.messages.stream(**kwargs) as s:
            for ev in s:
                if getattr(ev, "type", "") == "content_block_delta" and getattr(ev.delta, "type", "") == "text_delta":
                    piece = ev.delta.text
                    buf.append(piece)
                    yield {"type": "text", "delta": piece}
            final = s.get_final_message()
            finish = final.stop_reason or ""
            usage_in = getattr(final.usage, "input_tokens", 0) or 0
            usage_out = getattr(final.usage, "output_tokens", 0) or 0
            cache_read = getattr(final.usage, "cache_read_input_tokens", 0) or 0
            cache_create = getattr(final.usage, "cache_creation_input_tokens", 0) or 0
        yield {"type": "done", "result": ChatResult(
            text="".join(buf), model=model,
            input_tokens=usage_in, output_tokens=usage_out,
            cache_read_tokens=cache_read, cache_create_tokens=cache_create,
            latency_s=time.perf_counter() - t0,
            finish_reason=finish, raw=None,
        )}
        return

    # ---- llm_service stream via OpenAI-compatible SSE ----
    from llm_service import llm_service
    c = llm_service.client
    resp = c.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=temperature, stream=True, stream_options={"include_usage": True},
    )
    buf: List[str] = []
    usage_in = usage_out = cached = 0
    finish = ""
    for chunk in resp:
        if not chunk.choices:
            if chunk.usage is not None:
                usage_in = getattr(chunk.usage, "prompt_tokens", 0) or 0
                usage_out = getattr(chunk.usage, "completion_tokens", 0) or 0
                details = getattr(chunk.usage, "prompt_tokens_details", None)
                if details is not None:
                    cached = getattr(details, "cached_tokens", 0) or 0
            continue
        ch = chunk.choices[0]
        delta = ch.delta
        if getattr(delta, "content", None):
            buf.append(delta.content)
            yield {"type": "text", "delta": delta.content}
        if ch.finish_reason:
            finish = ch.finish_reason
    yield {"type": "done", "result": ChatResult(
        text="".join(buf), model=model,
        input_tokens=usage_in, output_tokens=usage_out,
        cache_read_tokens=cached, cache_create_tokens=0,
        latency_s=time.perf_counter() - t0,
        finish_reason=finish, raw=None,
    )}


async def stream_async(*args, **kwargs) -> AsyncIterator[Dict[str, Any]]:
    """Async-iterator wrapper for ``stream`` — yields the same events."""
    import asyncio
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _pump():
        try:
            for ev in stream(*args, **kwargs):
                asyncio.run_coroutine_threadsafe(q.put(ev), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    loop.run_in_executor(None, _pump)
    while True:
        ev = await q.get()
        if ev is None:
            return
        yield ev


# ---------------------------------------------------------------------------
# Embeddings (with progressive fallback)
# ---------------------------------------------------------------------------
_st_model = None


def _hash_embed(text: str, dim: int = 384) -> List[float]:
    """Deterministic hash-based fallback — produces unit-length ``dim`` vector.

    Good enough for demo k-NN when no embedding provider is configured.
    """
    buckets = [0.0] * dim
    for i, tok in enumerate(text.split()):
        h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest()[:8], 16)
        buckets[h % dim] += 1.0
    n = math.sqrt(sum(x * x for x in buckets)) or 1.0
    return [x / n for x in buckets]


def embed(text: str) -> List[float]:
    """Embed ``text`` using the best provider available, falling back gracefully.

    Order: ``voyageai`` → ``sentence-transformers`` → hash-based stub.
    """
    # 1) Voyage
    if os.getenv("VOYAGE_API_KEY"):
        try:
            import voyageai
            vo = voyageai.Client()
            return vo.embed([text], model="voyage-3-lite").embeddings[0]
        except Exception:
            pass
    # 2) sentence-transformers (installed in repo requirements)
    global _st_model
    try:
        if _st_model is None:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vec = _st_model.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()
    except Exception:
        pass
    # 3) Hash-based stub
    return _hash_embed(text)


# ---------------------------------------------------------------------------
# Pretty-printing helpers used by most demos
# ---------------------------------------------------------------------------
def banner(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def dollars(cost: float) -> str:
    """Format a $ value with 4–6 decimals depending on magnitude."""
    if cost >= 1:
        return f"${cost:.2f}"
    if cost >= 0.01:
        return f"${cost:.4f}"
    return f"${cost:.6f}"
