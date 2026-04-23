"""
Langfuse Tracing — nested spans for the agent, its LLM calls, and tools
=======================================================================

What this demo shows
--------------------
A small 2-step agent (plan → search → summarise) is instrumented with the
Langfuse v4 OTel-native decorator API.  Every LLM call + tool call becomes a
span with cost, latency, token usage, and user/session tags.

No Langfuse account?  The demo still runs end-to-end — it skips the flush and
prints what WOULD be sent.  To see the traces:
  1. ``export LANGFUSE_HOST=http://localhost:3000``
     ``export LANGFUSE_PUBLIC_KEY=...``
     ``export LANGFUSE_SECRET_KEY=...``
  2. ``python demo.py``
  3. open the Langfuse UI and inspect the ``agent_run`` trace.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Importing the backend triggers the repo .env to be loaded (via src.config).
# Do this BEFORE inspecting LANGFUSE_* vars so the demo picks them up.
from _common.backend import BACKEND, MODELS, banner, chat  # noqa: E402

LF_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _null_decorator(*_args, **_kwargs):
    def _inner(fn):
        return fn
    return _inner


if LF_ENABLED:
    try:
        from langfuse import observe, get_client
        lf = get_client()
    except Exception as e:
        print(f"[warn] Langfuse import failed: {e}")
        LF_ENABLED = False

if not LF_ENABLED:
    observe = _null_decorator  # type: ignore
    lf = None  # type: ignore


@observe(name="tool.search_docs", as_type="tool")
def search_docs(query: str) -> str:
    # Deterministic mock result.
    return f"Top hit for '{query}': /docs/sre/runbook.md (excerpt)."


@observe(name="llm.plan", as_type="generation")
def plan(goal: str) -> str:
    r = chat(
        messages=[
            {"role": "system", "content": "Write a 3-step plan as a numbered list."},
            {"role": "user", "content": goal},
        ],
        tier="cheap",
        max_tokens=200,
        extra={"reasoning_effort": "low"},
    )
    return r.text


@observe(name="llm.summarize", as_type="generation")
def summarize(text: str) -> str:
    r = chat(
        messages=[
            {"role": "system", "content": "Summarise in 2 sentences."},
            {"role": "user", "content": text[:4000]},
        ],
        tier="cheap",
        max_tokens=200,
        extra={"reasoning_effort": "low"},
    )
    return r.text


@observe(name="agent.run", as_type="agent")
def agent_run(goal: str, user_id: str, session_id: str) -> str:
    if LF_ENABLED:
        # Newer SDKs expose propagate_attributes; fall back silently if missing.
        try:
            from langfuse import propagate_attributes
            ctx = propagate_attributes(
                user_id=user_id,
                session_id=session_id,
                tags=["demo", "production_patterns", "02_langfuse"],
                metadata={"backend": BACKEND, "model": MODELS["cheap"]},
            )
        except Exception:
            from contextlib import nullcontext
            ctx = nullcontext()
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        steps = plan(goal)
        evidence = search_docs(goal)
        return summarize(f"Plan:\n{steps}\n\nEvidence:\n{evidence}")


def main() -> None:
    banner(f"Langfuse tracing demo  |  backend={BACKEND}  langfuse_enabled={LF_ENABLED}")
    if not LF_ENABLED:
        print("LANGFUSE_* env vars not set; traces will NOT be flushed.")
        print("Everything else still runs so you can inspect the structure.\n")

    result = agent_run(
        goal="Triage a p99 latency spike on /api/checkout",
        user_id="u_42", session_id="t_abc",
    )
    print("\n--- agent output ---")
    print(result)
    if LF_ENABLED:
        lf.flush()
        print("\nSpans flushed to Langfuse — open your project to inspect the trace.")


if __name__ == "__main__":
    main()
