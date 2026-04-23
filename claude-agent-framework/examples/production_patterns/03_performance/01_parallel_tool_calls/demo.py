"""
Parallel Tool Calls — run independent tool invocations concurrently
===================================================================

What this demo shows
--------------------
Every tool in this demo (``fetch_weather``, ``search_docs``, ``lookup_user``)
has an artificial 1-second IO-bound sleep so the difference between
sequential and concurrent execution is large and measurable.

We prompt the agent with a question that REQUIRES calling all three tools.
On the first assistant turn we collect the tool calls emitted by the model
and execute them in two ways:

  * ``sequential`` — ``await tool()`` one at a time
  * ``parallel``   — ``asyncio.gather(*[...])``

Wall-clock latency is reported for each.  On most providers the parallel
path is ~N× faster where N is the number of tool calls.

Why it matters
--------------
The article reports 3-5× end-to-end latency reductions from parallel tool use.
When the LLM decides which tools to call, the agent harness controls HOW fast
they execute — parallelism is a single-line win.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import BACKEND, MODELS, banner, chat_async  # noqa: E402

TOOL_SLEEP_S = 1.0  # simulate remote IO latency


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
async def fetch_weather(city: str) -> str:
    await asyncio.sleep(TOOL_SLEEP_S)
    return f"Weather in {city}: 18C, partly cloudy."


async def search_docs(query: str) -> str:
    await asyncio.sleep(TOOL_SLEEP_S)
    return f"Top hit for '{query}': /runbooks/incident-response.md"


async def lookup_user(user_id: str) -> str:
    await asyncio.sleep(TOOL_SLEEP_S)
    return f"User {user_id}: alice@example.com, role=SRE"


TOOL_IMPL = {
    "fetch_weather": fetch_weather,
    "search_docs":   search_docs,
    "lookup_user":   lookup_user,
}


TOOLS = [
    {
        "name": "fetch_weather",
        "description": "Return current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
    {
        "name": "search_docs",
        "description": "Search the internal docs for a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "lookup_user",
        "description": "Look up a user record by id.",
        "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]},
    },
]


PROMPT = (
    "I'm paging oncall for user u_77. Please do all of these for me:\n"
    "1) get the weather in Tokyo\n"
    "2) find the incident response runbook\n"
    "3) look up the oncall contact info for user id u_77.\n"
    "Call all three tools at once."
)


async def get_tool_calls() -> List[Dict[str, Any]]:
    """Ask the LLM once and collect the tool_calls it emits."""
    r = await chat_async(
        messages=[
            {"role": "system", "content": "Use the provided tools to gather information in one turn."},
            {"role": "user", "content": PROMPT},
        ],
        tier="medium",
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=600,
        extra={"reasoning_effort": "low"},
    )
    return r.tool_calls


async def run_sequential(calls: List[Dict[str, Any]]) -> List[str]:
    out = []
    for c in calls:
        out.append(await TOOL_IMPL[c["name"]](**c["arguments"]))
    return out


async def run_parallel(calls: List[Dict[str, Any]]) -> List[str]:
    return await asyncio.gather(*(TOOL_IMPL[c["name"]](**c["arguments"]) for c in calls))


async def _main() -> None:
    global TOOL_SLEEP_S
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=float, default=TOOL_SLEEP_S,
                        help="Simulated tool IO latency (seconds)")
    args = parser.parse_args()
    TOOL_SLEEP_S = args.sleep

    banner(f"Parallel tool calls demo  |  backend={BACKEND}  model={MODELS['medium']}")
    print(f"Tool IO simulated at {args.sleep}s each.\n")

    calls = await get_tool_calls()
    if len(calls) < 2:
        # Fallback: not every model emits multiple tool_calls in one turn.
        # Pre-seed canonical calls so the performance comparison still runs.
        print(f"Note: model emitted {len(calls)} tool_call(s); "
              "seeding 3 canonical calls so the parallelism speedup is measurable.\n")
        calls = [
            {"id": "1", "name": "fetch_weather", "arguments": {"city": "Tokyo"}},
            {"id": "2", "name": "search_docs",   "arguments": {"query": "incident response runbook"}},
            {"id": "3", "name": "lookup_user",   "arguments": {"user_id": "u_77"}},
        ]
    print(f"model requested {len(calls)} tool calls:")
    for c in calls:
        print(f"  - {c['name']}({c['arguments']})")
    print()

    t0 = time.perf_counter()
    results_seq = await run_sequential(calls)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    results_par = await run_parallel(calls)
    t_par = time.perf_counter() - t0

    banner("Results")
    print(f"sequential latency: {t_seq:.2f}s  ({len(calls)} tools × {args.sleep}s ≈ ideal)")
    print(f"parallel latency  : {t_par:.2f}s")
    print(f"speedup            : {t_seq / max(t_par, 1e-6):.2f}x")
    assert results_seq == results_par, "results should be identical"
    print("\nparallel tool-call output (same content as sequential):")
    for r in results_par:
        print(f"  - {r}")


if __name__ == "__main__":
    asyncio.run(_main())
