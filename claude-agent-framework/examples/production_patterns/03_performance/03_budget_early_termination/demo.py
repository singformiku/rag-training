"""
Budget Caps & Early Termination — stop runaway agents
======================================================

What this demo shows
--------------------
A trivial "loop-happy" agent is given a tool that ALWAYS returns
``need_more=True``, tempting it to keep calling forever.  We run it under
four increasingly strict budgets:

  * ``no_budget``      — only the safety rail of 50 iterations (our sanity cap)
  * ``iter_cap=5``     — at most 5 LLM calls
  * ``wall_cap=8s``    — at most 8 seconds of wall-clock time
  * ``token_cap=4000`` — at most 4k cumulative input tokens

For each run we report iterations, wall time, total tokens, final outcome,
and the $ cost against the Sonnet reference price.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import BACKEND, MODELS, banner, chat, dollars  # noqa: E402

PRICE_IN_PER_M = 3.00
PRICE_OUT_PER_M = 15.00


@dataclass
class Budget:
    max_iters: int = 50
    max_wall_s: float = 120.0
    max_input_tokens: int = 10**9
    max_tool_calls: int = 10**9


class BudgetExceeded(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


TOOLS = [{
    "name": "look_up_fact",
    "description": "Always returns a tiny fact + a hint to keep searching.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}]


def fake_tool(query: str) -> str:
    return json.dumps({
        "fact": f"Fact about '{query}': it is related to many things.",
        "need_more": True,
        "hint": "try another related query",
    })


PROMPT = (
    "Use the look_up_fact tool to research everything you can about 'entropy'. "
    "Keep calling the tool until you are satisfied, THEN answer in one paragraph. "
    "The tool may keep suggesting new queries."
)


def run(budget: Budget) -> Dict[str, Any]:
    """A tiny OpenAI-style agent loop with hard budgets."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "Gather information with the tool, then give a final answer."},
        {"role": "user", "content": PROMPT},
    ]
    iters = tool_calls = in_tokens = out_tokens = 0
    started = time.perf_counter()
    reason = "stop"
    final_text = ""
    try:
        for _ in range(budget.max_iters):
            if time.perf_counter() - started >= budget.max_wall_s:
                raise BudgetExceeded("wall")
            if in_tokens >= budget.max_input_tokens:
                raise BudgetExceeded("tokens")
            if tool_calls >= budget.max_tool_calls:
                raise BudgetExceeded("tool_calls")

            r = chat(
                messages=messages,
                tier="medium",
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=300,
                extra={"reasoning_effort": "low"},
            )
            iters += 1
            in_tokens += r.input_tokens
            out_tokens += r.output_tokens

            if r.tool_calls:
                # Append assistant message that carries the tool_calls, then
                # run each tool + append the tool result in a shape
                # ``llm_service.complete_with_tools`` can consume next turn.
                messages.append({
                    "role": "assistant",
                    "content": r.text or "",
                    "tool_calls": [
                        {"id": tc["id"], "type": "function",
                         "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                        for tc in r.tool_calls
                    ],
                })
                for tc in r.tool_calls:
                    tool_calls += 1
                    output = fake_tool(**tc["arguments"])
                    messages.append({
                        "role": "tool", "tool_call_id": tc["id"], "content": output,
                    })
                continue

            final_text = r.text or ""
            break
        else:
            reason = "iter_cap"
    except BudgetExceeded as e:
        reason = f"budget:{e.reason}"

    wall = time.perf_counter() - started
    cost = (in_tokens * PRICE_IN_PER_M + out_tokens * PRICE_OUT_PER_M) / 1e6
    return {
        "iters": iters, "tool_calls": tool_calls,
        "in_tokens": in_tokens, "out_tokens": out_tokens,
        "wall_s": wall, "stop_reason": reason,
        "cost_usd": cost, "final_text": final_text[:140],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    banner(f"Budget & early termination demo  |  backend={BACKEND}  model={MODELS['medium']}")
    budgets = [
        ("no_budget",      Budget(max_iters=20)),
        ("iter_cap=5",     Budget(max_iters=5)),
        ("wall_cap=8s",    Budget(max_iters=20, max_wall_s=8.0)),
        ("token_cap=4000", Budget(max_iters=20, max_input_tokens=4000)),
    ]
    all_results = []
    for name, b in budgets:
        print(f"\n>>> scenario: {name}")
        r = run(b)
        all_results.append((name, r))
        print(
            f"    iters={r['iters']:<2} tool_calls={r['tool_calls']:<2} "
            f"in_tok={r['in_tokens']:<5} out_tok={r['out_tokens']:<4} "
            f"wall={r['wall_s']:5.2f}s  stop={r['stop_reason']:<15}  "
            f"cost={dollars(r['cost_usd'])}"
        )
        if r["final_text"]:
            print(f"    final_text: {r['final_text']}")

    banner("Summary")
    print("budgets prevent runaway loops from burning $$ AND sanity.")
    worst = max(all_results, key=lambda x: x[1]["cost_usd"])
    best_capped = min(
        (x for x in all_results if x[0] != "no_budget"),
        key=lambda x: x[1]["cost_usd"],
    )
    print(f"most expensive run : {worst[0]}  — {dollars(worst[1]['cost_usd'])}")
    print(f"cheapest capped run: {best_capped[0]} — {dollars(best_capped[1]['cost_usd'])}")


if __name__ == "__main__":
    main()
