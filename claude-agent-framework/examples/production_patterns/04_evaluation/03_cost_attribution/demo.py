"""
Cost Attribution — per-user, per-task $ buckets via a context manager
=====================================================================

What this demo shows
--------------------
A small multi-tenant workload: 3 users submit 2 tasks each.  Every LLM call
is recorded into a contextvars-backed ``CostBucket`` keyed by ``user_id +
task_id``.  At the end we print a breakdown by user, by model, and by tool.

In production you'd push these buckets into Redis or your tracing backend.
Here we keep it in-process so the demo is dependency-free and reproducible.

Why it matters
--------------
Teams routinely find 30–50% of agent spend going to a handful of "power users"
or a single verbose tool (a search tool returning 20 KB of unsummarised HTML
is the classic example).  Attribution makes those fixable.
"""
from __future__ import annotations

import argparse
import contextvars
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import BACKEND, MODELS, banner, chat, dollars  # noqa: E402


PRICING = {
    "claude-haiku-4-5":   {"in": 1.00, "out": 5.00},
    "claude-sonnet-4-5":  {"in": 3.00, "out": 15.00},
    "claude-opus-4-5":    {"in": 5.00, "out": 25.00},
    # Internal endpoint — priced at Sonnet-equivalent for illustration only.
    "gpt-oss-120b":       {"in": 3.00, "out": 15.00},
}


@dataclass
class CostBucket:
    user_id: str
    task_id: str
    by_model: Dict[str, float] = field(default_factory=dict)
    by_tool: Dict[str, float] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    total_usd: float = 0.0


_ctx: contextvars.ContextVar[CostBucket | None] = contextvars.ContextVar("cost", default=None)
_lock = threading.Lock()


@contextmanager
def track_cost(user_id: str, task_id: str) -> Iterator[CostBucket]:
    bucket = CostBucket(user_id=user_id, task_id=task_id)
    token = _ctx.set(bucket)
    try:
        yield bucket
    finally:
        _ctx.reset(token)


def _record(model: str, tool: str | None, tokens_in: int, tokens_out: int) -> None:
    b = _ctx.get()
    if b is None:
        return
    p = PRICING.get(model, {"in": 3.00, "out": 15.00})
    cost = (tokens_in * p["in"] + tokens_out * p["out"]) / 1e6
    with _lock:
        b.tokens_in += tokens_in
        b.tokens_out += tokens_out
        b.total_usd += cost
        b.by_model[model] = b.by_model.get(model, 0.0) + cost
        if tool:
            b.by_tool[tool] = b.by_tool.get(tool, 0.0) + cost


def tracked_chat(user_query: str, *, tier: str = "cheap", tool: str | None = None) -> str:
    r = chat(
        messages=[
            {"role": "system", "content": "Be concise. Answer in one sentence."},
            {"role": "user", "content": user_query},
        ],
        tier=tier,
        max_tokens=200,
        extra={"reasoning_effort": "low"},
    )
    _record(r.model, tool, r.input_tokens, r.output_tokens)
    return r.text


WORKLOAD: List[Dict[str, str]] = [
    {"user": "alice",   "task": "q1", "prompt": "What is the capital of Japan?"},
    {"user": "alice",   "task": "q2", "prompt": "Explain garbage collection in 1 sentence."},
    {"user": "bob",     "task": "q1", "prompt": "Who wrote The Odyssey?"},
    {"user": "bob",     "task": "q2", "prompt": "Define 'idempotent' in computing."},
    {"user": "charlie", "task": "q1", "prompt": "Summarise TCP congestion control in 1 sentence."},
    {"user": "charlie", "task": "q2", "prompt": "What does CAP stand for?"},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=len(WORKLOAD))
    args = parser.parse_args()

    banner(f"Cost attribution demo  |  backend={BACKEND}  model={MODELS['cheap']}")

    buckets: List[CostBucket] = []
    for row in WORKLOAD[: args.tasks]:
        with track_cost(user_id=row["user"], task_id=row["task"]) as b:
            answer = tracked_chat(row["prompt"], tool="chat")
            buckets.append(b)
            print(f"user={row['user']:<7} task={row['task']}  cost={dollars(b.total_usd)}  "
                  f"answer={answer[:70]}")

    banner("By user")
    per_user: Dict[str, float] = {}
    for b in buckets:
        per_user[b.user_id] = per_user.get(b.user_id, 0) + b.total_usd
    for user, c in sorted(per_user.items(), key=lambda x: -x[1]):
        print(f"  {user:<10}  {dollars(c)}")

    banner("By model")
    per_model: Dict[str, float] = {}
    for b in buckets:
        for m, c in b.by_model.items():
            per_model[m] = per_model.get(m, 0.0) + c
    for m, c in sorted(per_model.items(), key=lambda x: -x[1]):
        print(f"  {m:<20}  {dollars(c)}")

    total = sum(per_user.values())
    print(f"\ntotal spend: {dollars(total)}  ({sum(b.tokens_in for b in buckets)} in, "
          f"{sum(b.tokens_out for b in buckets)} out)")


if __name__ == "__main__":
    main()
