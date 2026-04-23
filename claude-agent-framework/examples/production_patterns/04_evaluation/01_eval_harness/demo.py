"""
Minimal Agent Eval Harness — tasks, trials, graders, pass@k / pass^k
=====================================================================

What this demo shows
--------------------
A tiny agent runs 3 tasks × k=3 trials each.  Graders are pure-Python checks
(no LLM judges — those live in demo 05_llm_judge/).

Outputs per task:
  * pass@1 — first trial passed
  * pass@k — at least one trial passed
  * pass^k — all k trials passed (Sierra's "you can rely on it" metric)
  * mean score, total tokens, total $

Why pass^k is the only metric that matters for customer-facing agents:
  if pass@1 = 0.9, then pass^5 = 0.59 — you fail 41% of users eventually.

Run
---
    python demo.py
    python demo.py --k 5
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import BACKEND, MODELS, banner, chat, dollars  # noqa: E402

PRICE_IN_PER_M = 3.00
PRICE_OUT_PER_M = 15.00


# ---------------------------------------------------------------------------
# Graders — deterministic, zero-temperature-safe.
# ---------------------------------------------------------------------------
def grade_contains(needle: str) -> Callable[[str], float]:
    def _g(text: str) -> float:
        return 1.0 if needle.lower() in (text or "").lower() else 0.0
    return _g


def grade_json_keys(keys: List[str]) -> Callable[[str], float]:
    def _g(text: str) -> float:
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not m:
            return 0.0
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return 0.0
        have = sum(1 for k in keys if k in obj)
        return have / len(keys)
    return _g


def grade_numeric(expected: int, tolerance: int = 0) -> Callable[[str], float]:
    def _g(text: str) -> float:
        for m in re.finditer(r"-?\d+", (text or "").replace(",", "")):
            v = int(m.group(0))
            if abs(v - expected) <= tolerance:
                return 1.0
        return 0.0
    return _g


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
@dataclass
class Task:
    id: str
    prompt: str
    graders: List[Callable[[str], float]]
    temperature: float = 0.2  # non-zero so pass^k meaningfully varies


TASKS: List[Task] = [
    Task("json_city",
         "Return ONLY a JSON object with keys city, population, country for Tokyo.",
         [grade_json_keys(["city", "population", "country"])]),
    Task("math_99_squared",
         "What is 99 squared? Reply with ONLY the integer.",
         [grade_numeric(9801)]),
    Task("fact_guido",
         "Which programming language was invented by Guido van Rossum? One word.",
         [grade_contains("python")]),
]


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------
@dataclass
class Trial:
    task_id: str
    trial_id: str
    score: float = 0.0
    passed: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    text: str = ""


def run_trial(task: Task) -> Trial:
    t0 = time.perf_counter()
    r = chat(
        messages=[
            {"role": "system", "content": "Follow instructions literally."},
            {"role": "user", "content": task.prompt},
        ],
        tier="cheap",
        max_tokens=300,
        temperature=task.temperature,
        extra={"reasoning_effort": "low"},
    )
    scores = [g(r.text) for g in task.graders]
    mean_score = sum(scores) / len(scores)
    return Trial(
        task_id=task.id,
        trial_id=str(uuid.uuid4())[:8],
        score=mean_score, passed=mean_score >= 0.9,
        tokens_in=r.input_tokens, tokens_out=r.output_tokens,
        latency_s=time.perf_counter() - t0,
        text=r.text[:120],
    )


def run_suite(tasks: List[Task], k: int) -> List[Dict[str, Any]]:
    results = []
    for task in tasks:
        trials = [run_trial(task) for _ in range(k)]
        scores = [t.score for t in trials]
        results.append({
            "task_id": task.id,
            "trials": [asdict(t) for t in trials],
            "pass_at_1":  scores[0] >= 0.9,
            f"pass_at_{k}": any(s >= 0.9 for s in scores),
            f"pass_hat_{k}": all(s >= 0.9 for s in scores),
            "mean_score": sum(scores) / k,
            "avg_tokens_in":  sum(t.tokens_in for t in trials) / k,
            "avg_tokens_out": sum(t.tokens_out for t in trials) / k,
            "total_cost_usd": sum(
                (t.tokens_in * PRICE_IN_PER_M + t.tokens_out * PRICE_OUT_PER_M) / 1e6
                for t in trials
            ),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    banner(f"Minimal eval harness  |  backend={BACKEND}  model={MODELS['cheap']}  k={args.k}")
    rows = run_suite(TASKS, k=args.k)

    for r in rows:
        preview = r["trials"][0]["text"]
        print(
            f"task={r['task_id']:<18}  "
            f"pass@1={r['pass_at_1']}  "
            f"pass@{args.k}={r[f'pass_at_{args.k}']}  "
            f"pass^{args.k}={r[f'pass_hat_{args.k}']}  "
            f"mean={r['mean_score']:.2f}  cost={dollars(r['total_cost_usd'])}"
        )

    banner("Aggregate")
    n = len(rows)
    suite_pass_1   = sum(1 for r in rows if r["pass_at_1"]) / n
    suite_pass_atk = sum(1 for r in rows if r[f"pass_at_{args.k}"]) / n
    suite_pass_hat = sum(1 for r in rows if r[f"pass_hat_{args.k}"]) / n
    print(f"suite pass@1  : {suite_pass_1*100:5.1f}%")
    print(f"suite pass@{args.k}  : {suite_pass_atk*100:5.1f}%  (optimistic)")
    print(f"suite pass^{args.k}  : {suite_pass_hat*100:5.1f}%  (production-relevant)")
    print(
        "\nSierra's rule of thumb: a 0.61 pass^1 compounded over 8 turns is ~25% — "
        "reliability collapses fast.  Grade what users actually experience."
    )


if __name__ == "__main__":
    main()
