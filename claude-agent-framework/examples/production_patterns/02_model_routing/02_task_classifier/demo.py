"""
Task Classifier Routing — one cheap call decides which tier to use
==================================================================

What this demo shows
--------------------
A small classifier (tier="cheap") labels each incoming query as
``simple | medium | hard``.  The full answer is then generated on:

  * ``classifier`` policy: cheap/medium/expensive per the label
  * ``always_big`` policy: always on expensive

Metrics: per-task chosen tier, latency, $ cost.  A synthetic "gold tier"
is encoded in the task list so we can also report classifier accuracy.

Why it matters: if your distribution is 70% simple / 25% medium / 5% hard
(a common shape for customer support), a well-tuned classifier collapses
the expected cost to ~15–20% of always-big, with negligible quality drop
on the 95% of queries that don't need the flagship.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import BACKEND, MODELS, banner, chat, dollars  # noqa: E402

PRICE = {
    "cheap":     {"in": 1.00, "out": 5.00},
    "medium":    {"in": 3.00, "out": 15.00},
    "expensive": {"in": 5.00, "out": 25.00},
}


TASKS: List[Tuple[str, str]] = [
    ("What timezone is Tokyo in?", "simple"),
    ("List three common HTTP status codes.", "simple"),
    ("Translate 'good morning' to Japanese.", "simple"),
    ("Explain the difference between JWT and session cookies.", "medium"),
    ("Write a Python function that deduplicates a list while preserving order.", "medium"),
    ("Summarise the CAP theorem in two paragraphs for a junior engineer.", "medium"),
    ("Prove that the halting problem is undecidable.", "hard"),
    ("Design a global rate-limiter that tolerates clock skew across regions and survives network partitions.", "hard"),
    ("Debug this Haskell type error: ``Couldn't match type ‘Maybe (IO a)’ with ‘a0 -> IO b0’``.", "hard"),
]


CLASSIFIER_PROMPT = (
    "Classify the complexity of the user request. Reply with a single word from "
    "{simple, medium, hard} only. "
    "Rules:\n"
    "- simple: factual lookup, short translation, single-step reasoning, <=20-word answer.\n"
    "- medium: explanation, short code snippet, 2-3 step reasoning.\n"
    "- hard: formal proof, system design, multi-hop reasoning, obscure debugging.\n"
)


def classify(query: str) -> str:
    r = chat(
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": query},
        ],
        tier="cheap",
        max_tokens=32,
        extra={"reasoning_effort": "low"},
    )
    m = re.search(r"\b(simple|medium|hard)\b", (r.text or "").lower())
    return m.group(1) if m else "medium"


def answer(query: str, tier: str) -> tuple[str, int, int, float]:
    r = chat(
        messages=[
            {"role": "system", "content": "Be helpful and concise."},
            {"role": "user", "content": query},
        ],
        tier=tier,
        max_tokens=512,
        extra={"reasoning_effort": {"cheap": "low", "medium": "medium", "expensive": "high"}[tier]},
    )
    return r.text, r.input_tokens, r.output_tokens, r.latency_s


TIER_OF = {"simple": "cheap", "medium": "medium", "hard": "expensive"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=len(TASKS))
    args = parser.parse_args()

    banner(f"Task-classifier routing demo  |  backend={BACKEND}")
    print(f"cheap={MODELS['cheap']}  medium={MODELS['medium']}  expensive={MODELS['expensive']}\n")

    rows = []
    correct = 0
    for query, gold_label in TASKS[: args.tasks]:
        predicted_label = classify(query)
        if predicted_label == gold_label:
            correct += 1
        chosen = TIER_OF[predicted_label]
        txt, ti, to, lat = answer(query, chosen)
        rows.append({
            "query": query[:60], "gold": gold_label, "pred": predicted_label,
            "tier": chosen, "in": ti, "out": to, "latency_s": lat,
            "cost_classifier": _cost("cheap", 80, 10),   # approximate classifier cost per call
            "cost_router": _cost(chosen, ti, to),
            "cost_always_big": _cost("expensive", ti, to),
        })
        print(
            f"[{predicted_label:<6}] gold={gold_label:<6} tier={chosen:<10} "
            f"lat={lat:5.2f}s  cost_router={dollars(rows[-1]['cost_router'])}  "
            f"query={query[:55]}"
        )

    banner("Summary")
    acc = correct / len(rows)
    total_router = sum(r["cost_router"] + r["cost_classifier"] for r in rows)
    total_big = sum(r["cost_always_big"] for r in rows)
    print(f"classifier accuracy   : {acc*100:.1f}%")
    print(f"router total cost     : {dollars(total_router)}  (answer + classifier)")
    print(f"always-big total cost : {dollars(total_big)}")
    print(f"savings               : {(1 - total_router/total_big)*100:+.1f}%")


def _cost(tier: str, tok_in: int, tok_out: int) -> float:
    p = PRICE[tier]
    return (tok_in * p["in"] + tok_out * p["out"]) / 1e6


if __name__ == "__main__":
    main()
