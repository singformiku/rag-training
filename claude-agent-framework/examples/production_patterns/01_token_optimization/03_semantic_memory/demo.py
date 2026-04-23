"""
Semantic Memory — k-NN over past turns beats naive full history
===============================================================

What this demo shows
--------------------
We seed a "conversation" with many turns covering different topics, then ask
questions that need information from a single earlier turn (the "needle").

For each question we compare three retrieval policies:
  * ``full_history``    — send every prior turn every time (token-heavy)
  * ``rolling_window``  — last N turns only (cheap but lossy)
  * ``semantic``        — embed each turn, retrieve top-k by cosine + recency

Metrics reported: needle-recall (did the needle make it into the context?),
context tokens, and a projected cost against the Claude Sonnet $/MTok baseline.

No external vector DB is required — we keep everything in-process.  Embeddings
use whichever provider the backend picks (Voyage → sentence-transformers →
hash-based fallback), see ``_common.backend.embed``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import banner, count_tokens, embed  # noqa: E402

PRICE_IN_PER_M = 3.00
PRICE_OUT_PER_M = 15.00


@dataclass
class Turn:
    idx: int
    role: str
    content: str
    ts: float
    vector: List[float] = field(default_factory=list)


def _cos(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0
    return num / den


def build_synthetic_history(n_turns: int) -> Tuple[List[Turn], List[Tuple[str, str]]]:
    """Return ``(history, needles)``.

    Each needle is ``(question, substring_that_must_appear_in_retrieved_context)``.
    """
    topics = [
        ("payments", "The staging DB password rotates every 30 days."),
        ("frontend", "The feature flag `checkout_v2` is canary at 5%."),
        ("oncall",   "Incident IC for October is alice@example.com."),
        ("release",  "Release tag v4.12.7 introduced the retry-with-jitter patch."),
        ("infra",    "Grafana dashboard `svc-gateway` panels are read from Mimir."),
    ]
    history: List[Turn] = []
    t0 = time.time()
    # Plant each needle once at known indices, then fill with noise.
    for i, (topic, needle) in enumerate(topics):
        idx = i * (n_turns // (len(topics) + 1)) + 3
        history.append(Turn(idx, "user", f"Reminder about {topic}.", t0 - (n_turns - idx) * 60))
        history.append(Turn(idx + 1, "assistant", needle, t0 - (n_turns - idx) * 60 + 1))
    planted_idx = {t.idx for t in history}
    for i in range(n_turns):
        if i in planted_idx:
            continue
        noise = f"Small talk #{i}: nothing important happened here."
        history.append(Turn(i, "user" if i % 2 == 0 else "assistant", noise, t0 - (n_turns - i) * 60))
    history.sort(key=lambda t: t.idx)
    needles = [
        ("What is the rotation period of the staging DB password?", "30 days"),
        ("Who is the October oncall IC?",                            "alice@example.com"),
        ("Which release introduced retry-with-jitter?",              "v4.12.7"),
        ("Where do svc-gateway panels source their data?",           "Mimir"),
        ("What's the canary percent for checkout_v2?",               "5%"),
    ]
    return history, needles


def embed_history(history: List[Turn]) -> None:
    print(f"Embedding {len(history)} turns (first call may download a model) ...")
    for t in history:
        text = f"[{t.role}] {t.content}"
        t.vector = embed(text)


def retrieve_semantic(query: str, history: List[Turn], k: int, alpha: float = 0.8) -> List[Turn]:
    """Top-k by cosine similarity with a mild recency boost."""
    q_vec = embed(query)
    now = max(t.ts for t in history)
    scored = []
    for t in history:
        cos = _cos(q_vec, t.vector)
        decay = 1.0 / (1.0 + (now - t.ts) / 3600.0)
        scored.append((alpha * cos + (1 - alpha) * decay, t))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:k]]


def evaluate(strategy: str, history: List[Turn], needles, *, k: int, window: int) -> Dict[str, float]:
    recall_hits = 0
    context_tokens_total = 0
    for question, must_include in needles:
        if strategy == "full_history":
            context = history
        elif strategy == "rolling_window":
            context = history[-window:]
        elif strategy == "semantic":
            context = retrieve_semantic(question, history, k=k)
        else:
            raise ValueError(strategy)
        text = "\n".join(f"{t.role}: {t.content}" for t in context)
        if must_include.lower() in text.lower():
            recall_hits += 1
        context_tokens_total += count_tokens(text)
    n = len(needles)
    return {
        "strategy": strategy,
        "recall": recall_hits / n,
        "avg_context_tokens": context_tokens_total / n,
        "cost_usd_per_1k_queries": context_tokens_total * PRICE_IN_PER_M / 1e6 * (1000 / n),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=60)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--window", type=int, default=8)
    args = parser.parse_args()

    banner("Semantic memory vs naive history")
    history, needles = build_synthetic_history(args.turns)
    embed_history(history)

    results = [
        evaluate("full_history", history, needles, k=args.k, window=args.window),
        evaluate("rolling_window", history, needles, k=args.k, window=args.window),
        evaluate("semantic", history, needles, k=args.k, window=args.window),
    ]
    print(json.dumps(results, indent=2))

    banner("Takeaway")
    print(
        "Semantic retrieval hits the needle without paying for the full history.\n"
        "Rolling window is cheap but drops information the user still cares about.\n"
        "In production, blend a short summary (temporal grounding) + semantic k-NN."
    )


if __name__ == "__main__":
    main()
