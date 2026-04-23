"""
Context Compression — summarize older turns with a cheap model
==============================================================

What this demo shows
--------------------
A 20-turn toy conversation is simulated.  Once total tokens cross a threshold,
older turns are summarised by the "cheap" tier and replaced with a single
``<prior_summary>`` block, while the most recent ``keep_last`` turns are kept
verbatim.

Three strategies are compared side-by-side on the same history:
  * ``none``           — send full history every turn (cost ceiling rises monotonically)
  * ``rolling_window`` — keep only the last ``keep_last`` turns
  * ``summarize``      — summarise older + keep last N verbatim

We report the per-turn context size and a projected $ cost against the
reference Claude Sonnet price (the article's baseline).

Run
---
    python demo.py
    python demo.py --turns 24 --threshold 6000 --keep-last 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import MODELS, BACKEND, banner, chat, count_tokens, dollars  # noqa: E402

PRICE_IN_PER_M = 3.00    # $/1M input tokens (Sonnet reference from dossier)
PRICE_OUT_PER_M = 15.00  # $/1M output tokens


def _fake_turn(turn: int) -> List[Dict[str, str]]:
    """Simulate a user/assistant exchange that grows with turn index."""
    user = f"[turn {turn}] Explain in 2 sentences: what changed at step {turn}?"
    asst = (
        f"At step {turn} the team modified the rollout job: "
        + ("payload details, " * 10).rstrip(", ")
        + f" and recorded outcome={turn % 2 == 0}."
    )
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": asst},
    ]


SUMMARIZER_SYSTEM = (
    "You compress assistant/user history for an agent. "
    "Preserve: user goals, decisions, unresolved TODOs, IDs, file paths, errors. "
    "Drop: pleasantries, verbose tool payloads, obsolete plans. "
    "Output 3–6 bullet points under the heading `Decisions so far`."
)


def summarize(history: List[Dict[str, str]]) -> str:
    """Use the cheap tier to compress ``history`` into a short summary."""
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    r = chat(
        messages=[
            {"role": "system", "content": SUMMARIZER_SYSTEM},
            {"role": "user", "content": f"Summarize:\n{convo[:12000]}"},
        ],
        tier="cheap",
        max_tokens=400,
        extra={"reasoning_effort": "low"},
    )
    return r.text.strip() or "(empty summary)"


def run_strategy(
    *, strategy: str, turns: int, threshold_tokens: int, keep_last: int,
) -> Dict[str, float]:
    history: List[Dict[str, str]] = []
    summary = ""
    per_turn_context: List[int] = []
    summarize_cost_tokens_in = 0
    summarize_cost_tokens_out = 0

    for t in range(1, turns + 1):
        history.extend(_fake_turn(t))

        if strategy == "none":
            effective = history[:]
        elif strategy == "rolling_window":
            effective = history[-keep_last * 2:]
        elif strategy == "summarize":
            total_tok = count_tokens(history)
            
            if total_tok >= threshold_tokens:
                older = history[: -keep_last * 2]
                recent = history[-keep_last * 2:]
                summary_text = summarize(older)
                # Credit the summarizer run to its own bucket.
                summarize_cost_tokens_in += count_tokens(older)
                summarize_cost_tokens_out += count_tokens(summary_text)
                history = recent
                summary = summary_text
            effective = (
                [{"role": "system", "content": f"<prior_summary>\n{summary}\n</prior_summary>"}]
                if summary else []
            ) + history
        else:
            raise ValueError(strategy)

        per_turn_context.append(count_tokens(effective))

    total_in = sum(per_turn_context)
    # The main-agent output per turn is a small fixed budget — normalize to 200.
    total_out = 200 * turns
    cost_main = (total_in * PRICE_IN_PER_M + total_out * PRICE_OUT_PER_M) / 1e6
    # Use Haiku-like price for summary calls: article cites ~$1 in / $5 out per MTok.
    cost_summary = (summarize_cost_tokens_in * 1.00 + summarize_cost_tokens_out * 5.00) / 1e6
    return {
        "strategy": strategy,
        "peak_context_tokens": max(per_turn_context),
        "avg_context_tokens": sum(per_turn_context) / len(per_turn_context),
        "total_main_cost_usd": cost_main,
        "summary_calls_cost_usd": cost_summary,
        "total_cost_usd": cost_main + cost_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=16)
    parser.add_argument("--threshold", type=int, default=4000)
    parser.add_argument("--keep-last", type=int, default=3)
    args = parser.parse_args()

    banner(f"Context compression demo  |  backend={BACKEND}  model={MODELS['medium']}")
    print(f"Simulating {args.turns} turns, threshold={args.threshold} tokens, "
          f"keep_last={args.keep_last}\n")

    results = []
    for strat in ("none", "rolling_window", "summarize"):
        print(f"running strategy: {strat} ...")
        r = run_strategy(
            strategy=strat, turns=args.turns,
            threshold_tokens=args.threshold, keep_last=args.keep_last,
        )
        results.append(r)

    banner("Results")
    print(json.dumps(results, indent=2))

    baseline = next(r for r in results if r["strategy"] == "none")["total_cost_usd"]
    for r in results:
        delta = 1 - r["total_cost_usd"] / max(baseline, 1e-12)
        print(
            f"{r['strategy']:<16} peak={r['peak_context_tokens']:>6} toks  "
            f"cost={dollars(r['total_cost_usd'])}  savings_vs_none={delta*100:+.1f}%"
        )

    print(
        "\nTakeaway: summarisation pays off whenever the saved input tokens cost "
        "more than the summariser's own I/O.  Plot this ratio against your real "
        "workload to find the right threshold."
    )


if __name__ == "__main__":
    main()
