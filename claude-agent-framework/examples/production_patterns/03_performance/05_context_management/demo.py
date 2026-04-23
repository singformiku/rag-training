"""
Context Window Management — sliding window, summarize-on-overflow, prefix cache
==============================================================================

What this demo shows
--------------------
A toy 30-turn conversation exceeds a small pretend ``budget_tokens=6000``
cap.  We try three context policies and report, at every turn, the size of
the context that gets sent to the main model:

  * ``sliding_window``      — keep only the last N turns
  * ``summarize_on_overflow`` — summarise older turns when over budget,
                                 keep the most recent ones verbatim
  * ``prefix_cache``        — stable system prompt marked cacheable (Anthropic);
                              simulated as "cheaper input" on llm_service.

A final table shows total context tokens and projected cost under each.

This is the composable pattern set from section "Context window management"
in the dossier.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import BACKEND, MODELS, banner, chat, count_tokens, dollars  # noqa: E402

PRICE_IN_PER_M = 3.00
PRICE_OUT_PER_M = 15.00
PRICE_CACHE_READ_PER_M = 0.30


SYSTEM_PROMPT_BASE = (
    "You are a senior SRE assistant. Follow the runbook strictly. "
    "You have access to incident dashboards, git logs, and paging info. "
    "Coding standards: idiomatic Python, type-annotated, small pure functions. "
) * 4


def make_turn(i: int) -> List[Dict[str, str]]:
    u = f"[turn {i}] I need to triage alert #{i}. Walk me through it."
    a = (
        f"Triage for alert #{i}:\n"
        "1. Check the affected service's p95 latency graph.\n"
        "2. Look at recent deploys.\n"
        "3. Correlate with DB CPU.\n"
        f"Observation: alert #{i} shows a latency blip around 14:{i:02d}."
    )
    return [{"role": "user", "content": u}, {"role": "assistant", "content": a}]


def summarize(older: List[Dict[str, str]]) -> str:
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in older)
    r = chat(
        messages=[
            {"role": "system", "content":
                "Compress this SRE dialogue into 5 bullet points. Preserve alert IDs and decisions."},
            {"role": "user", "content": convo[:8000]},
        ],
        tier="cheap",
        max_tokens=300,
        extra={"reasoning_effort": "low"},
    )
    return r.text.strip() or "(empty summary)"


def run_policy(policy: str, turns: int, budget: int, keep_last: int) -> Dict[str, float]:
    history: List[Dict[str, str]] = []
    summary = ""
    per_turn_ctx_tokens: List[int] = []
    summarize_io_in = 0
    summarize_io_out = 0
    cache_read_total = 0

    for t in range(1, turns + 1):
        history.extend(make_turn(t))

        if policy == "sliding_window":
            effective_msgs = [{"role": "system", "content": SYSTEM_PROMPT_BASE}] + history[-keep_last * 2:]

        elif policy == "summarize_on_overflow":
            current = [{"role": "system", "content": SYSTEM_PROMPT_BASE}] + history
            if count_tokens(current) > budget:
                older = history[: -keep_last * 2]
                recent = history[-keep_last * 2:]
                new_summary = summarize(older)
                summarize_io_in += count_tokens(older)
                summarize_io_out += count_tokens(new_summary)
                history = recent
                summary = new_summary
            effective_msgs = [{"role": "system", "content": SYSTEM_PROMPT_BASE}]
            if summary:
                effective_msgs.append({"role": "system", "content": f"<prior_summary>{summary}</prior_summary>"})
            effective_msgs.extend(history)

        elif policy == "prefix_cache":
            effective_msgs = [{"role": "system", "content": SYSTEM_PROMPT_BASE}] + history
            # The stable system prompt is cacheable; credit it as a cache hit
            # on every turn after the first.
            if t > 1:
                cache_read_total += count_tokens(SYSTEM_PROMPT_BASE)

        else:
            raise ValueError(policy)

        per_turn_ctx_tokens.append(count_tokens(effective_msgs))

    total_input = sum(per_turn_ctx_tokens)
    main_cost = total_input * PRICE_IN_PER_M / 1e6
    if policy == "prefix_cache":
        main_cost -= cache_read_total * PRICE_IN_PER_M / 1e6                 # remove full-price share
        main_cost += cache_read_total * PRICE_CACHE_READ_PER_M / 1e6         # add cached-price share
    summary_cost = (summarize_io_in * 1.00 + summarize_io_out * 5.00) / 1e6  # cheap-tier summariser
    total_cost = main_cost + summary_cost
    return {
        "policy": policy,
        "total_input_tokens": total_input,
        "peak_input_tokens": max(per_turn_ctx_tokens),
        "cache_read_tokens": cache_read_total,
        "summary_tokens_in": summarize_io_in,
        "summary_tokens_out": summarize_io_out,
        "total_cost_usd": total_cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--budget", type=int, default=6000)
    parser.add_argument("--keep-last", type=int, default=4)
    args = parser.parse_args()

    banner(f"Context management demo  |  backend={BACKEND}  model={MODELS['medium']}")
    print(f"turns={args.turns}, budget={args.budget}, keep_last={args.keep_last}\n")

    results = []
    for p in ("sliding_window", "summarize_on_overflow", "prefix_cache"):
        print(f"running policy: {p} ...")
        results.append(run_policy(p, args.turns, args.budget, args.keep_last))

    banner("Results")
    print(json.dumps(results, indent=2))
    for r in results:
        print(
            f"{r['policy']:<22} peak={r['peak_input_tokens']:>6} toks  "
            f"cost={dollars(r['total_cost_usd'])}  "
            f"cache_read={r['cache_read_tokens']}"
        )
    print(
        "\nIn production these compose: prefix-cache the system prompt, sliding-"
        "window the recent dialogue, summarise everything older.  Measure on "
        "YOUR traffic — the right cutoff depends on how often users re-reference "
        "old context."
    )


if __name__ == "__main__":
    main()
