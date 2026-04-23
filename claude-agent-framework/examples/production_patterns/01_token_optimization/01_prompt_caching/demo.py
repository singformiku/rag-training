"""
Prompt Caching — the dominant token-optimization lever
======================================================

What this demo shows
--------------------
1. Build a large, STABLE system prompt (~thousands of tokens).
2. Run the same user query N times.
3. Compare input-token usage turn-by-turn.

On Anthropic the first call pays ``cache_creation_input_tokens``; every call
after that reports ``cache_read_input_tokens`` which are billed at 10% of the
normal input price. On the internal OpenAI-compatible endpoint,
``prompt_tokens_details.cached_tokens`` reports how many of the input tokens
were served from the provider's automatic prefix cache.

Why it matters
--------------
On long-running agents, caching cuts per-turn input cost by 80–95% and TTFT
by 50–85% — see section "Prompt caching is the dominant lever" in the dossier.

Run
---
    python demo.py                      # defaults: 4 turns, ~10k system prompt
    python demo.py --turns 6 --repeats 200

Costs ~fractions of a cent on Anthropic; $0 on the internal endpoint.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Make ``_common`` importable when demo is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import BACKEND, MODELS, banner, chat, count_tokens, dollars, price_usd  # noqa: E402


def build_stable_prefix(repeats: int) -> str:
    """A fully deterministic block — same bytes on every call, crucial for caching."""
    block = (
        "You are a senior SRE assistant. Follow the runbook strictly.\n"
        "## Coding standards\n"
        "- Write idiomatic Python, type-annotated.\n"
        "- Prefer small pure functions.\n"
        "## Incident protocol\n"
        "- Acknowledge the alert, summarise the top-3 likely root causes,\n"
        "  then propose the safest mitigation with rollback plan.\n"
        "## Output contract\n"
        "- Always structure responses as: Summary, Evidence, Next actions.\n"
    )
    return "\n".join(block for _ in range(repeats))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=120,
                        help="Number of times to repeat the stable block.")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    banner(f"Prompt Caching demo  |  backend={BACKEND}  model={MODELS['medium']}")
    stable = build_stable_prefix(args.repeats)
    tok_in_prefix = count_tokens(stable)
    print(f"Stable system prefix: ~{tok_in_prefix:,} tokens")
    print(f"Running {args.turns} sequential turns with IDENTICAL system prompt.\n")

    user_query = (
        "Alert: p99 latency on /api/checkout just crossed 1200ms for 5 minutes. "
        "Error rate is flat. What do we do?"
    )

    rows: List[dict] = []
    for t in range(1, args.turns + 1):
        messages = [
            {"role": "system", "content": stable},
            {"role": "user", "content": user_query},
        ]
        r = chat(
            messages,
            tier="medium",
            system_cache=True,           # Anthropic: enable ephemeral cache_control
            max_tokens=args.max_tokens,
            temperature=0.0,
            extra={"reasoning_effort": "low"},  # gpt-oss: keep output budget predictable
        )
        rows.append({
            "turn": t,
            "in": r.input_tokens,
            "out": r.output_tokens,
            "cache_read": r.cache_read_tokens,
            "cache_create": r.cache_create_tokens,
            "latency_s": r.latency_s,
            "cost_usd": price_usd(r.model, r),
        })
        print(
            f"turn {t}: in={r.input_tokens:>6}  out={r.output_tokens:>4}  "
            f"cache_read={r.cache_read_tokens:>6}  cache_create={r.cache_create_tokens:>6}  "
            f"latency={r.latency_s:5.2f}s  cost={dollars(price_usd(r.model, r))}"
        )

    banner("Summary")
    warm = rows[1:] if len(rows) > 1 else rows
    cold = rows[0]
    avg_warm_cache_read = sum(r["cache_read"] for r in warm) / max(1, len(warm))
    avg_warm_in = sum(r["in"] for r in warm) / max(1, len(warm))
    avg_warm_lat = sum(r["latency_s"] for r in warm) / max(1, len(warm))

    total_cost = sum(r["cost_usd"] for r in rows)
    # Hypothetical: no cache on any turn — pay full input on every call.
    no_cache_cost = sum(
        price_usd(MODELS["medium"], {"input_tokens": r["in"] + r["cache_read"],
                                     "output_tokens": r["out"],
                                     "cache_read_tokens": 0,
                                     "cache_create_tokens": 0})
        for r in rows
    )

    print(f"Cold turn 1  : in={cold['in']}  cache_read={cold['cache_read']}  latency={cold['latency_s']:.2f}s")
    print(f"Warm avg     : in={avg_warm_in:.0f}  cache_read={avg_warm_cache_read:.0f}  latency={avg_warm_lat:.2f}s")
    print(f"Total cost   : {dollars(total_cost)}")
    print(f"If uncached  : {dollars(no_cache_cost)}   (savings: {dollars(no_cache_cost - total_cost)})")

    if BACKEND == "llm_service" and all(r["cache_read"] == 0 for r in rows):
        print("\nNOTE: the internal gpt-oss-120b endpoint did not report a cache hit. "
              "Automatic prefix caching is only active on some providers; the technique "
              "itself is still shown — observe input tokens stay flat while latency may "
              "drop on warm calls due to provider-side KV reuse.")


if __name__ == "__main__":
    main()
