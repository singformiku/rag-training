"""
Speculative Execution — race cheap vs expensive, keep cheap if verified
=======================================================================

What this demo shows
--------------------
For each task we launch two calls concurrently:
  * ``cheap`` tier (fast, low-latency)
  * ``expensive`` tier (slow, higher-quality)

As soon as ``cheap`` returns, we run a deterministic verifier.  If it passes,
we cancel / ignore the expensive call and use the cheap answer.  Otherwise we
wait for the expensive answer.

This is the article's "speculative execution" pattern.  Trade-off: extra $$
spent on the losing race, but lower p95 latency.

Metrics reported:
  * wall-clock latency per task
  * $ spent on the winning call + $ wasted on the losing call
  * % of tasks where the cheap answer was good enough
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import BACKEND, MODELS, banner, chat_async, dollars  # noqa: E402

PRICE = {
    "cheap":     {"in": 1.00, "out": 5.00},
    "expensive": {"in": 5.00, "out": 25.00},
}


def verify_json_city(text: str) -> bool:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return False
    return {"city", "population", "country"}.issubset(obj.keys())


def verify_integer(text: str) -> bool:
    return bool(re.search(r"\b\d{3,}\b", text))


def verify_bullet_list(text: str) -> bool:
    return len([ln for ln in text.splitlines() if ln.lstrip().startswith(("-", "*"))]) >= 2


def verify_contains(word: str) -> Callable[[str], bool]:
    def _v(text: str) -> bool:
        return word.lower() in text.lower()
    return _v


TASKS: List[Tuple[str, Callable[[str], bool]]] = [
    ("Return ONLY a JSON object with keys city, population, country for Tokyo.",   verify_json_city),
    ("Return ONLY a JSON object with keys city, population, country for Berlin.", verify_json_city),
    ("99 squared is?",                                                             verify_integer),
    ("List 3 advantages of PostgreSQL over MySQL as bullet points.",               verify_bullet_list),
    ("Name the programming language invented by Guido van Rossum.",                verify_contains("python")),
]


async def run_speculative(prompt: str, verifier: Callable[[str], bool]) -> Dict:
    t0 = time.perf_counter()
    cheap_task = asyncio.create_task(chat_async(
        messages=[
            {"role": "system", "content": "Follow instructions literally."},
            {"role": "user", "content": prompt},
        ],
        tier="cheap", max_tokens=400, extra={"reasoning_effort": "low"},
    ))
    exp_task = asyncio.create_task(chat_async(
        messages=[
            {"role": "system", "content": "Follow instructions literally."},
            {"role": "user", "content": prompt},
        ],
        tier="expensive", max_tokens=400, extra={"reasoning_effort": "high"},
    ))

    cheap = await cheap_task
    cheap_cost = (cheap.input_tokens * PRICE["cheap"]["in"]
                  + cheap.output_tokens * PRICE["cheap"]["out"]) / 1e6
    if verifier(cheap.text):
        exp_task.cancel()
        try:
            await exp_task
        except (asyncio.CancelledError, Exception):
            pass
        return {
            "winner": "cheap", "winner_text": cheap.text.strip(),
            "cheap_cost": cheap_cost, "exp_cost": 0.0,
            "latency_s": time.perf_counter() - t0,
        }
    exp = await exp_task
    exp_cost = (exp.input_tokens * PRICE["expensive"]["in"]
                + exp.output_tokens * PRICE["expensive"]["out"]) / 1e6
    return {
        "winner": "expensive", "winner_text": exp.text.strip(),
        "cheap_cost": cheap_cost, "exp_cost": exp_cost,
        "latency_s": time.perf_counter() - t0,
    }


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=len(TASKS))
    args = parser.parse_args()

    banner(f"Speculative execution demo  |  backend={BACKEND}  "
           f"cheap={MODELS['cheap']}  expensive={MODELS['expensive']}")

    rows = []
    for prompt, v in TASKS[: args.tasks]:
        r = await run_speculative(prompt, v)
        rows.append(r)
        print(
            f"[{r['winner']:<9}] lat={r['latency_s']:5.2f}s  "
            f"cheap_cost={dollars(r['cheap_cost'])}  exp_cost={dollars(r['exp_cost'])}  "
            f"text={r['winner_text'][:70]}"
        )

    banner("Summary")
    n = len(rows)
    cheap_wins = sum(1 for r in rows if r["winner"] == "cheap")
    total_spec_cost = sum(r["cheap_cost"] + r["exp_cost"] for r in rows)
    total_always_exp = sum(r["exp_cost"] or (r["cheap_cost"] * PRICE["expensive"]["in"] / PRICE["cheap"]["in"])
                           for r in rows)
    avg_lat = sum(r["latency_s"] for r in rows) / n
    print(f"cheap-wins            : {cheap_wins}/{n}")
    print(f"avg latency           : {avg_lat:.2f}s")
    print(f"speculative total cost: {dollars(total_spec_cost)}")
    print(f"always-exp baseline   : {dollars(total_always_exp)}")
    print(
        "\nSpeculative execution trades $ for p95 latency. Best used when:\n"
        "- cheap wins often AND the expensive model is significantly slower.\n"
        "- failed calls are safe to discard (no side effects)."
    )


if __name__ == "__main__":
    asyncio.run(_main())
