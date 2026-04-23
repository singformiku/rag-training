"""
Cascade Routing — start cheap, escalate on verified failure
===========================================================

What this demo shows
--------------------
For each task we:
  1. Answer with the ``cheap`` tier.
  2. Run a deterministic **verifier** (schema check, unit test, regex, etc.).
  3. If the verifier fails, escalate to ``expensive`` and re-run.

We report per-task the tier(s) used, verifier outcome, total input/output
tokens, and a blended $ cost computed against the article's Haiku / Opus
pricing table.

The break-even property: cascade is a win whenever
    cheap_cost + P(fail) × (cheap_cost + expensive_cost) < expensive_cost
i.e. the expected escalation rate stays below a task-specific threshold.

The demo prints that threshold and the observed escalation rate so you can
see whether the cascade would pay off on your workload.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import BACKEND, MODELS, banner, chat, dollars  # noqa: E402

# Article-referenced pricing for illustrative blended cost.
PRICE = {
    "cheap":     {"in": 1.00, "out": 5.00},   # Haiku-class
    "expensive": {"in": 5.00, "out": 25.00},  # Opus-class
}


# ---------------------------------------------------------------------------
# Tasks + verifiers
# ---------------------------------------------------------------------------
def verify_json(text: str) -> bool:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return False
    return {"city", "population", "country"}.issubset(obj.keys())


def verify_math(text: str) -> bool:
    # Looking for the integer 9801 (99^2) anywhere in the response.
    return "9801" in text.replace(",", "")


def verify_regex(text: str) -> bool:
    # Must produce a regex that matches valid IPv4 addresses in the output.
    return bool(re.search(r"\^.*\\.?\d.*\$", text) or re.search(r"\^(?:\d{1,3}\\?\.){3}\d{1,3}\$", text))


def verify_nonempty(text: str) -> bool:
    return len(text.strip()) > 40


TASKS: List[Tuple[str, str, Callable[[str], bool]]] = [
    ("Return ONLY a JSON object with keys city, population, country for Tokyo.",
     "json_extract", verify_json),
    ("What is 99 squared? Reply with ONLY the integer.",
     "exact_math", verify_math),
    ("Write a regex that matches any IPv4 address. Reply with ONLY the regex.",
     "regex_author", verify_regex),
    ("Summarise in 3 sentences: what is a Kubernetes Deployment?",
     "summary", verify_nonempty),
    ("Return ONLY a JSON object with keys city, population, country for São Paulo.",
     "json_extract", verify_json),
]


def run_task(prompt: str, verifier: Callable[[str], bool]) -> Dict:
    # 1) cheap tier
    cheap = chat(
        messages=[
            {"role": "system", "content": "Follow the instructions exactly. Return only what is asked."},
            {"role": "user", "content": prompt},
        ],
        tier="cheap",
        max_tokens=400,
        extra={"reasoning_effort": "low"},
    )
    cheap_ok = verifier(cheap.text)
    cost = (cheap.input_tokens * PRICE["cheap"]["in"] + cheap.output_tokens * PRICE["cheap"]["out"]) / 1e6
    if cheap_ok:
        return {
            "tier": "cheap", "verifier_passed": True,
            "cheap_tokens": (cheap.input_tokens, cheap.output_tokens),
            "expensive_tokens": (0, 0),
            "cost_usd": cost, "final_text": cheap.text.strip(),
        }

    # 2) escalate
    exp = chat(
        messages=[
            {"role": "system", "content": "Follow the instructions exactly. Return only what is asked."},
            {"role": "user", "content": prompt},
        ],
        tier="expensive",
        max_tokens=800,
        extra={"reasoning_effort": "high"},
    )
    exp_ok = verifier(exp.text)
    cost += (exp.input_tokens * PRICE["expensive"]["in"] + exp.output_tokens * PRICE["expensive"]["out"]) / 1e6
    return {
        "tier": "expensive", "verifier_passed": exp_ok,
        "cheap_tokens": (cheap.input_tokens, cheap.output_tokens),
        "expensive_tokens": (exp.input_tokens, exp.output_tokens),
        "cost_usd": cost, "final_text": exp.text.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=len(TASKS))
    args = parser.parse_args()

    banner(f"Cascade routing demo  |  backend={BACKEND}  "
           f"cheap={MODELS['cheap']}  expensive={MODELS['expensive']}")
    print(f"{args.tasks} tasks. Cheap tier tries first; escalate only on verifier fail.\n")

    results: List[Dict] = []
    for prompt, kind, v in TASKS[: args.tasks]:
        r = run_task(prompt, v)
        r["kind"] = kind
        results.append(r)
        print(
            f"[{r['tier']:<9}] kind={kind:<13} verified={r['verifier_passed']}  "
            f"cost={dollars(r['cost_usd'])}  out={r['final_text'][:70]}"
        )

    banner("Summary")
    n = len(results)
    escalated = sum(1 for r in results if r["tier"] == "expensive")
    cheap_only_cost = sum(
        ((r["cheap_tokens"][0] * PRICE["cheap"]["in"] + r["cheap_tokens"][1] * PRICE["cheap"]["out"]) / 1e6)
        for r in results
    )
    cascade_cost = sum(r["cost_usd"] for r in results)
    # Hypothetical: always-expensive baseline, same output tokens.
    baseline_expensive = sum(
        ((r["cheap_tokens"][0] * PRICE["expensive"]["in"] + r["cheap_tokens"][1] * PRICE["expensive"]["out"]) / 1e6)
        for r in results
    )
    print(f"escalation rate       : {escalated}/{n} = {100*escalated/n:.1f}%")
    print(f"cheap-only cost       : {dollars(cheap_only_cost)}")
    print(f"cascade cost          : {dollars(cascade_cost)}")
    print(f"always-expensive cost : {dollars(baseline_expensive)}")

    # Break-even escalation rate: cascade wins while escalation < threshold.
    # threshold = (Pexp − Pcheap) / (Pexp + Pcheap) on a matched token budget.
    p_in_ratio = PRICE["expensive"]["in"] / PRICE["cheap"]["in"]
    threshold = (p_in_ratio - 1) / (p_in_ratio + 1)
    print(f"break-even threshold  : ~{threshold*100:.0f}% (escalation rate)")
    if escalated / n <= threshold:
        print(">> Cascade pays off on this workload.")
    else:
        print(">> Cascade does NOT pay off — consider stronger cheap prompt or classifier.")


if __name__ == "__main__":
    main()
