"""
Semantic Snapshot Testing — cosine similarity + LLM judge fallback
==================================================================

What this demo shows
--------------------
For each prompt we:
  1. Load an "expected" snapshot from disk.
  2. Generate a fresh answer with the LLM (non-zero temperature).
  3. Decide equivalence via a 3-tier check:
       (a) exact string equality (cheapest)
       (b) cosine similarity of embeddings ≥ THRESHOLD
       (c) LLM judge on the pair → {equivalent: bool}

Metrics reported:
  * how many snapshots each tier caught
  * how many were flagged as regressions
  * time + $ per tier (embedding is free-ish on sentence-transformers)

First run creates the ``__snapshots__`` folder automatically (UPDATE_SNAPSHOTS=1
also re-writes them).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import BACKEND, MODELS, banner, chat, embed  # noqa: E402


SNAP_DIR = Path(__file__).with_name("__snapshots__")
SNAP_DIR.mkdir(exist_ok=True)


PROMPTS: List[Tuple[str, str]] = [
    ("q1_idempotent", "Define 'idempotent' in computing in one sentence."),
    ("q2_cap",        "Summarise the CAP theorem in two sentences."),
    ("q3_jwt",        "Explain the difference between JWT and session cookies in 1-2 sentences."),
]


def _cos(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0
    return num / den


def semantic_equivalent(expected: str, actual: str, threshold: float = 0.82) -> Tuple[bool, float]:
    a = embed(expected)
    b = embed(actual)
    sim = _cos(a, b)
    return sim >= threshold, sim


def llm_judge_equivalent(expected: str, actual: str) -> bool:
    r = chat(
        messages=[
            {"role": "system", "content":
                "Decide whether two texts express the SAME factual content. "
                "Respond with ONLY a JSON object: {\"equivalent\": true|false}."},
            {"role": "user", "content":
                f"EXPECTED:\n{expected}\n\nACTUAL:\n{actual}"},
        ],
        tier="cheap",
        max_tokens=64,
        extra={"reasoning_effort": "low"},
    )
    import re
    m = re.search(r"\{.*\}", r.text or "", re.DOTALL)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
        return bool(obj.get("equivalent"))
    except Exception:
        return False


def snapshot_check(name: str, actual: str, *, threshold: float, update: bool) -> Dict[str, str]:
    path = SNAP_DIR / f"{name}.txt"
    if update or not path.exists():
        path.write_text(actual, encoding="utf-8")
        return {"name": name, "decision": "created", "detail": "snapshot recorded"}

    expected = path.read_text(encoding="utf-8")
    if expected.strip() == actual.strip():
        return {"name": name, "decision": "exact_match", "detail": ""}
    ok, sim = semantic_equivalent(expected, actual, threshold=threshold)
    if ok:
        return {"name": name, "decision": "semantic_match", "detail": f"cos={sim:.3f}"}
    if llm_judge_equivalent(expected, actual):
        return {"name": name, "decision": "judge_match", "detail": f"cos={sim:.3f}"}
    return {"name": name, "decision": "regression", "detail": f"cos={sim:.3f}"}


def generate(prompt: str) -> str:
    r = chat(
        messages=[
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": prompt},
        ],
        tier="cheap",
        max_tokens=200,
        temperature=0.3,   # non-zero, to drive wording drift between runs
        extra={"reasoning_effort": "low"},
    )
    return r.text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--update", action="store_true",
                        help="Overwrite snapshots with the current output.")
    args = parser.parse_args()
    update = args.update or os.getenv("UPDATE_SNAPSHOTS") == "1"

    banner(f"Semantic snapshot demo  |  backend={BACKEND}  model={MODELS['cheap']}")
    print(f"snapshot dir: {SNAP_DIR}")
    print(f"threshold   : {args.threshold}  update={update}\n")

    rows = []
    for name, prompt in PROMPTS:
        actual = generate(prompt)
        result = snapshot_check(name, actual, threshold=args.threshold, update=update)
        rows.append({**result, "actual_preview": actual[:120]})
        print(f"{name:<14} decision={result['decision']:<15} {result['detail']}")
        print(f"               actual: {actual[:100]!r}")

    banner("Summary")
    decisions = {}
    for r in rows:
        decisions[r["decision"]] = decisions.get(r["decision"], 0) + 1
    for d, c in decisions.items():
        print(f"  {d:<15}: {c}")

    if decisions.get("regression", 0):
        print("\n>>> regressions detected — inspect the diff and decide whether to "
              "update the snapshot (UPDATE_SNAPSHOTS=1) or fix the prompt/model.")


if __name__ == "__main__":
    main()
