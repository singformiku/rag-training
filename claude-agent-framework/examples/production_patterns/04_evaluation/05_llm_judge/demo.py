"""
LLM-as-Judge with bias mitigations (position swap + ensemble)
=============================================================

What this demo shows
--------------------
For each (query, answer_A, answer_B) tuple we run a pairwise judge vote in
two positions (A=X,B=Y) and (A=Y,B=X) — this is the position-swap
mitigation from MT-Bench & Arena-Hard.  Votes where the judge flipped
after the swap are counted as ``inconsistent`` and discarded.

We report:
  * winner (X / Y / tie)
  * consistency (swaps that agreed vs flipped)
  * cost per comparison

When two models are available (``medium`` and ``expensive``) we also run an
**ensemble** across both and combine votes — the article cites this as the
cure for self-preference bias.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import BACKEND, MODELS, banner, chat, dollars  # noqa: E402

PRICE_IN_PER_M = 3.00
PRICE_OUT_PER_M = 15.00


JUDGE_PROMPT = """You are an expert evaluator. Two AI assistants answered the same query.
Think step by step through the criteria BEFORE stating a verdict.
Criteria (in order): factual correctness > instruction adherence > helpfulness > concision.
Explicitly IGNORE answer length as a quality signal.

User query: {query}
[Assistant A]
{a}
[End A]
[Assistant B]
{b}
[End B]

Return STRICT JSON only: {{"reasoning": "...", "verdict": "A" | "B" | "tie"}}"""


PAIRS: List[Dict[str, str]] = [
    {
        "query": "Define 'idempotent' in one sentence.",
        "x": "An idempotent operation produces the same result whether you apply it once or many times.",
        "y": "Idempotent means 'the same' in Latin and describes functions that can be applied repeatedly without changing the result beyond the initial application; see also HTTP PUT.",
    },
    {
        "query": "What's 99 squared? Respond with ONLY the integer.",
        "x": "9801",
        "y": "Approximately 10000.",
    },
    {
        "query": "Explain the CAP theorem in two sentences.",
        "x": "CAP states that a distributed system can only guarantee two of Consistency, Availability, and Partition tolerance at any given moment. In practice, partitions happen, so systems choose between CP and AP trade-offs.",
        "y": "CAP is a theorem about caches. It says systems should be cached, available, and partition-resistant.",
    },
]


def _judge_once(query: str, a: str, b: str, *, tier: str) -> tuple[dict, int, int]:
    r = chat(
        messages=[
            {"role": "system", "content": "You judge answer quality. Be strict."},
            {"role": "user", "content": JUDGE_PROMPT.format(query=query, a=a, b=b)},
        ],
        tier=tier,
        max_tokens=500,
        extra={"reasoning_effort": "medium"},
    )
    m = re.search(r"\{.*\}", r.text or "", re.DOTALL)
    if not m:
        return {"verdict": "tie"}, r.input_tokens, r.output_tokens
    try:
        obj = json.loads(m.group(0))
    except Exception:
        obj = {"verdict": "tie"}
    if obj.get("verdict") not in ("A", "B", "tie"):
        obj["verdict"] = "tie"
    return obj, r.input_tokens, r.output_tokens


def pairwise_vote(query: str, x: str, y: str, judges: List[str]) -> Dict:
    """Returns {'winner', 'confidence', 'by_judge', 'tokens_in', 'tokens_out'}."""
    votes = {"X": 0, "Y": 0, "tie": 0, "inconsistent": 0}
    by_judge = {}
    tot_in = tot_out = 0
    for j in judges:
        v1, ti1, to1 = _judge_once(query, x, y, tier=j)  # X=A, Y=B
        v2, ti2, to2 = _judge_once(query, y, x, tier=j)  # swapped
        tot_in += ti1 + ti2
        tot_out += to1 + to2
        d1, d2 = v1["verdict"], v2["verdict"]
        if d1 == "A" and d2 == "B":
            final = "X"
        elif d1 == "B" and d2 == "A":
            final = "Y"
        elif d1 == "tie" and d2 == "tie":
            final = "tie"
        else:
            final = "inconsistent"
        by_judge[j] = {"position1": d1, "position2": d2, "decision": final}
        votes[final] += 1
    valid = votes["X"] + votes["Y"] + votes["tie"]
    if valid == 0:
        winner = "inconclusive"
        confidence = 0.0
    else:
        winner = max(("X", "Y", "tie"), key=lambda k: votes[k])
        confidence = votes[winner] / valid
    return {
        "winner": winner, "confidence": confidence,
        "by_judge": by_judge, "votes": votes,
        "tokens_in": tot_in, "tokens_out": tot_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", action="store_true",
                        help="Run both 'medium' and 'expensive' tiers as judges.")
    args = parser.parse_args()

    banner(f"LLM-as-judge demo  |  backend={BACKEND}  ensemble={args.ensemble}")
    judges = ["medium", "expensive"] if args.ensemble else ["medium"]
    print(f"judges: {judges}  (position-swap enabled on every judge)\n")

    rows = []
    for pair in PAIRS:
        res = pairwise_vote(pair["query"], pair["x"], pair["y"], judges)
        cost = (res["tokens_in"] * PRICE_IN_PER_M + res["tokens_out"] * PRICE_OUT_PER_M) / 1e6
        rows.append({**res, "query": pair["query"], "cost_usd": cost})
        print(
            f"query='{pair['query'][:50]}...'  winner={res['winner']}  "
            f"confidence={res['confidence']:.2f}  "
            f"cost={dollars(cost)}  votes={res['votes']}"
        )
        for j, detail in res["by_judge"].items():
            print(f"   judge={j:<9} pos1={detail['position1']} "
                  f"pos2={detail['position2']} → {detail['decision']}")

    banner("Summary")
    flip_rate = sum(r["votes"]["inconsistent"] for r in rows) / max(1, sum(len(r["by_judge"]) for r in rows))
    print(f"position-swap inconsistency rate: {flip_rate*100:.1f}%")
    print(
        "\nIn production: only trust pairs where position-swap agrees.  Run 2+ "
        "judges from different model families for release-gate evals — this is "
        "the setup LMArena and Arena-Hard-v2 standardise on."
    )


if __name__ == "__main__":
    main()
