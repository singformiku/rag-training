"""
Dynamic Few-Shot Selection — pick the 3 most similar examples per query
======================================================================

What this demo shows
--------------------
We have an "example bank" of 20 text-to-SQL pairs.  For each new question,
two strategies are compared:

  * ``static_8``    — always include the first 8 examples in the prompt
  * ``dynamic_k=3`` — pick top-3 examples by semantic similarity to the query

We report: input tokens, generated SQL (one-line), and whether it passes a
deterministic structural check (expected table + column).

The point is NOT to build a great text-to-SQL model — it's to measure how
dynamic few-shot uses *fewer* tokens AND produces *better* outputs than static
few-shot on the same bank.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import BACKEND, MODELS, banner, chat, count_tokens, dollars, embed  # noqa: E402

PRICE_IN_PER_M = 0.75
PRICE_OUT_PER_M = 4.50


EXAMPLE_BANK: List[Dict[str, str]] = [
    {"q": "How many users signed up last week?",
     "sql": "SELECT COUNT(*) FROM users WHERE signup_date >= CURRENT_DATE - INTERVAL '7 day';"},
    {"q": "Top 5 countries by revenue in Q3.",
     "sql": "SELECT country, SUM(amount) s FROM orders WHERE quarter='2025-Q3' GROUP BY country ORDER BY s DESC LIMIT 5;"},
    {"q": "Average order value for VIP customers in 2025.",
     "sql": "SELECT AVG(amount) FROM orders WHERE customer_tier='VIP' AND EXTRACT(year FROM created_at)=2025;"},
    {"q": "Number of pull requests opened this week per repo.",
     "sql": "SELECT repo, COUNT(*) FROM pull_requests WHERE opened_at >= CURRENT_DATE - INTERVAL '7 day' GROUP BY repo;"},
    {"q": "Which products have stock below 10?",
     "sql": "SELECT product_id, stock FROM inventory WHERE stock < 10;"},
    {"q": "List users whose email ends in @acme.com",
     "sql": "SELECT id, email FROM users WHERE email LIKE '%@acme.com';"},
    {"q": "Refund rate per month in 2025.",
     "sql": "SELECT DATE_TRUNC('month', created_at) m, AVG(CASE WHEN status='refunded' THEN 1.0 ELSE 0 END) FROM orders WHERE EXTRACT(year FROM created_at)=2025 GROUP BY m;"},
    {"q": "Top-10 slowest API endpoints last 24h.",
     "sql": "SELECT path, AVG(latency_ms) l FROM api_logs WHERE ts >= NOW()-INTERVAL '1 day' GROUP BY path ORDER BY l DESC LIMIT 10;"},
    {"q": "How many distinct customers placed orders yesterday?",
     "sql": "SELECT COUNT(DISTINCT customer_id) FROM orders WHERE DATE(created_at)=CURRENT_DATE - 1;"},
    {"q": "Churned customers who returned in the last 30 days.",
     "sql": "SELECT customer_id FROM customers WHERE churned_at IS NOT NULL AND last_login >= CURRENT_DATE - INTERVAL '30 day';"},
    {"q": "Average tickets per agent this month.",
     "sql": "SELECT agent_id, COUNT(*)/30.0 FROM support_tickets WHERE DATE_TRUNC('month', created_at)=DATE_TRUNC('month', CURRENT_DATE) GROUP BY agent_id;"},
    {"q": "Orders placed on Black Friday 2024.",
     "sql": "SELECT id FROM orders WHERE created_at::date = DATE '2024-11-29';"},
    {"q": "DAU for the mobile app over the past 30 days.",
     "sql": "SELECT DATE(ts) d, COUNT(DISTINCT user_id) FROM events WHERE source='mobile' AND ts >= CURRENT_DATE - INTERVAL '30 day' GROUP BY d;"},
    {"q": "Which campaigns had a CTR above 10%?",
     "sql": "SELECT campaign_id FROM marketing_campaigns WHERE ctr > 0.10;"},
    {"q": "Invoices overdue by more than 30 days.",
     "sql": "SELECT id, customer_id FROM invoices WHERE due_date < CURRENT_DATE - INTERVAL '30 day' AND status<>'paid';"},
    {"q": "Top-5 search queries in the last 24 hours.",
     "sql": "SELECT query, COUNT(*) c FROM search_logs WHERE ts >= NOW()-INTERVAL '1 day' GROUP BY query ORDER BY c DESC LIMIT 5;"},
    {"q": "Customers subscribed to the Pro plan.",
     "sql": "SELECT id FROM customers WHERE plan='Pro';"},
    {"q": "Order count by shipping country for the last quarter.",
     "sql": "SELECT shipping_country, COUNT(*) FROM orders WHERE created_at >= CURRENT_DATE - INTERVAL '90 day' GROUP BY shipping_country;"},
    {"q": "Agents who closed more than 200 tickets this year.",
     "sql": "SELECT agent_id FROM support_tickets WHERE closed_at IS NOT NULL AND EXTRACT(year FROM closed_at)=2025 GROUP BY agent_id HAVING COUNT(*)>200;"},
    {"q": "Users who have never placed an order.",
     "sql": "SELECT u.id FROM users u LEFT JOIN orders o ON o.customer_id=u.id WHERE o.id IS NULL;"},
]

# Test queries + required (table, column) that a correct SQL must mention.
TEST_QUERIES: List[Tuple[str, str, str]] = [
    ("How many sign-ups happened yesterday?",                         "users",      "signup_date"),
    ("Revenue total from VIP customers in August 2025.",              "orders",     "customer_tier"),
    ("Top 3 slowest API endpoints in the past hour.",                 "api_logs",   "latency_ms"),
    ("List invoices overdue by 14 days.",                             "invoices",   "due_date"),
]


def _cos(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0
    return num / den


def embed_bank() -> List[List[float]]:
    return [embed(e["q"]) for e in EXAMPLE_BANK]


def pick_dynamic(query: str, bank_vecs: List[List[float]], k: int) -> List[Dict[str, str]]:
    q_vec = embed(query)
    scored = [(_cos(q_vec, v), i) for i, v in enumerate(bank_vecs)]
    scored.sort(key=lambda x: -x[0])
    return [EXAMPLE_BANK[i] for _, i in scored[:k]]


def format_fewshot(examples: List[Dict[str, str]]) -> str:
    return "\n\n".join(f"Q: {e['q']}\nSQL: {e['sql']}" for e in examples)


def ask_for_sql(query: str, fewshot: str) -> tuple[str, int, int]:
    r = chat(
        messages=[
            {"role": "system", "content":
                "You translate an English question into a single-line SQL "
                "statement. Do not include explanation, only the SQL."},
            {"role": "user", "content":
                f"Examples:\n{fewshot}\n\nNow answer only with SQL:\nQ: {query}\nSQL:"},
        ],
        tier="cheap",
        max_tokens=200,
        extra={"reasoning_effort": "low"},
    )
    sql = (r.text or "").strip().splitlines()[0] if r.text else ""
    return sql, r.input_tokens, r.output_tokens


def check(sql: str, must_table: str, must_col: str) -> bool:
    return must_table.lower() in sql.lower() and must_col.lower() in sql.lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    banner(f"Dynamic few-shot demo  |  backend={BACKEND}  model={MODELS['cheap']}")
    print(f"Bank size={len(EXAMPLE_BANK)}, test queries={len(TEST_QUERIES)}, k={args.k}\n")
    bank_vecs = embed_bank()

    static_examples = EXAMPLE_BANK[:8]
    static_fewshot = format_fewshot(static_examples)

    results: List[Dict] = []
    for q, tbl, col in TEST_QUERIES:
        for policy in ("static_8", f"dynamic_k{args.k}"):
            if policy == "static_8":
                fs = static_fewshot
                examples_used = len(static_examples)
            else:
                picked = pick_dynamic(q, bank_vecs, args.k)
                fs = format_fewshot(picked)
                examples_used = len(picked)
            sql, ti, to = ask_for_sql(q, fs)
            ok = check(sql, tbl, col)
            results.append({
                "policy": policy, "query": q, "examples": examples_used,
                "fewshot_tokens": count_tokens(fs),
                "input_tokens": ti, "output_tokens": to,
                "sql": sql, "structural_pass": ok,
            })
            print(f"[{policy:<11}] ({examples_used}ex, in={ti}) pass={ok} sql={sql[:80]}")

    banner("Summary")
    for policy in sorted({r["policy"] for r in results}):
        subset = [r for r in results if r["policy"] == policy]
        pass_rate = sum(1 for r in subset if r["structural_pass"]) / len(subset)
        avg_fs = sum(r["fewshot_tokens"] for r in subset) / len(subset)
        avg_in = sum(r["input_tokens"] for r in subset) / len(subset)
        cost = sum((r["input_tokens"] * PRICE_IN_PER_M + r["output_tokens"] * PRICE_OUT_PER_M)
                    for r in subset) / 1e6
        print(
            f"{policy:<12} pass_rate={pass_rate*100:>5.1f}%  "
            f"avg_fewshot_tokens={avg_fs:>5.0f}  "
            f"avg_input_tokens={avg_in:>5.0f}  total_cost={dollars(cost)}"
        )


if __name__ == "__main__":
    main()
