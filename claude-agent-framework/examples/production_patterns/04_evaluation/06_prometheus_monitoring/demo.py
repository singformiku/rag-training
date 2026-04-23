"""
Prometheus Monitoring — the minimum-viable agent dashboard
==========================================================

What this demo shows
--------------------
A tiny agent (1 LLM call + 1 flaky tool) is instrumented with the nine
counters/histograms recommended by the article:

    llm_calls_total{model,status}
    llm_tokens_total{model,kind}        # kind ∈ in|out|cache_read
    llm_cost_usd_total{model,user}
    llm_latency_seconds{model}
    tool_calls_total{tool,status}
    tool_latency_seconds{tool}
    agent_tasks_total{status}
    agent_inflight
    user_feedback_total{sentiment}

After ~20 simulated requests, the demo:
  1. Serves Prometheus metrics on http://localhost:9464/metrics
  2. Prints sample PromQL queries you can paste into Grafana
  3. Prints the raw ``/metrics`` output locally

Run
---
    python demo.py
    python demo.py --requests 50 --port 9465
    curl http://localhost:9464/metrics
"""
from __future__ import annotations

import argparse
import functools
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from prometheus_client import (Counter, Gauge, Histogram, generate_latest,
                               start_http_server)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import BACKEND, MODELS, banner, chat  # noqa: E402

LLM_CALLS  = Counter("llm_calls_total",        "LLM calls",              ["model", "status"])
LLM_TOKENS = Counter("llm_tokens_total",       "LLM tokens",             ["model", "kind"])
LLM_USD    = Counter("llm_cost_usd_total",     "LLM cost USD",           ["model", "user"])
LLM_LAT    = Histogram("llm_latency_seconds",  "LLM latency",            ["model"],
                        buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120))
TOOL_CALLS = Counter("tool_calls_total",       "Tool invocations",       ["tool", "status"])
TOOL_LAT   = Histogram("tool_latency_seconds", "Tool latency",           ["tool"])
TASK_OK    = Counter("agent_tasks_total",      "Agent task outcomes",    ["status"])
INFLIGHT   = Gauge   ("agent_inflight",        "In-flight tasks")
FEEDBACK   = Counter("user_feedback_total",    "User feedback",          ["sentiment"])


PRICING = {
    "claude-haiku-4-5":   {"in": 1.00, "out": 5.00},
    "claude-sonnet-4-5":  {"in": 3.00, "out": 15.00},
    "claude-opus-4-5":    {"in": 5.00, "out": 25.00},
    "gpt-oss-120b":       {"in": 3.00, "out": 15.00},
}


def price_of(model: str, tokens_in: int, tokens_out: int) -> float:
    p = PRICING.get(model, {"in": 3.00, "out": 15.00})
    return (tokens_in * p["in"] + tokens_out * p["out"]) / 1e6


def trace_llm(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args, user: str = "anon", **kwargs):
        t0 = time.monotonic()
        try:
            r = fn(*args, **kwargs)
            LLM_CALLS.labels(r.model, "ok").inc()
            LLM_TOKENS.labels(r.model, "in").inc(r.input_tokens)
            LLM_TOKENS.labels(r.model, "out").inc(r.output_tokens)
            LLM_TOKENS.labels(r.model, "cache_read").inc(r.cache_read_tokens)
            LLM_USD.labels(r.model, user).inc(price_of(r.model, r.input_tokens, r.output_tokens))
            return r
        except Exception:
            LLM_CALLS.labels(kwargs.get("model") or "?", "error").inc()
            raise
        finally:
            LLM_LAT.labels(kwargs.get("model") or MODELS["cheap"]).observe(time.monotonic() - t0)
    return wrapper


@trace_llm
def chat_tracked(**kw):
    return chat(**kw)


def fake_tool(name: str, ok_rate: float) -> str:
    t0 = time.monotonic()
    time.sleep(random.uniform(0.02, 0.15))
    ok = random.random() < ok_rate
    TOOL_LAT.labels(name).observe(time.monotonic() - t0)
    TOOL_CALLS.labels(name, "ok" if ok else "error").inc()
    if not ok:
        raise RuntimeError("simulated upstream error")
    return "tool_ok_result"


@contextmanager
def track_task():
    INFLIGHT.inc()
    status = "success"
    t0 = time.monotonic()
    try:
        yield
    except Exception:
        status = "fail"
        raise
    finally:
        TASK_OK.labels(status).inc()
        INFLIGHT.dec()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--port", type=int, default=9464)
    parser.add_argument("--tool-ok-rate", type=float, default=0.85)
    args = parser.parse_args()

    banner(f"Prometheus monitoring demo  |  backend={BACKEND}  model={MODELS['cheap']}")
    start_http_server(args.port)
    print(f"metrics server listening on  http://localhost:{args.port}/metrics")

    users = ["alice", "bob", "charlie"]
    for i in range(args.requests):
        user = random.choice(users)
        with track_task():
            r = chat_tracked(
                messages=[
                    {"role": "system", "content": "Answer with ONE short sentence."},
                    {"role": "user", "content": f"Sample question #{i}: what is {i}+{i}?"},
                ],
                tier="cheap",
                max_tokens=40,
                user=user,
                extra={"reasoning_effort": "low"},
            )
            try:
                fake_tool("lookup", args.tool_ok_rate)
            except Exception:
                pass  # recorded as tool error counter
            FEEDBACK.labels(random.choice(["positive", "positive", "positive", "negative"])).inc()
        if (i + 1) % 5 == 0:
            print(f"  processed {i+1}/{args.requests}")

    banner("Example PromQL queries")
    print(
        "# task success rate (5m)\n"
        '  sum(rate(agent_tasks_total{status="success"}[5m]))\n'
        "  / sum(rate(agent_tasks_total[5m]))\n\n"
        "# tool failure rate per tool\n"
        '  sum by (tool) (rate(tool_calls_total{status="error"}[5m]))\n'
        "  / sum by (tool) (rate(tool_calls_total[5m]))\n\n"
        "# cost per user in the last hour\n"
        "  sum by (user) (increase(llm_cost_usd_total[1h]))\n\n"
        "# p95 LLM latency per model\n"
        "  histogram_quantile(0.95, rate(llm_latency_seconds_bucket[5m]))"
    )

    banner("Current /metrics sample (first 20 lines)")
    out = generate_latest().decode().splitlines()
    for line in out[:20]:
        print(line)
    print("...")

    print("\nServer still running.  Try in another shell:")
    print(f"  curl http://localhost:{args.port}/metrics")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
