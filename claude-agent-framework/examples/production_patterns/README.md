# Production Agent Patterns — runnable demos

Companion demos for the report **"Production AI agent engineering, beyond
tutorials"** (`../../../Production AI agent engineering beyond turoials.md`).

Each subfolder contains a self-contained `demo.py`; run it from anywhere with:

```bash
# from the repo root
/root/rag-training/.venv/bin/python claude-agent-framework/examples/production_patterns/<topic>/<technique>/demo.py
```

## Backend selection

Every demo imports from the shared `_common.backend` helper, which picks:

* **Anthropic SDK** when `ANTHROPIC_API_KEY` is set (real Claude models:
  `claude-haiku-4-5`, `claude-sonnet-4-5`, `claude-opus-4-5` — override via
  `ANTHROPIC_MODEL_CHEAP` / `_MEDIUM` / `_BIG`).
* **`llm_service.py`** fallback — your internal OpenAI-compatible endpoint
  (`gpt-oss-120b`).  All three tiers collapse to the one model; tier selection
  becomes `reasoning_effort="low|medium|high"` instead.

Run `python -c "from _common.backend import BACKEND, MODELS; print(BACKEND, MODELS)"`
from the `production_patterns/` folder to see which backend was picked.

## Dependencies

Installed from the repo-root `requirements.txt`:

```bash
pip install -r /root/rag-training/requirements.txt
```

Additional packages this folder expects: `anthropic`, `openai`, `tiktoken`,
`aiolimiter`, `tenacity`, `prometheus-client`, `langfuse`,
`sentence-transformers` (for embedding fallback).

## Demo index

### Topic 1 — Token optimization

| # | Folder | What it demonstrates |
|---|---|---|
| 1.1 | `01_token_optimization/01_prompt_caching/` | `cache_control` breakpoints + OpenAI auto prefix caching, with $ and latency deltas. |
| 1.2 | `01_token_optimization/02_context_compression/` | Summarise-on-overflow vs rolling window vs full history. |
| 1.3 | `01_token_optimization/03_semantic_memory/` | k-NN retrieval over past turns, recall ↑, tokens ↓. |
| 1.4 | `01_token_optimization/04_dynamic_tool_loading/` | Router-first selection, tool schemas 1,700 → ~320 tokens. |
| 1.5 | `01_token_optimization/05_structured_output/` | Freeform vs `json_object` vs `json_schema` strict mode. |
| 1.6 | `01_token_optimization/06_dynamic_few_shot/` | k-NN over an example bank beats static 8-shot. |

### Topic 2 — Model routing

| # | Folder | What it demonstrates |
|---|---|---|
| 2.1 | `02_model_routing/01_cascade_routing/` | Cheap → verify → escalate, with break-even threshold math. |
| 2.2 | `02_model_routing/02_task_classifier/` | Single cheap classifier picks the right tier. |
| 2.3 | `02_model_routing/03_speculative_execution/` | Parallel cheap + expensive race, cancel the loser. |

### Topic 3 — Performance engineering

| # | Folder | What it demonstrates |
|---|---|---|
| 3.1 | `03_performance/01_parallel_tool_calls/` | `asyncio.gather` speedup for N tools per turn. |
| 3.2 | `03_performance/02_streaming/` | TTFT vs total latency, blocking vs streaming. |
| 3.3 | `03_performance/03_budget_early_termination/` | Iteration/wall-clock/token caps vs runaway loop. |
| 3.4 | `03_performance/04_failure_recovery/` | Naive vs retry vs circuit breaker vs self-heal. |
| 3.5 | `03_performance/05_context_management/` | Sliding window, summarise-on-overflow, prefix cache. |
| 3.6 | `03_performance/06_concurrency_rate_limiting/` | Unbounded vs semaphore vs token bucket. |

### Topic 4 — Evaluation & observability

| # | Folder | What it demonstrates |
|---|---|---|
| 4.1 | `04_evaluation/01_eval_harness/` | pass@k, pass^k, mean score across trials. |
| 4.2 | `04_evaluation/02_langfuse_tracing/` | Nested spans for agent + LLM + tool calls. |
| 4.3 | `04_evaluation/03_cost_attribution/` | contextvars-backed per-user/task cost buckets. |
| 4.4 | `04_evaluation/04_semantic_snapshots/` | Exact → cosine → LLM-judge regression testing. |
| 4.5 | `04_evaluation/05_llm_judge/` | Position-swap + ensemble for bias-mitigated judging. |
| 4.6 | `04_evaluation/06_prometheus_monitoring/` | The 9 core metrics on `:9464/metrics`. |

## What was fixed vs the source report

The original article contains:

* **Fictional 2026 models** (`claude-opus-4-5`, `gpt-5`, `o3` …) that the
  demos here map to tier aliases so the same code runs on real models today.
* **Fragment snippets** with undefined variables (`TOOLS = [...]`,
  `load_tools()`, `tool_impls`, `env`, `task.graders`) — each demo is now a
  full `main()` with deterministic inputs + measurable outputs.
* **Anthropic-only primitives** (`cache_control`, `tool_use` content blocks,
  fine-grained tool streaming) — the shared backend adapts each one to the
  equivalent OpenAI Chat Completions feature so the internal endpoint works.
* **Hardcoded credentials** (Langfuse keys, Anthropic keys) — all read from
  env vars with graceful no-op fallbacks.
* **Unbounded loops / missing retries** in the agent loop code — bounded by
  `Budget` and wrapped with tenacity where relevant.

Read the per-demo module docstring for the precise technique each one shows.
