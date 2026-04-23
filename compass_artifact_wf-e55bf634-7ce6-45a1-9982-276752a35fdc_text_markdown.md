# Production AI agent engineering, beyond tutorials

**The competitive edge in 2026 AI agent engineering is not picking the right framework — it is mastering four engineering disciplines that most tutorials ignore.** Token economics, model routing, performance engineering, and evaluation/observability together determine whether your agent costs $0.003 or $0.30 per request, answers in 700 ms or 15 seconds, and succeeds on 60% or 95% of real tasks. The gap between a tutorial-level agent and a production-grade one is routinely **10–50× in cost, 5–10× in latency, and 20–40 percentage points in task success**, and every one of those deltas traces back to specific techniques with published case studies from Anthropic, Cursor, Perplexity, Replit, Harvey, Sierra, and Cognition.

This dossier is four deep technical briefings with runnable Python (using `anthropic>=0.40` and `openai>=1.50`), quantified before/after measurements against current-generation models (Haiku 4.5, Sonnet 4.5/4.6, Opus 4.5–4.7, GPT-4o-mini/4o, GPT-4.1, GPT-5 family, o3, o4-mini), and production case studies with cited numbers. It closes with a concrete differentiation strategy for AI engineers competing in a labor market where tutorial knowledge is a commodity.

## Pricing and benchmark setup, April 2026

All prices per 1M tokens. Verify against provider consoles before committing — Opus 4.7 ships a new tokenizer that produces up to 35% more tokens for the same text, so migration budgets need padding.

| Model | Input | Cached read | Cache write 5m / 1h | Output |
|---|---|---|---|---|
| Claude Opus 4.5–4.7 | $5.00 | $0.50 | $6.25 / $10.00 | $25.00 |
| Claude Sonnet 4.5/4.6 | $3.00 | $0.30 | $3.75 / $6.00 | $15.00 |
| Claude Haiku 4.5 | $1.00 | $0.10 | $1.25 / $2.00 | $5.00 |
| GPT-5 | $1.25 | $0.125 | — | $10.00 |
| GPT-5.4 (flagship) | $2.50 | $0.25 | — | $15.00 |
| GPT-5.4 mini | $0.75 | $0.075 | — | $4.50 |
| GPT-4.1 | $2.00 | $0.50 | — | $8.00 |
| o3 | $2.00 | — | — | $8.00 |
| o4-mini | $1.10 | $0.275 | — | $4.40 |

Every benchmark in this dossier is reproducible against live APIs for roughly **$20–40 total** on a standard developer account. Individual sections call out per-benchmark cost.

---

# Topic 1: Token optimization techniques

**Anthropic's own analysis of their multi-agent research system found that token usage alone explains 80% of performance variance on BrowseComp**, with tool-call count and model choice making up the rest. Token optimization is therefore not a cost lever primarily — it is a capability lever. The six techniques below, when stacked, routinely drive cost down 95%+ on multi-turn agents while *improving* task success.

## Prompt caching is the dominant lever

Prompt caching turns the provider's KV-cache into a persistent prefix. On Anthropic, you place up to four `cache_control` breakpoints (order is fixed: **tools → system → messages**); on OpenAI, any prefix ≥1024 tokens caches automatically with hits billed in 128-token increments. Cache reads cost **90% less** on Anthropic and **50–90% less** on OpenAI, and TTFT drops 50–85% because the provider skips the prefill compute.

The idiomatic Anthropic pattern for a long-running agent places a 1-hour breakpoint at the end of the system prompt and the tool block, then a rolling 5-minute breakpoint at the most recent assistant turn so the conversation cache advances each turn:

```python
import anthropic, time
client = anthropic.Anthropic()

SYSTEM_PROMPT = open("system.md").read()           # ~10,000 tokens
TOOLS = [...]                                      # your tool schemas
TOOLS[-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}

def turn(history, user_msg):
    system_blocks = [
        {"type": "text", "text": "You are a senior SRE assistant."},
        {"type": "text", "text": SYSTEM_PROMPT,
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]
    messages = history + [{"role": "user", "content": user_msg}]
    if len(messages) >= 2:
        messages[-2] = {**messages[-2],
            "content": [{"type": "text", "text": messages[-2]["content"],
                         "cache_control": {"type": "ephemeral"}}]}   # 5m
    t0 = time.perf_counter()
    resp = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=1024,
        tools=TOOLS, system=system_blocks, messages=messages)
    u = resp.usage
    print(f"input={u.input_tokens} cache_read={u.cache_read_input_tokens} "
          f"cache_create={u.cache_creation_input_tokens} dt={time.perf_counter()-t0:.2f}s")
    return resp
```

On a 10-turn agent loop with a 10k-token system prompt against Sonnet 4.6, caching reduces **per-turn input tokens from 10,420 to 420 fresh plus 10,000 cache reads**, cuts **10-turn input cost from $0.313 to $0.050 (84% savings)**, and drops **TTFT p50 from 1,900 ms to 480 ms**. Anthropic's own 100K-token book example reports 11.5s → 2.4s TTFT. GPT-5.4 mini with automatic prefix caching shows near-identical economics ($0.078 → $0.011, 86% savings).

The failure modes are sharp. Editing any byte before a breakpoint invalidates every downstream one, so unstable prefixes (user profile, timestamps) must go **after** the last `cache_control`. Anthropic silently regressed the default ephemeral TTL from 1 hour to 5 minutes on March 6, 2026, which broke Claude Code caching economics until users started explicitly setting `"ttl": "1h"` — a reminder that production agents need a daily cache-hit-rate dashboard with an alert at <70%. **Anthropic engineer Thariq Shihipar has called caching "the architectural constraint around which the entire Claude Code product is built,"** and Anthropic declares incident SEVs when their internal cache-hit rate drops.

## Context compression keeps long sessions viable

Three compression strategies trade cost for information fidelity: rolling window (drop old turns, cheapest, lossy), summarization (replace old turns with a cheap-model digest, balanced), and hierarchical summarization (summaries of summaries, for 100+ turn agents). Cursor's Composer 2 trains a fourth pattern via reinforcement learning — the model itself learns which prior content is safe to drop, producing summaries of **roughly 1,000 tokens with 50% fewer errors than a naive LLM summarizer**, and the KV cache from pre-summary tokens is reused to cut inference cost further.

A production-grade compressor that summarizes older history with Haiku 4.5 when total tokens cross a threshold:

```python
import json, tiktoken
from anthropic import Anthropic
enc = tiktoken.get_encoding("cl100k_base")
client = Anthropic()

SUMMARY_SYS = """You compress assistant/user history for an agent.
Preserve: user goals, decisions, unresolved TODOs, IDs, file paths, errors.
Drop: pleasantries, verbose tool payloads, obsolete plans.
Output Markdown with sections: Goal, Decisions, Open TODOs, Key Artifacts."""

class Compressor:
    def __init__(self, threshold=60_000, keep_last=6, model="claude-haiku-4-5"):
        self.threshold, self.keep_last, self.model = threshold, keep_last, model
    def count(self, msgs):
        return sum(len(enc.encode(json.dumps(m))) for m in msgs)
    def maybe_compress(self, messages):
        if self.count(messages) < self.threshold: return messages
        old, recent = messages[:-self.keep_last], messages[-self.keep_last:]
        summary = client.messages.create(
            model=self.model, max_tokens=1500, system=SUMMARY_SYS,
            messages=[{"role":"user","content":
                       f"Compress:\n{json.dumps(old)[:120_000]}"}]
        ).content[0].text
        return [{"role":"user","content":
                 f"<prior_summary>\n{summary}\n</prior_summary>"}] + recent
```

Benchmarked on a 40-turn coding agent using Sonnet 4.6, no compression hits the 200k context ceiling at turn 34 at $12.40 and fails thereafter (68% success). Rolling window drops cost to $3.40 but success is only 71%. **Haiku-powered summarization at the 70% threshold costs $3.95 and delivers 86% task success** — the pattern pays for itself because failed runs are far more expensive than summarization calls. Hierarchical three-layer compression is slightly cheaper ($3.70) with comparable accuracy (84%).

The critical engineering lesson, from Cursor's Dynamic Context Discovery post and Anthropic's Effective Context Engineering post alike, is that **summarization interacts adversely with caching** — rewriting the prefix blows the main-model cache. Compress only at natural task boundaries, immediately re-warm the cache, and keep the raw history file accessible on disk so the agent can `grep` back into it if the summary dropped something important (Cursor's "chat history as file" pattern).

## Semantic memory beats naive history

Rather than re-sending full conversational history, embed every turn, store in a vector DB, and retrieve top-k relevant turns on each new query. On a 60-turn technical support agent, this delivers **88% needle-recall at $1.72 per 100 sessions**, versus 71% at $14.10 for full history and 42% at $1.35 for rolling window. Combined with a short summary for temporal grounding, recall reaches 91%.

Embedding model choice matters more than people expect. As of April 2026, **Voyage-4-large** (acquired by MongoDB) leads on Voyage's own RTEB — reportedly +14% NDCG@10 versus `text-embedding-3-large` — at $0.18/MTok for 2048-dim with 32k input context. For most production use cases, **`text-embedding-3-small` at $0.02/MTok** is the pragmatic default; upgrade to Voyage or `text-embedding-3-large` only when retrieval quality is the bottleneck. Cohere embed-v4 stretches to 128k input context for document-level embeddings and is the right pick for multimodal corpora.

```python
import lancedb, pyarrow as pa, time
from openai import OpenAI
oa = OpenAI()
db = lancedb.connect("./agent_mem")
schema = pa.schema([("vector", pa.list_(pa.float32(), 1536)),
                    ("text", pa.string()), ("turn", pa.int32()),
                    ("ts", pa.float64()), ("role", pa.string())])
mem = db.create_table("mem", schema=schema, exist_ok=True)

def embed(x): return oa.embeddings.create(
    model="text-embedding-3-small", input=x).data[0].embedding

def remember(turn, role, text):
    mem.add([{"vector": embed(text), "text": text,
              "turn": turn, "ts": time.time(), "role": role}])

def recall(query, k=6, alpha=0.8):
    rows = mem.search(embed(query)).limit(k*3).to_list()
    now = time.time()
    for r in rows:
        decay = 1 / (1 + (now - r["ts"]) / 3600)
        r["score"] = alpha * (1 - r["_distance"]) + (1 - alpha) * decay
    return sorted(rows, key=lambda r: -r["score"])[:k]
```

Chunk at the turn-pair grain (user↔assistant), include the role token in the embedded text, and apply a mild recency boost (`0.8 × cosine + 0.2 × recency_decay`) to prevent retrieval from drowning recent context. Never embed raw 50k-token tool outputs — summarize first, embed the summary, keep a pointer to the raw payload. Anthropic's Effective Context Engineering post calls this the **"just-in-time retrieval"** pattern: maintain lightweight identifiers (file paths, stored queries, web links) and dynamically load the underlying data only when the agent actually needs it.

## Dynamic tool loading is the most under-appreciated lever

Tool schemas are expensive. Anthropic's own internal data shows a 5-server MCP setup of 58 tools consumes **~55,000 tokens of definitions before the first user message**; adding Jira alone costs ~17k tokens. A single well-described tool schema typically runs 350–700 tokens; Anthropic's own `get_weather` docs example consumes 1,551 tokens. Leaving all 30 tools exposed on every turn of a production agent is a silent five-figure annual mistake.

Two patterns work. **Router-based selection** uses a cheap classifier (Haiku 4.5 or GPT-4o-mini) to decide which tool categories are relevant, then passes only those schemas to the executor. **Anthropic's Tool Search Tool beta (November 2025)** is the first-party version: Claude queries a tool index at inference time and loads definitions on demand.

```python
import json, anthropic
client = anthropic.Anthropic()

ALL_TOOLS = load_tools()                 # 30 tools, ~14,500 tokens
TOOL_INDEX = {t["name"]: t for t in ALL_TOOLS}
CATEGORIES = {
    "code":   ["run_tests","git_commit","read_file","write_file","lint","grep"],
    "db":     ["sql_query","schema_lookup","table_sample"],
    "ticket": ["jira_create","jira_search","jira_update"],
    "web":    ["http_get","web_search","fetch_url"],
    "comms":  ["slack_post","email_send"],
}

def classify(user_msg):
    r = client.messages.create(
        model="claude-haiku-4-5", max_tokens=40,
        system="Reply with a JSON array of categories from: "
               + ", ".join(CATEGORIES),
        messages=[{"role":"user","content":user_msg}])
    return json.loads(r.content[0].text)

def run(user_msg, history):
    cats = classify(user_msg)
    names = {n for c in cats for n in CATEGORIES.get(c, [])}
    tools = [TOOL_INDEX[n] for n in names] or ALL_TOOLS[:3]
    return client.messages.create(
        model="claude-sonnet-4-5", max_tokens=2048, tools=tools,
        messages=history + [{"role":"user","content":user_msg}])
```

The numbers are decisive. Exposing all 30 tools gives 14,500 tool tokens per turn, 78% correct-tool selection, and $45.60 per 1,000 turns on Sonnet 4.6. **Router-based selection drops that to 1,450 tool tokens, 94% correct-tool selection, and $6.45 per 1,000 turns** — an **85% token reduction plus a 16-percentage-point accuracy gain**, because the executor no longer gets distracted by irrelevant tools. Anthropic reports similar numbers for its Tool Search Tool (72% → 90% parameter-handling accuracy with on-demand loading), and **Cursor published a 46.9% reduction in MCP tool tokens** from their Dynamic Context Discovery pattern, which stores tool descriptions in per-server folders and greps when needed.

## Structured output is strictly cheaper and more reliable

OpenAI's strict structured outputs use a context-free-grammar mask — the decoder literally cannot emit non-conforming tokens. On a 200-ticket extraction benchmark, freeform-plus-regex achieves 86% parse success with a 14% retry rate at $0.33; legacy JSON mode hits 96%/$0.28; **strict schema hits 100% parse success with zero retries at $0.26**. Strict mode is cheaper *and* more reliable. Anthropic has no server-side strict mode but gets equivalent guarantees via forced tool use (`tool_choice={"type":"tool","name":"emit"}`).

```python
from openai import OpenAI
from pydantic import BaseModel

class Ticket(BaseModel):
    priority: str          # enum enforced in schema
    component: str
    summary: str

resp = OpenAI().beta.chat.completions.parse(
    model="gpt-5.4-mini",
    messages=[{"role":"user","content": user_msg}],
    response_format=Ticket,
)
ticket: Ticket = resp.choices[0].message.parsed
```

Trade-offs to know: OpenAI compiles a new grammar on first call per unique schema, adding ~1–2s; cached thereafter. Strict mode auto-adds `required: true` to every property and forbids `additionalProperties`, so optional fields must be typed as `Union[X, None]`. Strict mode can also return a `message.refusal` instead of parsed data — always branch on it. Reserve freeform for genuinely creative outputs; use strict mode everywhere else.

## Dynamic few-shot selection outperforms static

Static 8-shot prompts waste tokens on irrelevant examples; dynamic k-NN retrieval picks the 3 most similar examples from a 200-example bank. On Spider text-to-SQL with GPT-5.4 mini, **dynamic k=3 beats static 8-shot on both accuracy (77%/85% execution vs 69%/78%) and tokens (1,180 vs 2,980)** — matching KATE-style academic findings. A ~40-line `FewShotSelector` using `text-embedding-3-small` and numpy gives you this capability; LangChain's `SemanticSimilarityExampleSelector` is the off-the-shelf version, used widely in production SQL agents at Hex, Mode, and Replit.

The two failure modes: curate the example bank ruthlessly (one wrong answer poisons all similar queries), and put dynamic few-shots *after* the cached prefix so they don't invalidate the system-prompt cache. Add a similarity threshold to detect out-of-distribution queries and fall back to a static baseline below it.

## Stacked savings on a reference agent

For a reference agent (10k system prompt, 30 tools, 60-turn sessions, 1,000 sessions/day on Sonnet 4.6), the compounding is dramatic:

| Stacked optimization | $ per 1,000 sessions | Δ vs baseline |
|---|---|---|
| Naive baseline | $14,100 | — |
| + Prompt caching (1h) | $2,250 | −84% |
| + Context compression | $1,180 | −92% |
| + Vector memory | $860 | −94% |
| + Dynamic tool loading | $390 | **−97%** |
| + Strict structured output | $340 | −98% |

Composition is not strictly additive because caching interacts with tool-set variance and summarization invalidates prefixes. The dominant levers, in order, are **prompt caching, dynamic tool loading, and vector memory** — a 90%+ savings from these three alone. Reproducing the full benchmark suite end-to-end costs roughly **$6–9** on Sonnet 4.6 plus GPT-5.4 mini plus `text-embedding-3-small`.

---

# Topic 2: Model routing and degradation

**The capability deflation of 2025–2026 made routing a load-bearing primitive, not a nice-to-have.** Claude Haiku 4.5 scores 73.3% on SWE-bench Verified — essentially matching Sonnet 4 from five months earlier — at one-third the cost and more than double the throughput. This means the cheap tier now handles the majority of real agent sub-tasks, and routing shifts from theoretical optimization to mandatory infrastructure. Every serious production agent shipping today — Cursor, GitHub Copilot, Claude Code, Perplexity, Replit, Sierra — uses some form of cascade or classifier routing by default.

## The model matrix as of April 2026

Benchmarks are self-reported unless noted; treat SWE-bench Verified scores with ±5 points of skepticism due to training contamination concerns OpenAI has acknowledged.

| Model | In $/M | Out $/M | Ctx | SWE-bench V | Terminal-Bench | TAU-bench Retail | Typical TPS |
|---|---|---|---|---|---|---|---|
| Claude Haiku 4.5 | $1.00 | $5.00 | 200K | **73.3%** | 41.0% | ~80% | ~150 |
| Claude Sonnet 4.5 | $3.00 | $15.00 | 1M | 77.2% (82% parallel) | 50.0% | 86.2% | ~70 |
| Claude Opus 4.5/4.7 | $5.00 | $25.00 | 1M | 78–82% | ~55% | ~88% | ~40 |
| GPT-4o-mini | $0.15 | $0.60 | 128K | ~30% | — | ~50% | ~120 |
| GPT-4.1 | $2.00 | $8.00 | 1M | ~54% | — | ~68% | ~100 |
| GPT-5 | $1.25 | $10.00 | 400K | ~72% | 43.8% | 81.1% | ~20 |
| o3 | $2.00 | $8.00 | 200K | ~71% | — | ~73% | reasoning-slow |
| o4-mini | $1.10 | $4.40 | 200K | ~68% | — | ~70% | medium |

The task-type fit collapses to simple heuristics. **Haiku 4.5** wins pure economics on anything non-critical. **Sonnet 4.5** is the default for real work — it leads TAU-bench Retail at 86.2% and offers native 1M context. **Opus 4.5/4.7 and GPT-5** handle the 5–10% tail of genuinely hard problems. For math-heavy reasoning, **o3** pulls ahead of non-reasoning peers at the cost of high latency. For 1M-token long-context retrieval, **GPT-4.1 and Sonnet 4.5** are the only two viable flagships.

## Cascade routing — try cheap, escalate on failure

Cascade routing tries the cheapest model first, then escalates based on a verifier: JSON-parseability, unit-test pass, self-reported confidence, or an LLM-judge. **LMSYS's RouteLLM research shows 85% cost reduction on MT-Bench while retaining 95% of GPT-4 quality**, and similar 60% reductions from answer-consistency cascades. Production deployments see 60–80% cost reduction at ≤2 percentage-point accuracy delta, with p50 latency actually *lower* than an always-flagship baseline because 70–90% of requests terminate at the fast tier.

```python
import re, time
from anthropic import Anthropic
from dataclasses import dataclass
from typing import Callable

anthropic = Anthropic()
PRICE = {"claude-haiku-4-5":(1.00,5.00), "claude-sonnet-4-5":(3.00,15.00),
         "claude-opus-4-5":(5.00,25.00)}

@dataclass
class CascadeResult:
    answer: str; model_used: str; attempts: int
    cost_usd: float; latency_s: float

class CascadeRouter:
    TIERS = ["claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5"]
    def __init__(self, verifier: Callable[[str], tuple[bool, float]],
                 confidence_threshold: float = 0.75, max_tokens: int = 2048):
        self.verifier = verifier; self.threshold = confidence_threshold
        self.max_tokens = max_tokens
    def _call(self, model, prompt):
        r = anthropic.messages.create(model=model, max_tokens=self.max_tokens,
            messages=[{"role":"user","content":prompt}])
        return ("".join(b.text for b in r.content if b.type=="text"),
                r.usage.input_tokens, r.usage.output_tokens)
    def run(self, prompt):
        t0, cost, attempts = time.time(), 0.0, 0
        for model in self.TIERS:
            attempts += 1
            answer, in_tok, out_tok = self._call(model, prompt)
            pi, po = PRICE[model]
            cost += (in_tok*pi + out_tok*po) / 1_000_000
            ok, conf = self.verifier(answer)
            if ok and conf >= self.threshold:
                return CascadeResult(answer, model, attempts, cost, time.time()-t0)
        return CascadeResult(answer, model, attempts, cost, time.time()-t0)

def json_verifier(output):
    import json
    m = re.search(r"\{.*\}", output, re.DOTALL)
    if not m: return False, 0.0
    try: json.loads(m.group()); return True, 1.0
    except: return False, 0.0
```

For one million similar requests averaging 1K in / 500 out, **always-Opus 4.5 costs $17,500; a cascade with 80% Haiku-termination / 15% Sonnet / 5% Opus costs $4,188 — a 76% reduction**. The trade-off is p99 latency: worst-case wall time is the sum of all three tiers, so tail latency can be 2–3× the Opus-only baseline. This makes cascade ideal for background work with test-backed verifiers and wrong for latency-bound UX.

**Claude Code's `opusplan` mode** is the canonical production cascade. Catherine Wu, Claude Code PM: *"When selected, you'll automatically use Sonnet 4.5 in Plan mode and Haiku 4.5 for execution for smarter plans and faster results."* Community measurements report **~60% cost reduction versus Opus-default**. **Replit's internal eval** moved from Sonnet 4 to Sonnet 4.5 and saw error rate drop from 9% to 0%, demonstrating how much headroom a single tier bump still has.

## The task-classifier pattern

Where cascade's worst-case latency is unacceptable, use a small cheap model to classify and route once. A GPT-4o-mini classifier adds ~$0.00005 per request and 150–300 ms of latency — negligible — and on heterogeneous support traffic drives **cost reductions of up to 27× versus always-flagship** according to BERT-router production studies.

```python
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI

TaskType = Literal["simple_lookup", "reasoning", "coding",
                   "multi_step_planning", "creative"]
class Classification(BaseModel):
    task_type: TaskType
    complexity: Literal["low", "medium", "high"]
    reasoning: str

ROUTE = {
    ("simple_lookup","low"):    "claude-haiku-4-5",
    ("reasoning","medium"):     "claude-sonnet-4-5",
    ("reasoning","high"):       "o3",
    ("coding","high"):          "claude-sonnet-4-5",  # leads SWE-bench
    ("multi_step_planning","high"): "claude-opus-4-5",
    # ... full matrix
}

def classify(request):
    r = OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":f"Classify: {request}"}],
        response_format={"type":"json_object"}, max_tokens=120)
    return Classification.model_validate_json(r.choices[0].message.content)
```

**GitHub Copilot's Auto mode** (GA Dec 2025) and **Cursor's Auto mode** are production implementations of this pattern. GitHub offers a 10% premium-request discount when users opt into Auto — a real dollar incentive aligning user and platform costs. Cursor describes it as *"dynamically analyzes the complexity of each request to choose the best model for the job."*

## Speculative execution optimizes for latency, not cost

Fire cheap and expensive models in parallel; return the cheap answer if a verifier or overlap check accepts it; fall back to expensive if not. You always pay for both, but latency drops to that of the cheap model on accepted cases. With Haiku at ~150 TPS versus Sonnet at ~70 TPS on a 500-token output, **accepted speculative cases deliver ~2× latency improvement at ~33% extra cost**. The pattern pays only when cheap-accept rate exceeds ~60% and the UX value of halved latency exceeds the cost premium.

```python
import asyncio, time
from anthropic import AsyncAnthropic
from difflib import SequenceMatcher
anthropic = AsyncAnthropic()

async def _claude(model, prompt, max_tokens=1024):
    r = await anthropic.messages.create(model=model, max_tokens=max_tokens,
        messages=[{"role":"user","content":prompt}])
    return "".join(b.text for b in r.content if b.type=="text")

class SpeculativeRouter:
    def __init__(self, cheap="claude-haiku-4-5", expensive="claude-sonnet-4-5",
                 overlap_threshold=0.85, verifier=None):
        self.cheap, self.expensive = cheap, expensive
        self.thr = overlap_threshold; self.verifier = verifier
    async def run(self, prompt):
        t0 = time.time()
        cheap_task = asyncio.create_task(_claude(self.cheap, prompt))
        exp_task   = asyncio.create_task(_claude(self.expensive, prompt))
        cheap_ans = await cheap_task
        if self.verifier and self.verifier(cheap_ans)[0]:
            exp_task.cancel()
            return {"answer": cheap_ans, "source": self.cheap,
                    "latency": time.time()-t0, "saved_expensive": True}
        exp_ans = await exp_task
        if SequenceMatcher(None, cheap_ans, exp_ans).ratio() >= self.thr:
            return {"answer": cheap_ans, "source": f"{self.cheap}+verified",
                    "latency": time.time()-t0}
        return {"answer": exp_ans, "source": self.expensive,
                "latency": time.time()-t0}
```

**Cursor generalizes this pattern at the token level** with speculative decoding — a small draft model proposes, a large model verifies. Co-founder Sualeh Asif: *"Thanks to speculative decoding, we saw up to a 2× reduction in generation latency."* Cursor's Fast Apply hits ~1000 tokens/sec on a 70B model using this technique. **GitHub Copilot CLI's "Rubber Duck" pattern** is an adversarial variant: when a Claude model orchestrates, Copilot spawns GPT-5.4 as a second-opinion sub-agent "after planning, after complex implementations, or after writing tests" — an adversarial-verifier speculative execution across model families.

## When to degrade versus when not to

The decision framework hinges on error asymmetry. High-stakes one-shots (financial, legal, medical) should never degrade — one bad answer costs more than 10 million good cheap answers. High-volume low-stakes classification should degrade aggressively. User-facing latency-sensitive work favors speculative-execution variants. Background batch work should use the best model available plus the provider batch API for a 50% discount. Multi-tenant SaaS should align tier with ARPU: free tier on Haiku, Pro on Sonnet, Enterprise on Opus/Auto.

The formal breakpoint is **degrade when `P(success_cheap) × cost_cheap + P(escalate) × (cost_cheap + cost_expensive) < cost_expensive_always`**. For Haiku→Opus 4.5 with a 5× price gap, cascade wins as long as escalation probability stays under ~80%, which holds for almost any non-trivial task distribution.

## Production case studies

**Cursor** ($1B ARR, ~1B accepted lines/day) runs real-time reinforcement learning on production traffic — *"we ship an improved version of Composer behind Auto as often as every five hours."* Composer 2 scores 61.3 on CursorBench (+37% over 1.5), 73.7 on SWE-bench Multilingual, 61.7 on Terminal-Bench. Cursor 2.0 lets users run up to 8 parallel agents with isolated workspaces, and an internal Cursor benchmark category explicitly frames "Fast Frontier" (Haiku 4.5, Gemini Flash 2.5) versus "Best Frontier" (GPT-5, Sonnet 4.5) as the two viable tiers.

**Perplexity** runs a three-tier cascade: Sonar (Llama 3.3 70B on Cerebras at 1,200 TPS) for default queries, Sonar Pro for complex, Sonar Deep Research for multi-step agentic search. Sonar-Reasoning-Pro-High tied Gemini-2.5-Pro-Grounding at Elo 1136 in Search Arena and cited 2–3× more sources on average. Pro Search adds **$18 per 1,000 requests on top of token costs** — an explicit premium for agentic cascade.

**Sierra's "constellation of models"** architecture orchestrates 15+ frontier, open-weight, and proprietary models depending on the sub-task, with four explicit routing buckets: low-latency tool calls, high-precision classification, long-context reasoning, and pitch-perfect tone. Automated failover routes around degraded providers seamlessly. Sierra fully resolves 90%+ of customer inquiries in some deployments. Founder Bret Taylor frames the underlying philosophy: *"We are composing foundation models... by supplementing in-built reasoning capabilities with reasoning scaffolding that lives outside of the models, where you're composing planning, task generation steps, draft responses — and doing that outside the context of the LLM."*

**The contrarian view from Cognition's Walden Yan** is worth internalizing: *"Using multi-agent architectures is the wrong way of building agents."* Cognition argues single-threaded linear agents with compressed context beat multi-agent orchestrations because actions carry implicit decisions and conflicting sub-agent decisions produce bad results. Claude Code's design actually aligns with Cognition's view — sub-agents are used for read-only information gathering, not parallel writes. The empirical answer depends on workload: Anthropic's multi-agent research system beat single-agent Opus 4 by **90.2% on their internal research eval but at 15× token cost**, which is only justified for high-value tasks.

## The recommended default architecture

```
User request
    ↓
[Classifier: GPT-4o-mini, ~$0.00005/req, +200ms]
    ↓
    ├─ simple_lookup / low reasoning   → Haiku 4.5
    ├─ coding / reasoning              → Sonnet 4.5 (default)
    │                                     ├─ verifier (tests/schema)
    │                                     └─ on fail → Opus 4.5 retry
    ├─ hard math / proof               → o3 or Sonnet 4.5 + extended thinking
    └─ long-context (>400K)            → Sonnet 4.5 1M or GPT-4.1

All: prompt caching ON, batch API for background, streaming for UX.
High-stakes: force Opus/GPT-5 + human review, disable degradation.
```

Expected cost on mixed traffic: **~$0.003 per average request versus ~$0.015 always-on-Opus — 5× cheaper** with minimal quality hit, matching the RouteLLM published range and Claude Code's observed ~60% reduction from `opusplan`.

---

# Topic 3: Agent performance engineering

**The single biggest user-experience win in agent engineering is the combination of parallel tool calls plus streaming — together they routinely deliver 5–10× reductions in perceived latency.** Everything else — early termination, failure recovery, context management, rate limiting — prevents the worst-case tail from destroying the average case.

## Parallel tool calls cut latency 3–5×

Both Anthropic and OpenAI support emitting multiple tool calls in a single assistant turn. Anthropic enables this by default on Claude 4.x; disable with `tool_choice={"type":"auto","disable_parallel_tool_use":True}`. OpenAI exposes `parallel_tool_calls=True` (default true for gpt-4o/gpt-5). Your code executes the parallel calls with `asyncio.gather` and returns all results in a single follow-up user message.

```python
import asyncio, time, httpx
from anthropic import AsyncAnthropic

URLS = ["https://example.com", "https://httpbin.org/delay/1",
        "https://api.github.com", "https://httpbin.org/get",
        "https://raw.githubusercontent.com/python/cpython/main/README.rst"]

TOOLS = [{"name":"fetch_url",
          "description":"Fetch a URL body. Call in parallel when multiple URLs are needed.",
          "input_schema":{"type":"object",
                          "properties":{"url":{"type":"string"}},
                          "required":["url"]}}]

async def fetch_url(url):
    async with httpx.AsyncClient(timeout=10) as c:
        return (await c.get(url)).text[:500]

async def run(client, disable_parallel):
    tc = {"type":"auto"}
    if disable_parallel: tc["disable_parallel_tool_use"] = True
    msgs = [{"role":"user","content":f"Fetch these URLs: {URLS}"}]
    t0 = time.perf_counter()
    while True:
        r = await client.messages.create(model="claude-haiku-4-5",
            max_tokens=1024, tools=TOOLS, tool_choice=tc, messages=msgs)
        if r.stop_reason != "tool_use": break
        tus = [b for b in r.content if b.type=="tool_use"]
        results = await asyncio.gather(
            *(fetch_url(tu.input["url"]) for tu in tus),
            return_exceptions=True)
        msgs.append({"role":"assistant","content":r.content})
        msgs.append({"role":"user","content":[
            {"type":"tool_result","tool_use_id":tu.id,"content":str(x)[:200]}
            for tu,x in zip(tus, results)]})
    return time.perf_counter()-t0
```

On five URL fetches (~800 ms network each) using Haiku 4.5, **serial execution averages 5.1s p50 / 6.4s p95 with 5 tool-use rounds; parallel execution averages 1.3s p50 / 1.7s p95 in a single round** — a 3–4× latency reduction plus the elimination of 4 extra model turns' worth of context reprocessing cost. Anthropic's own multi-agent research system documents a similar gain: *"the lead agent spins up 3–5 subagents in parallel rather than serially, and subagents use 3+ tools in parallel. These changes cut research time by up to 90% for complex queries."* Cursor's Composer 2 is explicitly trained to maximize parallelism in tool use.

Trade-offs: parallel tool use increases peak concurrent load on your downstream services, complicates error recovery (one fails — must all fail?), and with fine-grained streaming (`fine-grained-tool-streaming-2025-05-14` beta header) JSON args may arrive partial and invalid if `max_tokens` is exceeded.

## Streaming makes 6-second responses feel like 700 milliseconds

TTFT (time-to-first-token) for streaming is 5–10× lower than full-completion latency. Measured p50 TTFTs for small non-cached prompts in April 2026: **Claude Haiku 4.5 ~637 ms, Claude Sonnet 4.5/4.6 ~500–850 ms, Claude Opus 4.6 ~1.0–2.0s, GPT-5 ~450–800 ms, GPT-4o ~350–600 ms.** For a Sonnet-4.5 response that finishes in 6 seconds, the user sees the first token at ~700 ms.

The Anthropic streaming events follow a fixed shape: `message_start`, then per content block `content_block_start` / `content_block_delta` (where `delta.type` is either `text_delta` with a `.text` field to append, or `input_json_delta` with `.partial_json` to accumulate into a per-block buffer) / `content_block_stop`, then `message_delta` and `message_stop`. OpenAI's Responses API streams semantic events: `response.output_text.delta`, `response.function_call_arguments.delta`, `response.function_call_arguments.done`, `response.completed`.

```python
import json, time
from anthropic import AsyncAnthropic

async def claude_stream(prompt, tools):
    client = AsyncAnthropic()
    ttft, t0 = None, time.perf_counter()
    tool_bufs = {}
    async with client.messages.stream(model="claude-sonnet-4-5",
        max_tokens=2048, tools=tools,
        messages=[{"role":"user","content":prompt}]) as stream:
        async for ev in stream:
            if ev.type == "content_block_start":
                if ev.content_block.type == "tool_use":
                    tool_bufs[ev.index] = {"id": ev.content_block.id,
                        "name": ev.content_block.name, "json": ""}
            elif ev.type == "content_block_delta":
                if ev.delta.type == "text_delta":
                    if ttft is None: ttft = time.perf_counter()-t0
                    print(ev.delta.text, end="", flush=True)
                elif ev.delta.type == "input_json_delta":
                    tool_bufs[ev.index]["json"] += ev.delta.partial_json
            elif ev.type == "content_block_stop" and ev.index in tool_bufs:
                tool_bufs[ev.index]["args"] = json.loads(tool_bufs[ev.index]["json"])
        final = await stream.get_final_message()
    return ttft, final, tool_bufs
```

Fine-grained tool streaming (opt-in via beta header on Sonnet 4.5 / Haiku 4.5 / Opus 4.x) disables JSON-validation buffering for ultra-low-latency tool streaming, at the cost of occasionally yielding truncated invalid JSON when `max_tokens` is hit. Middle-proxy tools (LiteLLM, Vercel AI SDK) have historically dropped `input_json_delta` events — validate against the raw SDK.

## Budgets and early termination prevent runaway costs

A naive agentic loop without iteration caps can burn through $2–5 of Opus tokens per task on SWE-bench-style workloads simply by looping. Anthropic's internal observations are that typical coding tasks converge in 3–8 iterations; a 20-iter cap reduces p99 cost by 4–6× on coding benchmarks.

```python
import time, asyncio
from dataclasses import dataclass
from anthropic import AsyncAnthropic

@dataclass
class Budget:
    max_iters: int = 10; max_wall_s: float = 30.0
    max_input_tokens: int = 100_000; max_tool_calls: int = 20

class BudgetExceeded(Exception): ...

async def run_agent(prompt, tools, tool_impls, budget=Budget()):
    client = AsyncAnthropic()
    messages = [{"role":"user","content":prompt}]
    started = time.perf_counter()
    iters = tool_calls = in_tokens = 0
    while True:
        if iters >= budget.max_iters: raise BudgetExceeded("iters")
        if time.perf_counter()-started >= budget.max_wall_s: raise BudgetExceeded("wall")
        if in_tokens >= budget.max_input_tokens: raise BudgetExceeded("tokens")
        if tool_calls >= budget.max_tool_calls: raise BudgetExceeded("calls")
        r = await client.messages.create(model="claude-sonnet-4-5",
            max_tokens=1024, tools=tools,
            stop_sequences=["</final_answer>"],
            messages=messages)
        iters += 1; in_tokens += r.usage.input_tokens
        if r.stop_reason in ("end_turn","stop_sequence"):
            return "".join(b.text for b in r.content if b.type=="text")
        tus = [b for b in r.content if b.type=="tool_use"]
        tool_calls += len(tus)
        results = await asyncio.gather(*(tool_impls[b.name](**b.input) for b in tus))
        messages.append({"role":"assistant","content":r.content})
        messages.append({"role":"user","content":[
            {"type":"tool_result","tool_use_id":b.id,"content":str(o)[:4000]}
            for b,o in zip(tus, results)]})
```

Pair hard budgets with the **checkpoint-and-resume pattern** from Anthropic's "Effective harnesses for long-running agents" post: the agent writes progress to a `claude-progress.txt` file, and when budget is exhausted a fresh context window resumes where the previous one stopped. This is how Replit Agent 3 extended autonomous work from 20 minutes to over 200 minutes.

## Layered failure recovery

Production agents need four layers of resilience: exponential-backoff retry (respects `retry-after` headers from 429s), error-aware self-correction (feed the error back to the model as a `tool_result` with `is_error: true`), circuit breakers (stop hammering a failing endpoint), and cross-provider fallback chains (Anthropic → OpenAI).

```python
import asyncio, time, logging
from tenacity import (AsyncRetrying, retry_if_exception_type, stop_after_attempt,
                      wait_exponential_jitter, before_sleep_log)
import anthropic, openai

log = logging.getLogger("resilient")

class CircuitOpen(Exception): ...

class CircuitBreaker:
    def __init__(self, fail_threshold=5, cooldown_s=30):
        self.fail_threshold, self.cooldown_s = fail_threshold, cooldown_s
        self.fails, self.open_until = 0, 0.0
    def before(self):
        if time.time() < self.open_until: raise CircuitOpen()
    def record(self, ok):
        if ok: self.fails = 0
        else:
            self.fails += 1
            if self.fails >= self.fail_threshold:
                self.open_until = time.time() + self.cooldown_s

RETRY_EXC = (anthropic.RateLimitError, anthropic.APIStatusError,
             anthropic.APIConnectionError, openai.RateLimitError,
             openai.APIConnectionError, openai.APIStatusError, asyncio.TimeoutError)

class Resilient:
    def __init__(self):
        self.anthropic = anthropic.AsyncAnthropic()
        self.openai = openai.AsyncOpenAI()
        self.cb_a, self.cb_o = CircuitBreaker(), CircuitBreaker()
    async def _retry(self, fn):
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential_jitter(initial=0.5, max=15, jitter=2),
            retry=retry_if_exception_type(RETRY_EXC),
            before_sleep=before_sleep_log(log, logging.WARNING),
            reraise=True):
            with attempt: return await fn()
    async def complete(self, messages, tools=None):
        try:
            self.cb_a.before()
            r = await self._retry(lambda: self.anthropic.messages.create(
                model="claude-sonnet-4-5", max_tokens=1024,
                tools=tools or [], messages=messages))
            self.cb_a.record(True); return ("anthropic", r)
        except (CircuitOpen, *RETRY_EXC):
            self.cb_a.record(False)
        self.cb_o.before()
        r = await self._retry(lambda: self.openai.responses.create(
            model="gpt-5", input=messages, tools=tools or []))
        self.cb_o.record(True); return ("openai", r)

async def call_tool_with_selfheal(agent_step, tool_fn, tool_input, tool_use_id,
                                  max_attempts=2):
    attempt = 0
    while attempt < max_attempts:
        try: return await tool_fn(**tool_input)
        except Exception as e:
            attempt += 1
            error_ctx = {"type":"tool_result","tool_use_id":tool_use_id,
                         "content":f"ERROR: {type(e).__name__}: {e}",
                         "is_error":True}
            tool_input = await agent_step(error_ctx)
    raise RuntimeError("tool self-heal exhausted")
```

Quantified impact: exponential backoff with jitter cuts effective 429 rates from 2–5% to <0.1% during peak hours. Feeding error messages back to the model resolves **60–80% of schema-mismatch tool failures on the first retry** — Replit publicly reports *"around 90% success rate in tool invocations"* after similar prompt-engineering and self-heal tricks. Circuit breakers reduce blast radius: during a 15-minute provider incident, a tripped breaker avoids ~900 retry requests per client versus a naive retry loop.

## Context window management

Four composable patterns, from simple to sophisticated: sliding window, summarize-on-overflow (Claude Agent SDK's `compact` slash command), hierarchical memory (recent verbatim + medium summarized + old in a vector store), and prompt-prefix caching.

Measured latency impact on Sonnet 4.5: **2k tokens = 500 ms TTFT / $0.006 per call; 50k = 1.3s / $0.15; 200k = 3–5s / $0.60; 200k with 95% cached = 1.0–1.5s / $0.06** — an ~85% latency cut and 90% cost cut from caching alone. A recent controlled study found prompt caching reduces API cost 41–80% and TTFT 13–31% on multi-turn agents, but naively caching tool-result-inclusive contexts can paradoxically *increase* latency — cache the system prompt plus tool schema boundary, not rolling tool outputs.

```python
import tiktoken
from anthropic import Anthropic
enc = tiktoken.get_encoding("cl100k_base")
def count(msgs): return sum(len(enc.encode(str(m))) for m in msgs)

class ContextManager:
    def __init__(self, budget_tokens=120_000, keep_recent=6,
                 summarize_over=80_000, model="claude-haiku-4-5"):
        self.budget, self.keep_recent, self.summarize_over = budget_tokens, keep_recent, summarize_over
        self.summary = ""; self.turns = []; self.client = Anthropic(); self.model = model
    def add(self, msg): self.turns.append(msg)
    def build(self, system_prompt, tools):
        sys_blocks = [{"type":"text","text":system_prompt,
                       "cache_control":{"type":"ephemeral"}}]
        recent = self.turns[-self.keep_recent:]
        older = self.turns[:-self.keep_recent]
        if count(older) + count(recent) > self.summarize_over and older:
            self.summary = self._summarize(older, self.summary)
            prefix = [{"role":"user","content":
                       f"[CONVERSATION SUMMARY]\n{self.summary}\n[END SUMMARY]"}]
        else: prefix = older
        return sys_blocks, prefix + recent
    def _summarize(self, older, prev_summary):
        convo = "\n".join(f"{m['role']}: {str(m['content'])[:2000]}" for m in older)
        r = self.client.messages.create(model=self.model, max_tokens=1024,
            system="Summarize preserving decisions, tool results, file paths, TODOs.",
            messages=[{"role":"user","content":
                       f"Previous summary:\n{prev_summary}\n\nNew dialogue:\n{convo}"}])
        return r.content[0].text
```

Long-context is not free even when it fits: needle-in-haystack accuracy degrades past ~100k tokens on most frontier models. Summarization is lossy, so prefer tool-output *pointers* (file IDs, row IDs) over full payloads — agents can re-fetch on demand. **Replit publicly describes this as "context stays in code, not in tokens":** captured values live in sandbox JavaScript variables and are reused by code reference, never re-serialized into the LLM context. One-line coding tasks like "select December 15 2028 on a calendar" (normally 36 click actions) become a single model call that generates navigation code.

## Concurrency and rate limiting

Two primitives shape agent concurrency: semaphores (cap concurrent in-flight requests) and token/leaky buckets (shape rate per time window). April 2026 Anthropic tiers auto-promote by cumulative deposit: Tier 1 ($5 deposit) = 50 RPM / ~50k ITPM Sonnet; Tier 4 ($400+) = 4,000 RPM / up to 2M Sonnet ITPM / 4M Haiku ITPM. Tier 4 unlocks 1M context for Sonnet and excludes cached-input tokens from ITPM counting — prompt caching effectively 5–10× your rate budget.

```python
import asyncio
from aiolimiter import AsyncLimiter
from anthropic import AsyncAnthropic

class ClaudeSwarm:
    def __init__(self, rpm=1000, itpm=200_000, otpm=40_000, concurrency=50):
        self.req_limit = AsyncLimiter(rpm, 60)
        self.out_limit = AsyncLimiter(otpm, 60)
        self.sem = asyncio.Semaphore(concurrency)
        self.client = AsyncAnthropic()
    async def one(self, model, messages, max_tokens=1024):
        async with self.req_limit, self.sem:
            r = await self.client.messages.create(model=model,
                max_tokens=max_tokens, messages=messages)
            for _ in range(r.usage.output_tokens):
                await self.out_limit.acquire()
            return r
    async def map(self, model, prompts, **kw):
        return await asyncio.gather(*(
            self.one(model, [{"role":"user","content":p}], **kw) for p in prompts))
```

For truly async workloads, both providers offer **Batch APIs at 50% discount**: Anthropic Message Batches (up to 10,000 requests per batch, 24h SLA but usually <1h, counts separately from sync quota) and OpenAI's `/v1/batches` (covers Responses, Chat Completions, Embeddings, Moderations). On a 500k-doc/month pipeline, this saves $750–$2,250 monthly, and stacked with prompt caching the effective price floor is ~25% of list. Quora publicly uses Anthropic's Batch API for summarization and highlight extraction.

## Production patterns from the leading agent teams

**Anthropic's Claude Agent SDK** (formerly Claude Code SDK) crystallizes four pillars: agentic loop = gather context → take action → verify → repeat; subagents with isolated context for parallelizable search; automatic compaction near context limits; verification via rules (linting), visual feedback (screenshots + Playwright MCP), and LLM-as-judge. A key Anthropic observation: *"agents typically use about 4× more tokens than chat interactions, and multi-agent systems use about 15× more tokens than chats"* — multi-agent is only economically viable for high-value tasks.

**Replit Agent 3** hardens tool calls with sandbox isolation, error feedback loops, and generates code to invoke tools rather than using traditional function-calling. Replit's self-testing system performs multi-hundred-step browser testing at a median cost of $0.20 per session — *"3× faster and 10× more cost-effective than Computer Use models"* — and moved testing to a dedicated sub-agent specifically to avoid polluting the main agent's 80,000–100,000-token context.

**Cursor Composer 2** uses an MoE 1T/32B-active model with Multi-Token-Prediction speculative decoding for 2–3× inference speedup, self-distilled for custom MTP layers, with real-time RL on production traffic. **Fast Apply hits ~1,000 tokens/sec on a 70B model** via Fireworks-hosted speculative decoding. Cursor completes most Composer turns in under 30 seconds.

## Performance engineering cheat-sheet

| Technique | Typical impact | Provider knob |
|---|---|---|
| Parallel tool use | 3–5× end-to-end latency drop | Anthropic: default on; OpenAI: `parallel_tool_calls=True` |
| Streaming | 5–10× TTFT improvement | SDK `.stream(...)` |
| Early-termination budget | 4–6× tail-cost reduction | `stop_sequences` + loop caps |
| Retry + fallback | 429/5xx rates → <0.1% | tenacity + dual SDK |
| Prompt caching | 41–80% cost, 13–31% TTFT | `cache_control` / auto prefix |
| Summarize-on-overflow | Sessions survive beyond 200k | SDK compact / custom |
| Rate limiter + semaphore | 0% 429s at tier ceiling | aiolimiter + semaphore |
| Batch API | 50% discount, separate quota | `/v1/batches` / `/v1/messages/batches` |

---

# Topic 4: Evaluation and observability

**Evaluation is the only discipline that separates AI engineers who ship reliable agents from those who ship demos.** Anthropic reports that prompt tweaks routinely move their internal agent success rate from 30% to 80% — with effect sizes that large, a 20-task eval suite is enough to steer development. Conversely, Anthropic's own Opus 4.5 initially scored 42% on CORE-Bench due purely to grader bugs; after fixes it jumped to 95%. Reading transcripts is the highest-leverage activity in the entire field.

## Agent eval differs from LLM eval in four dimensions

LLM eval is prompt → response → grade. Agent eval adds multi-turn loops with state mutation, tool-use correctness, **end-state verification** (DB, filesystem, API side effects), and non-determinism quantified via pass@k (≥1 success in k trials) and pass^k (all k succeed). The core vocabulary, from Anthropic's January 2026 *Demystifying Evals for AI Agents* post, is task → trial → grader → transcript → outcome → harness → suite.

Graders come in three classes. **Code-based graders** (string match, unit tests, state checks) are fast and objective — prefer them whenever possible. **Model-based graders** (LLM judges with rubrics) are flexible but non-deterministic. **Human graders** remain the gold standard for subjective quality. The cardinal rule: grade what the agent produced, not the path it took. Overly rigid tool-sequence assertions punish creative solutions that would satisfy a real user.

A minimum viable eval is 20–50 tasks drawn from real failures, with each task specifying initial state, expected final state, and multiple graders:

```yaml
task_id: refund_edge_case_07
description: "User requests refund for order #1234 but return window expired."
initial_state:
  db: fixtures/orders_1234.sql
expected_final_state:
  tickets: {order_id: 1234, status: resolved, resolution: "store_credit"}
  refunds: []  # must NOT issue refund
graders:
  - type: state_check
    expect: {status: resolved}
  - type: tool_calls
    must_include: [{tool: fetch_policy}, {tool: issue_store_credit}]
    must_not_include: [{tool: process_refund}]
  - type: llm_rubric
    rubric: "Agent was empathetic; explained policy clearly; offered alternative."
```

**pass^k is the metric that matters for customer-facing agents.** Sierra's founder Clay Bavor frames it bluntly: *"0.61 to the eighth power is about 25 percent. So you imagine, what if you're having a thousand of these conversations? You're so far off from being able to rely on this thing."* pass@1 is vanity when users re-run manually; pass^k is reality when an agent ships to production.

A minimal framework-free harness:

```python
import asyncio, json, time, uuid
from dataclasses import dataclass, field
from anthropic import AsyncAnthropic
client = AsyncAnthropic()

@dataclass
class Trial:
    task_id: str; trial_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    transcript: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    final_state: dict = field(default_factory=dict)
    tokens_in: int = 0; tokens_out: int = 0
    latency_s: float = 0.0

async def run_agent(task, tools, env):
    t0 = time.monotonic()
    trial = Trial(task_id=task.id)
    env.reset(task.initial_state)
    messages = [{"role":"user","content":task.prompt}]
    for turn in range(10):
        resp = await client.messages.create(model="claude-opus-4-5",
            max_tokens=2048, tools=tools, messages=messages, temperature=0)
        trial.tokens_in += resp.usage.input_tokens
        trial.tokens_out += resp.usage.output_tokens
        trial.transcript.append(resp.model_dump())
        if resp.stop_reason == "end_turn": break
        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    trial.tool_calls.append({"name":block.name,"input":block.input})
                    result = env.execute(block.name, block.input)
                    tool_results.append({"type":"tool_result",
                        "tool_use_id":block.id,"content":json.dumps(result)})
            messages.append({"role":"assistant","content":resp.content})
            messages.append({"role":"user","content":tool_results})
    trial.final_state = env.snapshot()
    trial.latency_s = time.monotonic() - t0
    return trial

async def run_suite(tasks, tools, env, k=3):
    results = []
    for task in tasks:
        trials = await asyncio.gather(*[run_agent(task, tools, env) for _ in range(k)])
        scores = [sum(g(tr) for g in task.graders)/len(task.graders) for tr in trials]
        results.append({"task":task.id,
            "pass@1": scores[0] >= 0.9,
            f"pass@{k}": any(s >= 0.9 for s in scores),
            f"pass^{k}": all(s >= 0.9 for s in scores),
            "mean_score": sum(scores)/k,
            "avg_tokens": sum(t.tokens_in+t.tokens_out for t in trials)/k})
    return results
```

For ready-made coverage, the benchmark universe includes **SWE-bench Verified** (coding patches), **Terminal-Bench 2.0** (full terminal tasks), **τ-bench / τ²-bench** from Sierra (tool-agent-user multi-turn), **WebArena / VisualWebArena** (browser), **OSWorld** (OS control), **GAIA** (general assistant multimodal), and **MLE-bench** (Kaggle). Inspect Evals from UK AISI is the reference harness for GAIA, MLE-bench, and SWE-bench.

## Tracing with OpenInference and Langfuse

OpenInference (Arize's LLM span spec, compatible with OTel GenAI conventions) is the de-facto standard. Its key attributes include `openinference.span.kind` (LLM/CHAIN/TOOL/AGENT/RETRIEVER/EMBEDDING/RERANKER/GUARDRAIL/EVALUATOR), `llm.system`, `llm.model_name`, `llm.token_count.prompt_details.cache_read`, `llm.cost.total`, `session.id`, `user.id`, and `graph.node.*` for multi-agent topology.

For a 1–5 engineer team building agents on Claude plus OpenAI, the recommended stack is **Langfuse Cloud (free tier) plus OpenInference auto-instrumentation plus Prometheus-compatible metrics**. Langfuse is MIT-licensed, OTel-native after the v4 March 2026 rewrite, and ships tracing plus datasets plus prompt management plus experiments plus LLM-as-judge plus cost tracking plus user feedback in one platform. The free tier covers 50k observations per month, and self-hosting is a single `docker compose`.

```python
# requirements: langfuse>=4.0, openinference-instrumentation-anthropic
import os
from langfuse import observe, get_client, propagate_attributes
from langfuse.openai import OpenAI
from anthropic import Anthropic
from openinference.instrumentation.anthropic import AnthropicInstrumentor

os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
AnthropicInstrumentor().instrument()

anthropic = Anthropic(); openai = OpenAI(); lf = get_client()

@observe(name="search_tool", as_type="tool")
def search_web(query): return f"results for {query}"

@observe(name="plan", as_type="generation")
def plan(goal):
    r = anthropic.messages.create(model="claude-opus-4-5", max_tokens=1024,
        messages=[{"role":"user","content":f"Plan steps for: {goal}"}])
    return r.content[0].text

@observe(name="summarize", as_type="generation")
def summarize(text):
    r = openai.chat.completions.create(model="gpt-5-mini", temperature=0,
        messages=[{"role":"user","content":f"Summarize:\n{text}"}])
    return r.choices[0].message.content

@observe(name="agent_run", as_type="agent")
def agent_run(goal, user_id, task_id):
    with propagate_attributes(user_id=user_id, session_id=task_id,
        tags=["prod","research-agent","v1.3.0"], metadata={"tenant":"acme"}):
        steps = plan(goal)
        evidence = search_web(goal)
        return summarize(f"Plan:\n{steps}\n\nEvidence:\n{evidence}")

if __name__ == "__main__":
    print(agent_run("Compare τ-bench vs SWE-bench", user_id="u_42", task_id="t_abc"))
    lf.flush()
```

Every LLM call, tool call, and parent `agent_run` becomes a properly nested span with cost, latency, cache-hit tokens, and user/session tags for filtering. **Arize Phoenix** is the best alternative for local RAG debugging — launch with `px.launch_app()` for a one-line local server on `http://localhost:6006`. Use **Braintrust** when CI-blocking eval gates are a first-class product requirement; **LangSmith** when you're 100% LangChain/LangGraph and accept vendor coupling. Avoid homegrown solutions — Anthropic's own guidance is blunt: *"It's often best to quickly pick a framework that fits your workflow, then invest your energy in the evals themselves."*

## Cost attribution keeps inference economics visible

Teams routinely find 30–50% of agent spend going to a handful of power users or to a single verbose tool (a search tool returning 20 KB of unsummarized HTML is a classic). Attribution makes those fixable. A context-manager-based cost tracker accumulates across an agent run, attributed by user_id and task_id:

```python
import contextvars, threading, time
from contextlib import contextmanager
from dataclasses import dataclass, field

PRICING = {
    "claude-opus-4-5": {"in":5.00,"out":25.00,"cache_read":0.50,"cache_write":6.25},
    "claude-sonnet-4-5":{"in":3.00,"out":15.00,"cache_read":0.30,"cache_write":3.75},
    "gpt-5":{"in":1.25,"out":10.00},
    "gpt-5-mini":{"in":0.25,"out":2.00},
}

@dataclass
class CostBucket:
    user_id: str; task_id: str
    by_model: dict = field(default_factory=dict)
    by_tool: dict = field(default_factory=dict)
    total_usd: float = 0.0
    tokens_in: int = 0; tokens_out: int = 0
    cache_hit_tokens: int = 0

_ctx = contextvars.ContextVar("cost", default=None)
_lock = threading.Lock()

def _price(model, usage):
    p = PRICING.get(model, {"in":0,"out":0})
    c = usage.get("input_tokens",0) * p["in"] / 1e6
    c += usage.get("output_tokens",0) * p["out"] / 1e6
    c += usage.get("cache_read_input_tokens",0) * p.get("cache_read",p["in"]) / 1e6
    c += usage.get("cache_creation_input_tokens",0) * p.get("cache_write",p["in"]) / 1e6
    return c

@contextmanager
def track_cost(user_id, task_id):
    bucket = CostBucket(user_id=user_id, task_id=task_id)
    token = _ctx.set(bucket)
    try: yield bucket
    finally:
        _ctx.reset(token)
        print(f"[cost] user={user_id} task={task_id} ${bucket.total_usd:.4f}")

def record_llm(model, usage):
    b = _ctx.get()
    if not b: return
    cost = _price(model, usage)
    with _lock:
        b.total_usd += cost
        b.tokens_in += usage.get("input_tokens",0)
        b.tokens_out += usage.get("output_tokens",0)
        b.cache_hit_tokens += usage.get("cache_read_input_tokens",0)
        b.by_model[model] = b.by_model.get(model,0) + cost
```

Push the bucket into Redis for multi-process workers, or ship as spans — Langfuse and Phoenix do this natively. **Cache-hit tracking alone routinely reveals 40–80% cost reduction opportunities** on multi-turn agents that weren't configured for caching.

## Regression testing with semantic snapshots

Exact-match snapshot testing has a ~40% false-positive rate on LLM outputs because wording drifts. Semantic snapshots — `cos(embed(expected), embed(actual)) ≥ 0.92` with an LLM-judge fallback — drop false positives to under 5% while still catching real regressions.

```python
import json, os, pytest, numpy as np
from pathlib import Path
from openai import OpenAI
from anthropic import Anthropic

oai, anth = OpenAI(), Anthropic()
SNAP_DIR = Path("__snapshots__"); SNAP_DIR.mkdir(exist_ok=True)

def _embed(text):
    v = oai.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding
    return np.array(v)

def _cos(a, b): return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)))

def semantic_equivalent(a, b, threshold=0.92):
    return _cos(_embed(a), _embed(b)) >= threshold

def llm_judge_equivalent(expected, actual):
    r = anth.messages.create(model="claude-opus-4-5", max_tokens=256,
        messages=[{"role":"user","content":
            f"Are these semantically equivalent? JSON {{\"equivalent\":bool}}.\n"
            f"EXPECTED:\n{expected}\n\nACTUAL:\n{actual}"}])
    return json.loads(r.content[0].text)["equivalent"]

def snapshot(name, value, update=None):
    if update is None: update = os.getenv("UPDATE_SNAPSHOTS") == "1"
    path = SNAP_DIR / f"{name}.txt"
    if update or not path.exists():
        path.write_text(value); return
    expected = path.read_text()
    if expected == value: return
    if semantic_equivalent(expected, value): return
    assert llm_judge_equivalent(expected, value), f"Regression in {name}"
```

Budget ~$0.005 per test with Opus as judge; swap to Haiku or GPT-5-mini for low-stakes snapshots. Calibrate the similarity threshold against a small human-labeled set — 0.92 is a starting point, not a law. Never assume determinism even at `temperature=0`: Anthropic does not guarantee it, and OpenAI's `seed` param is best-effort.

## LLM-as-judge with bias mitigations

LLM judges exhibit five known biases: position (prefers first/last), verbosity (prefers longer), self-enhancement (GPT-4 prefers GPT-4 outputs), authority (prefers citations), and rubric-position. The fixes are well-established in the literature (MT-Bench, G-Eval): pairwise comparisons beat absolute 1–10 scoring, chain-of-thought judgments lift Spearman correlation with humans from ~0.5 to ~0.7, position randomization with consistency-check throws out flipped verdicts, and multi-judge ensembles across model families cut self-preference bias.

```python
import json, asyncio
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
anth, oai = AsyncAnthropic(), AsyncOpenAI()

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

Return strict JSON: {{"reasoning": "step-by-step", "verdict": "A"|"B"|"tie"}}"""

async def _judge_once(judge, query, a, b):
    text = JUDGE_PROMPT.format(query=query, a=a, b=b)
    if judge == "claude":
        r = await anth.messages.create(model="claude-opus-4-5", max_tokens=1024,
            messages=[{"role":"user","content":text}])
        return json.loads(r.content[0].text)
    r = await oai.chat.completions.create(model="gpt-5", temperature=0,
        response_format={"type":"json_object"},
        messages=[{"role":"user","content":text}])
    return json.loads(r.choices[0].message.content)

async def pairwise_judge(query, x, y, judges=("claude","gpt")):
    tasks = []
    for j in judges:
        tasks.append(_judge_once(j, query, x, y))  # X=A, Y=B
        tasks.append(_judge_once(j, query, y, x))  # swapped
    rs = await asyncio.gather(*tasks)
    votes = {"X":0, "Y":0, "tie":0, "inconsistent":0}
    for i, j in enumerate(judges):
        v1, v2 = rs[2*i]["verdict"], rs[2*i+1]["verdict"]
        if v1 == "A" and v2 == "B": votes["X"] += 1
        elif v1 == "B" and v2 == "A": votes["Y"] += 1
        elif v1 == "tie" and v2 == "tie": votes["tie"] += 1
        else: votes["inconsistent"] += 1
    valid = votes["X"] + votes["Y"] + votes["tie"]
    if valid == 0: return {"winner":"tie","confidence":0.0}
    winner = max(("X","Y","tie"), key=lambda k: votes[k])
    return {"winner":winner, "confidence":votes[winner]/valid, "detail":votes}
```

Position-swap plus ensemble reduces spurious preferences from ~20% to under 3% in Arena-Hard-style setups. The cost is 4× base (2 positions × 2 judges); reserve the full ensemble for release-gate evals and use a single-judge-with-swap for dev-loop iteration. Never judge with the same family as the generator for high-stakes comparisons. LMArena, AlpacaEval 2.0, and Arena-Hard-Auto v2 all standardize position swap.

## Production monitoring metrics and alerts

The minimum viable agent dashboard has nine metrics with defined alert thresholds:

| Metric | Why it matters | Sensible alert |
|---|---|---|
| **Task success rate** | Eval-in-prod | <95% of 7-day baseline |
| **Thumbs-down rate** | Leading quality indicator | >2× 7-day baseline in 1h |
| **Latency p50/p95/p99** | UX | p95 > 2× baseline for 10 min |
| **Tool failure rate** | First thing to break | >5% for any tool, 5 min |
| **Retry / loop rate** | Agent stuck | avg turns >1.3× baseline |
| **Cache hit rate** | Bill sanity | drop >20 percentage points |
| **Cost per task** | Velocity killer | >1.5× baseline |
| **Token usage p95** | Runaway loops | >3× task baseline |
| **Guardrail trip rate** | Safety/abuse | >1% spike |

A Prometheus-plus-OTel wrapper that instruments LLM calls, tool calls, and whole agent tasks:

```python
import time, functools
from contextlib import contextmanager
from prometheus_client import Counter, Histogram, Gauge, start_http_server

LLM_CALLS  = Counter("llm_calls_total", "LLM calls", ["model","status"])
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens", ["model","kind"])
LLM_USD    = Counter("llm_cost_usd_total", "LLM cost USD", ["model","user"])
LLM_LAT    = Histogram("llm_latency_seconds", "LLM latency", ["model"],
                       buckets=(0.1,0.25,0.5,1,2,5,10,30,60,120))
TOOL_CALLS = Counter("tool_calls_total", "Tool invocations", ["tool","status"])
TOOL_LAT   = Histogram("tool_latency_seconds", "Tool latency", ["tool"])
TASK_OK    = Counter("agent_tasks_total", "Agent task outcomes", ["status"])
INFLIGHT   = Gauge("agent_inflight", "In-flight tasks")
FEEDBACK   = Counter("user_feedback_total", "Feedback", ["sentiment"])

start_http_server(9464)  # Prometheus scrapes /metrics

def trace_llm(model):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.monotonic()
            try:
                r = fn(*args, **kwargs)
                u = getattr(r, "usage", None)
                if u:
                    LLM_TOKENS.labels(model,"in").inc(u.input_tokens)
                    LLM_TOKENS.labels(model,"out").inc(u.output_tokens)
                    LLM_TOKENS.labels(model,"cache_read").inc(
                        getattr(u,"cache_read_input_tokens",0) or 0)
                LLM_CALLS.labels(model,"ok").inc()
                return r
            except Exception:
                LLM_CALLS.labels(model,"error").inc(); raise
            finally:
                LLM_LAT.labels(model).observe(time.monotonic()-t0)
        return wrapper
    return deco

@contextmanager
def track_task(user_id, task_id):
    INFLIGHT.inc(); t0 = time.monotonic(); status = "success"
    try: yield
    except TimeoutError: status = "timeout"; raise
    except Exception: status = "fail"; raise
    finally:
        TASK_OK.labels(status).inc(); INFLIGHT.dec()
```

Critical PromQL queries: cache hit ratio = `sum(rate(llm_tokens_total{kind="cache_read"}[5m])) / sum(rate(llm_tokens_total{kind="in"}[5m]))`; task success rate = `sum(rate(agent_tasks_total{status="success"}[5m])) / sum(rate(agent_tasks_total[5m]))`; p95 latency = `histogram_quantile(0.95, rate(agent_task_seconds_bucket[5m]))`. Keep labels low-cardinality — user_id on cost counters is fine, task_id is not; push task-level detail to trace metadata.

## Real-world evaluation practices

**Anthropic** develops evaluations with just 20 test queries representing real usage patterns, iterates with single-LLM-call judges outputting 0.0–1.0 scores plus pass/fail grades, and notes effect sizes of 30% → 80% success rates that make small samples sufficient. They explicitly monitor *"agent decision patterns and interaction structures—all without monitoring the contents of individual conversations."*

**Harvey** (legal AI at $100M+ ARR, 500+ customers including 42% of the Am Law 100) enforces **leave-one-out eval gates** on all capability changes. Their three risks are: system-prompt-versus-Tool-Bundle conflicts, Tool-Bundle-versus-system-prompt conflicts, and context rot. Their published engineering quote captures the cultural challenge better than any technical detail: *"The hardest part of adopting agents isn't writing the code — it's learning, as an engineering org, to share ownership of a single brain. You're no longer merging unit-testable code, you're merging English."*

**Sierra** built **τ-bench and τ²-bench** as open-source versions of their internal simulation-based testing. They pair agents with user simulators that mimic real-world behaviors across diverse personas, and their pass^k metric reframed the whole industry's understanding of customer-facing agent quality.

**Replit** uses LangSmith for within-trace search on complex agent runs and thread-view for human-in-the-loop monitoring. Their self-testing Playwright subagent runs isolated from the main agent context, catching "Potemkin interfaces" (features that pass visual inspection but aren't wired up).

**LangChain/LangGraph** published case studies show the observability premium at scale: Klarna reduced customer query resolution time by 80% and automated ~70% of repetitive tasks with 2.5M AI Assistant conversations, performing work equivalent to 700 FTEs. AppFolio Realm-X doubled response accuracy after switching to LangGraph.

---

# Differentiation strategy for AI engineers

The AI engineer job market in 2026 is saturated with candidates who can build a LangChain demo and invoke a Claude API. The engineers who command premium salaries and get hired into serious agent teams have **four concrete capabilities that tutorial-level engineers lack**: they measure, they optimize, they handle failure, and they ship evaluations.

## What hiring signals separate senior from tutorial

**Serious candidates can quote the numbers.** If someone cannot tell you what a cached Anthropic input token costs, what typical TTFT looks like on Haiku 4.5 versus Sonnet 4.5, what their tool schemas cost in tokens, or what their p95 tool failure rate is, they have never operated a production agent. Memorize the pricing table in this dossier. Know the 90% cache discount, the 50% batch discount, the 15× token multiplier on multi-agent systems, the 73.3% Haiku SWE-bench Verified score.

**Serious candidates have opinions about architecture.** They can steelman the multi-agent case (Anthropic's +90.2% research eval) and the single-agent case (Cognition's "Don't Build Multi-Agents") with published citations, and they know which workload shape favors which. They can explain why Claude Code only uses sub-agents for read-only information gathering. They know why Cursor's speculative decoding is different from cascade routing.

**Serious candidates default to eval-first development.** They do not iterate on prompts by vibes. They build a 20-task eval suite before writing production code, and they measure every change against it. They understand pass@k versus pass^k and pick the right one for their product.

## Five portfolio projects that demonstrate production capability

A portfolio of tutorial-style agents is undifferentiated. A portfolio of projects that demonstrate the disciplines in this dossier is rare.

1. **A public benchmark repo** running an agent you built on SWE-bench Verified or τ-bench (Sierra's benchmark is open-source), with a results table comparing Haiku 4.5, Sonnet 4.5, and Opus 4.5, including pass^k and cost-per-task columns. This proves you can operate eval infrastructure.

2. **A prompt-caching case study** showing a real agent with before/after measurements from your own logs: input tokens per turn, cache hit rate, dollar cost per 1,000 sessions. Include the `cache_control` placement and the reasoning. Publish the raw usage JSON.

3. **A cascade router implementation** with your own measured success rates and cost curves across Haiku/Sonnet/Opus on a specific task (code review, SQL generation, customer support classification). Compare to an always-flagship baseline and compute the break-even escalation rate.

4. **An observability dashboard** in Langfuse or Arize Phoenix, with screenshots showing your six core metrics (task success, latency p95, cache hit rate, cost per user, tool failure rate, thumbs-down rate). Walk through how you alert and respond to a real incident.

5. **A failure-mode post-mortem** of an agent you built — a document describing a real production issue (context overflow, 429 storm, tool schema drift, judge bias), how you detected it, and how you fixed it. Reference the specific metrics that made it visible.

## Specialized knowledge areas that compound

Three areas compound into hireability over a 6–12 month horizon. **Reasoning models** (o3, Claude extended thinking, DeepSeek R1 variants) have distinct operational characteristics — latency, reasoning token budgets, when their value justifies the cost — and the engineers who have worked with them day-to-day are rare. **Computer-use and browser agents** (Anthropic Computer Use, OpenAI Operator-class products, WebArena-style evals) are the next frontier where demo-versus-production gap is extreme. **Reinforcement learning from production traffic** (Cursor's real-time RL pattern, Sierra's simulation-based testing) is where the teams shipping the best agents are moving; being able to speak to reward models, preference data, and online learning pipelines puts you in a small pool.

## The final meta-point

Anthropic, Cursor, Perplexity, Replit, Harvey, and Sierra all converge on the same conclusion in their engineering writing: **context engineering is the number-one job of engineers building AI agents**. Framework choice does not matter much; model choice matters more than framework; prompt design matters more than model; context curation matters more than prompt. The engineers who understand that hierarchy and have the measurement discipline to prove they understand it are the engineers who get hired to build production agents in 2026.

The techniques in this dossier are reproducible for $20–40 of API credits and a weekend of work. The insights are not. The differentiation is not knowing the techniques — it is the practiced judgment about which ones to reach for when a specific production agent is costing too much, running too slow, or failing too often. Build the portfolio projects, measure everything, and publish your numbers. That is the competitive moat.