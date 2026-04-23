"""
Concurrency + Rate Limiting — asyncio semaphore + token-bucket shaping
======================================================================

What this demo shows
--------------------
We launch ``n_requests`` tiny chat calls and compare three dispatch regimes:

  * ``unbounded``       — ``asyncio.gather(*[...])`` — fastest, but can burst
                            past your provider's RPM/ITPM ceiling and get 429s.
  * ``semaphore``       — cap concurrency to K in-flight at once.
  * ``token_bucket``    — shape requests to at-most ``rpm`` per minute using
                            ``aiolimiter.AsyncLimiter``.

Every call wraps a mocked-429 backoff so the differences are visible without
actually hammering the provider.  We report: wall-clock, max concurrent
in-flight, number of simulated 429s.

Why it matters
--------------
Anthropic / OpenAI enforce tier-based RPM / ITPM / OTPM ceilings.  Unbounded
fan-out wastes both tokens (retries) AND wall time (exponential backoff).
A single-line semaphore + leaky bucket eliminates 429s at the tier ceiling.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

from aiolimiter import AsyncLimiter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import BACKEND, MODELS, banner, chat_async  # noqa: E402


# A trivial stand-in for provider rate-limit behaviour: we track an atomic
# "inflight" counter; anything above ``hard_ceiling`` simulates a 429 + retry.
class FakeProvider:
    def __init__(self, hard_ceiling: int):
        self.hard_ceiling = hard_ceiling
        self.inflight = 0
        self.max_inflight = 0
        self.simulated_429 = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def slot(self):
        async with self._lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            over = self.inflight > self.hard_ceiling
        try:
            if over:
                # Simulate a 429 + exponential backoff.
                self.simulated_429 += 1
                await asyncio.sleep(0.5)
            yield
        finally:
            async with self._lock:
                self.inflight -= 1


async def one_request(provider: FakeProvider, query: str) -> int:
    async with provider.slot():
        r = await chat_async(
            messages=[{"role": "user", "content": query}],
            tier="cheap",
            max_tokens=32,
            extra={"reasoning_effort": "low"},
        )
        return r.output_tokens


async def run_unbounded(provider: FakeProvider, queries: List[str]) -> Dict[str, float]:
    t0 = time.perf_counter()
    await asyncio.gather(*(one_request(provider, q) for q in queries))
    return {"regime": "unbounded", "wall_s": time.perf_counter() - t0}


async def run_semaphore(provider: FakeProvider, queries: List[str], k: int) -> Dict[str, float]:
    sem = asyncio.Semaphore(k)

    async def _wrapped(q: str) -> int:
        async with sem:
            return await one_request(provider, q)

    t0 = time.perf_counter()
    await asyncio.gather(*(_wrapped(q) for q in queries))
    return {"regime": f"semaphore={k}", "wall_s": time.perf_counter() - t0}


async def run_token_bucket(provider: FakeProvider, queries: List[str], rpm: int) -> Dict[str, float]:
    limiter = AsyncLimiter(rpm, 60)

    async def _wrapped(q: str) -> int:
        async with limiter:
            return await one_request(provider, q)

    t0 = time.perf_counter()
    await asyncio.gather(*(_wrapped(q) for q in queries))
    return {"regime": f"token_bucket rpm={rpm}", "wall_s": time.perf_counter() - t0}


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--hard-ceiling", type=int, default=3,
                        help="Simulated provider concurrent-request limit.")
    parser.add_argument("--sem", type=int, default=3)
    parser.add_argument("--rpm", type=int, default=60)
    args = parser.parse_args()

    banner(f"Concurrency + rate limiting demo  |  backend={BACKEND}  model={MODELS['cheap']}")
    print(f"{args.n} requests, fake provider ceiling={args.hard_ceiling} "
          f"(over-ceiling calls simulate a 429+backoff)\n")

    queries = [f"What's 2 + {i}? Reply with the digit only." for i in range(args.n)]

    rows = []
    for runner, kwargs in [
        (run_unbounded,    {}),
        (run_semaphore,    {"k": args.sem}),
        (run_token_bucket, {"rpm": args.rpm}),
    ]:
        provider = FakeProvider(args.hard_ceiling)
        r = await runner(provider, queries, **kwargs)
        r.update({"max_inflight": provider.max_inflight, "simulated_429": provider.simulated_429})
        rows.append(r)
        print(
            f"{r['regime']:<20}  wall={r['wall_s']:5.2f}s  "
            f"max_inflight={r['max_inflight']:<3}  simulated_429={r['simulated_429']}"
        )

    banner("Takeaway")
    print(
        "Unbounded fan-out is the fastest on paper but triggers 429s in "
        "production.  A semaphore + token bucket eliminates 429s with a "
        "modest wall-clock penalty — always the right default for batch jobs."
    )


if __name__ == "__main__":
    asyncio.run(_main())
