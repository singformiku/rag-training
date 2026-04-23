"""
Streaming — measure Time-To-First-Token (TTFT) and total wall-clock
===================================================================

What this demo shows
--------------------
Two identical prompts are sent to the same model.  Call A collects the whole
response via a normal blocking ``chat(...)``.  Call B streams token-by-token
and records the wall-clock time when the first text delta arrives
(TTFT) and when the stream completes.

Typical observation: TTFT on a streaming call is 5–10× lower than the
blocking call's total latency, because you start rendering the first word
while the model continues generating.

Why it matters
--------------
For interactive agents, TTFT is the "felt speed" metric.  The article quotes
500ms TTFT on Sonnet 4.5 for a 2k-token prompt; blocking calls on the same
prompt feel 5-10× slower to a human watching the cursor.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import BACKEND, MODELS, banner, chat, stream  # noqa: E402

PROMPT = (
    "Explain, in a single ~10-sentence paragraph, how HTTP/2 multiplexing "
    "improves over HTTP/1.1 head-of-line blocking.  Avoid bullet points."
)


def run_blocking() -> dict:
    t0 = time.perf_counter()
    r = chat(
        messages=[{"role": "user", "content": PROMPT}],
        tier="medium",
        max_tokens=500,
        extra={"reasoning_effort": "low"},
    )
    total = time.perf_counter() - t0
    return {
        "mode": "blocking", "ttft_s": total, "total_s": total,
        "out_tokens": r.output_tokens, "text_len": len(r.text),
    }


def run_streaming() -> dict:
    t0 = time.perf_counter()
    ttft = None
    total_chars = 0
    first_tokens_preview = ""
    print("stream > ", end="", flush=True)
    final_result = None
    for ev in stream(
        messages=[{"role": "user", "content": PROMPT}],
        tier="medium",
        max_tokens=500,
    ):
        if ev["type"] == "text":
            if ttft is None:
                ttft = time.perf_counter() - t0
            piece = ev["delta"]
            total_chars += len(piece)
            if len(first_tokens_preview) < 120:
                first_tokens_preview += piece
                print(piece, end="", flush=True)
        elif ev["type"] == "done":
            final_result = ev["result"]
    total = time.perf_counter() - t0
    print()
    return {
        "mode": "streaming",
        "ttft_s": ttft or total, "total_s": total,
        "out_tokens": final_result.output_tokens if final_result else None,
        "text_len": total_chars, "preview": first_tokens_preview,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", action="store_true",
                        help="Discard a throwaway call first to warm up the connection.")
    args = parser.parse_args()

    banner(f"Streaming TTFT demo  |  backend={BACKEND}  model={MODELS['medium']}")

    if args.warmup:
        print("warming up ...")
        chat(messages=[{"role": "user", "content": "hi"}], tier="medium", max_tokens=8)

    print("running blocking call ...")
    blocking = run_blocking()
    print("\nrunning streaming call ...")
    streaming = run_streaming()

    banner("Summary")
    print(f"blocking  : ttft={blocking['ttft_s']:5.2f}s  total={blocking['total_s']:5.2f}s")
    print(f"streaming : ttft={streaming['ttft_s']:5.2f}s  total={streaming['total_s']:5.2f}s")
    if streaming["ttft_s"]:
        ratio = blocking["total_s"] / max(streaming["ttft_s"], 1e-6)
        print(f"ttft speed-up over blocking: {ratio:.1f}x")
    print(
        "\nStreaming does NOT reduce total tokens/time — it just lets the UI "
        "start rendering sooner.  Combine with prompt caching for the big win."
    )


if __name__ == "__main__":
    main()
