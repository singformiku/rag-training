import argparse
import json
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_service import llm_service
from src.config import settings


def _read_field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _build_shared_prefix(repeats: int) -> str:
    block = (
        "Reference context for token cache benchmarking. "
        "This block is intentionally stable across requests so the model backend can reuse the same prefix tokens. "
        "It describes coding standards, evaluation rules, response format constraints, and project background."
    )
    return "\n".join(block for _ in range(repeats))


def _build_messages(shared_prefix: str, question: str, marker: str) -> list[dict[str, str]]:
    system_content = f"request_marker: {marker}\n{shared_prefix}"
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]


def _extract_usage(completion: Any) -> dict[str, Any]:
    usage = getattr(completion, "usage", None)
    prompt_tokens = _read_field(usage, "prompt_tokens")
    completion_tokens = _read_field(usage, "completion_tokens")
    total_tokens = _read_field(usage, "total_tokens")
    prompt_details = _read_field(usage, "prompt_tokens_details") or _read_field(usage, "input_tokens_details")
    cached_tokens = _read_field(prompt_details, "cached_tokens")
    cache_read_input_tokens = _read_field(usage, "cache_read_input_tokens")
    cache_creation_input_tokens = _read_field(usage, "cache_creation_input_tokens")
    if cached_tokens is None:
        cached_tokens = cache_read_input_tokens
    uncached_prompt_tokens = None
    if isinstance(prompt_tokens, int):
        uncached_prompt_tokens = prompt_tokens
        if isinstance(cached_tokens, int):
            uncached_prompt_tokens = max(prompt_tokens - cached_tokens, 0)
    cache_hit_ratio = None
    if isinstance(prompt_tokens, int) and prompt_tokens > 0 and isinstance(cached_tokens, int):
        cache_hit_ratio = cached_tokens / prompt_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "uncached_prompt_tokens": uncached_prompt_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_hit_ratio": cache_hit_ratio,
    }


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record[key] for record in records if isinstance(record.get(key), (int, float))]
    if not values:
        return None
    return statistics.mean(values)


def _run_case(name: str, shared_prefix: str, question: str, runs: int, max_tokens: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index in range(runs):
        marker = "stable-00000000-0000-0000-0000-000000000000"
        if name == "without_cache":
            marker = str(uuid.uuid4())
        messages = _build_messages(shared_prefix=shared_prefix, question=question, marker=marker)
        started_at = time.perf_counter()
        completion = llm_service.complete_with_tools(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=0.1,
        )
        latency_seconds = time.perf_counter() - started_at
        choice = completion.choices[0]
        usage = _extract_usage(completion)
        results.append(
            {
                "case": name,
                "run": index + 1,
                "latency_seconds": latency_seconds,
                "finish_reason": choice.finish_reason,
                "response_preview": (choice.message.content or "")[:200],
                **usage,
            }
        )
    return results


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": len(records),
        "avg_latency_seconds": _mean(records, "latency_seconds"),
        "avg_prompt_tokens": _mean(records, "prompt_tokens"),
        "avg_completion_tokens": _mean(records, "completion_tokens"),
        "avg_total_tokens": _mean(records, "total_tokens"),
        "avg_cached_tokens": _mean(records, "cached_tokens"),
        "avg_uncached_prompt_tokens": _mean(records, "uncached_prompt_tokens"),
        "avg_cache_hit_ratio": _mean(records, "cache_hit_ratio"),
    }


def _build_report(with_cache: list[dict[str, Any]], without_cache: list[dict[str, Any]], repeats: int, question: str, max_tokens: int) -> dict[str, Any]:
    with_cache_summary = _summarize(with_cache)
    with_cache_warm_summary = _summarize(with_cache[1:]) if len(with_cache) > 1 else None
    without_cache_summary = _summarize(without_cache)

    baseline_with_cache_summary = with_cache_warm_summary or with_cache_summary

    uncached_with_cache = baseline_with_cache_summary.get("avg_uncached_prompt_tokens")
    uncached_without_cache = without_cache_summary.get("avg_uncached_prompt_tokens")
    latency_with_cache = baseline_with_cache_summary.get("avg_latency_seconds")
    latency_without_cache = without_cache_summary.get("avg_latency_seconds")

    uncached_prompt_reduction_ratio = None
    if isinstance(uncached_with_cache, (int, float)) and isinstance(uncached_without_cache, (int, float)) and uncached_without_cache > 0:
        uncached_prompt_reduction_ratio = (uncached_without_cache - uncached_with_cache) / uncached_without_cache

    latency_improvement_ratio = None
    if isinstance(latency_with_cache, (int, float)) and isinstance(latency_without_cache, (int, float)) and latency_without_cache > 0:
        latency_improvement_ratio = (latency_without_cache - latency_with_cache) / latency_without_cache

    cache_supported = any(
        isinstance(record.get("cached_tokens"), int) and record.get("cached_tokens", 0) > 0
        for record in with_cache + without_cache
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": settings.LLM_MODEL,
        "prompt_repeats": repeats,
        "max_tokens": max_tokens,
        "question": question,
        "cache_supported_by_endpoint": cache_supported,
        "with_cache": {
            "summary": with_cache_summary,
            "warm_summary": with_cache_warm_summary,
            "runs": with_cache,
        },
        "without_cache": {
            "summary": without_cache_summary,
            "runs": without_cache,
        },
        "difference": {
            "avg_uncached_prompt_token_reduction_ratio": uncached_prompt_reduction_ratio,
            "avg_latency_improvement_ratio": latency_improvement_ratio,
            "avg_cached_tokens_with_cache": baseline_with_cache_summary.get("avg_cached_tokens"),
            "avg_cached_tokens_without_cache": without_cache_summary.get("avg_cached_tokens"),
        },
    }


def _print_summary(report: dict[str, Any]) -> None:
    with_cache_summary = report["with_cache"]["summary"]
    with_cache_warm_summary = report["with_cache"].get("warm_summary")
    without_cache_summary = report["without_cache"]["summary"]
    difference = report["difference"]

    print("=" * 72)
    print(f"Model: {report['model']}")
    print(f"Cache supported by endpoint: {report['cache_supported_by_endpoint']}")
    print("=" * 72)
    print("With cache")
    print(json.dumps(with_cache_summary, indent=2, ensure_ascii=False))
    if with_cache_warm_summary is not None:
        print("-" * 72)
        print("With cache (warm runs only)")
        print(json.dumps(with_cache_warm_summary, indent=2, ensure_ascii=False))
    print("-" * 72)
    print("Without cache")
    print(json.dumps(without_cache_summary, indent=2, ensure_ascii=False))
    print("-" * 72)
    print("Difference")
    print(json.dumps(difference, indent=2, ensure_ascii=False))



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prefix-repeats", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument(
        "--question",
        type=str,
        default="Summarize the main engineering trade-offs of agentic RAG in 5 bullet points.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("token_cache_results.json"),
    )
    args = parser.parse_args()

    settings.validate()
    shared_prefix = _build_shared_prefix(args.prefix_repeats)
    with_cache_results = _run_case(
        name="with_cache",
        shared_prefix=shared_prefix,
        question=args.question,
        runs=args.runs,
        max_tokens=args.max_tokens,
    )
    without_cache_results = _run_case(
        name="without_cache",
        shared_prefix=shared_prefix,
        question=args.question,
        runs=args.runs,
        max_tokens=args.max_tokens,
    )
    report = _build_report(
        with_cache=with_cache_results,
        without_cache=without_cache_results,
        repeats=args.prefix_repeats,
        question=args.question,
        max_tokens=args.max_tokens,
    )

    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_summary(report)
    print("-" * 72)
    print(f"Saved detailed report to: {args.output}")


if __name__ == "__main__":
    main()
