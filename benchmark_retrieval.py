"""Micro-benchmark retrieval paths (cold vs warm cache).

Usage:
  python benchmark_retrieval.py
  python benchmark_retrieval.py --iterations 50
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, List

from pramana_engine.hybrid_retrieval import clear_hybrid_retrieval_cache, hybrid_search
from pramana_engine.rag_pipeline import get_rag_pipeline


def _measure(label: str, fn: Callable[[], object], iterations: int) -> List[float]:
    timings_ms: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings_ms.append(elapsed_ms)

    avg = statistics.mean(timings_ms)
    p95 = sorted(timings_ms)[max(0, int(len(timings_ms) * 0.95) - 1)]
    print(f"{label:30} avg={avg:8.2f} ms  p95={p95:8.2f} ms")
    return timings_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark retrieval runtime with and without cache warmup")
    parser.add_argument("--iterations", type=int, default=20, help="Number of timed iterations")
    args = parser.parse_args()

    question = "What is the relation between hetu and vyapti in Nyaya inference?"
    cold_question = question + " (cold-run)"
    pramanas = ["inference", "testimony"]

    pipeline = get_rag_pipeline()
    if not pipeline._initialized:
        pipeline.initialize()

    print("Retrieval micro-benchmark")
    print("-" * 72)

    # Cold path
    clear_hybrid_retrieval_cache()
    pipeline.clear_runtime_caches()
    _measure(
        "hybrid_search cold",
        lambda: hybrid_search(question=cold_question, pramana_types=pramanas, k=10),
        1,
    )

    clear_hybrid_retrieval_cache()
    pipeline.clear_runtime_caches()
    _measure(
        "pipeline.search_only cold",
        lambda: pipeline.search_only(question=cold_question, pramana_types=pramanas, k=10),
        1,
    )

    # Warm path
    _measure(
        "hybrid_search warm",
        lambda: hybrid_search(question=question, pramana_types=pramanas, k=10),
        args.iterations,
    )
    _measure(
        "pipeline.search_only warm",
        lambda: pipeline.search_only(question=question, pramana_types=pramanas, k=10),
        args.iterations,
    )

    print("-" * 72)
    print("Benchmark complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
