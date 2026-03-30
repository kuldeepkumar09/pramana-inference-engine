"""
Pramana Engine - Demo Benchmark
Measures API endpoint latency (cold vs warm cache) without requiring Ollama.

Usage:
    python demo_benchmark.py
    python demo_benchmark.py --iterations 10
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, List, Tuple

from pramana_engine.web import create_app


def _timeit(fn: Callable[[], object], n: int) -> List[float]:
    """Run fn n times, return elapsed milliseconds per call."""
    times: List[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _row(label: str, cache: str, times: List[float]) -> Tuple[str, ...]:
    avg = statistics.mean(times)
    p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
    return (label, cache, f"{avg:7.1f}", f"{p95:7.1f}", f"{min(times):7.1f}", str(len(times)))


def _print_table(rows: List[Tuple[str, ...]]) -> None:
    headers = ("Endpoint", "Cache", "Avg ms", "P95 ms", "Min ms", "N")
    col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        print(fmt.format(*r))
    print(sep)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pramana Engine API latency benchmark")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Warm-cache iterations per endpoint (default: 5)")
    n = max(1, parser.parse_args().iterations)

    # Flask test client — no live server, no Ollama needed
    app = create_app()
    app.testing = True
    client = app.test_client()

    INFER_PAYLOAD = {
        "paksha": "the distant hill",
        "sadhya": "fire is present",
        "hetu": "smoke is visible",
        "udaharana": "as in a kitchen hearth",
        "hetuConf": 0.85,
        "vyaptiStr": 0.90,
        "pramanaType": "Anumana",
    }
    COMPARE_PAYLOAD = {
        "paksha": "the distant hill",
        "sadhya": "fire is present",
        "hetu": "smoke is visible",
        "hetuConf": 0.85,
        "vyaptiStr": 0.90,
    }
    RAG_SEARCH_PAYLOAD = {
        "question": "What is the relation between hetu and vyapti in Nyaya inference?",
        "k": 5,
    }
    QA_PAYLOAD = {"question": "What is pratyaksha?"}

    calls = {
        "/api/infer":          lambda: client.post("/api/infer", json=INFER_PAYLOAD),
        "/api/compare":        lambda: client.post("/api/compare", json=COMPARE_PAYLOAD),
        "/api/rag/search":     lambda: client.post("/api/rag/search", json=RAG_SEARCH_PAYLOAD),
        "/api/question-solve": lambda: client.post("/api/question-solve", json=QA_PAYLOAD),
    }

    print("\nPramana Engine - API Latency Benchmark")
    print("(LLM-free mode: symbolic reasoning + retrieval only)\n")

    rows: List[Tuple[str, ...]] = []
    speedups: dict[str, float] = {}

    for endpoint, fn in calls.items():
        cold = _timeit(fn, 1)
        warm = _timeit(fn, n)
        rows.append(_row(endpoint, "cold", cold))
        rows.append(_row(endpoint, "warm", warm))
        cold_avg = statistics.mean(cold)
        warm_avg = statistics.mean(warm)
        speedups[endpoint] = cold_avg / warm_avg if warm_avg > 0 else 0.0

    _print_table(rows)

    print("\nCache speedup (cold avg / warm avg):")
    for ep, s in speedups.items():
        print(f"  {ep:<25} {s:.1f}x")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
