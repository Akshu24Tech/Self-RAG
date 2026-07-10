"""
Concurrent Load Test for Self-RAG Pipeline
==========================================
Measures:
  1. Latency under concurrency (min/avg/p95/max, throughput)
  2. IsSUP grounding accuracy stability under load
  3. Revision loop (retries) backpressure effects

Uses asyncio + ThreadPoolExecutor because LangGraph invoke() is synchronous.
"""

import asyncio
import json
import time
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from self_rag import SelfRAG

# ── Test Queries ─────────────────────────────────────────────────────────
# Mix of document-grounded (quantum_computing.pdf) and general knowledge
TEST_QUERIES = [
    # Document-grounded queries (need retrieval)
    "What is quantum computing?",
    "How do qubits differ from classical bits?",
    "What is quantum superposition?",
    "Explain quantum entanglement.",
    "What are the applications of quantum computing?",
    "What is a quantum gate?",
    "How does quantum error correction work?",
    "What is the current state of quantum computing hardware?",
    # General knowledge queries (should skip retrieval)
    "What is 2 + 2?",
    "Who wrote Romeo and Juliet?",
]


@dataclass
class QueryResult:
    """Result from a single query execution."""
    query: str
    latency_s: float
    success: bool
    issup: Optional[str] = None
    retries: int = 0
    rewrite_tries: int = 0
    need_retrieval: Optional[bool] = None
    isuse: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ConcurrencyLevelResult:
    """Aggregated results for one concurrency level."""
    concurrency: int
    total_time_s: float
    results: List[QueryResult] = field(default_factory=list)

    @property
    def latencies(self) -> List[float]:
        return [r.latency_s for r in self.results if r.success]

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def throughput(self) -> float:
        return self.success_count / self.total_time_s if self.total_time_s > 0 else 0

    def latency_stats(self) -> dict:
        lats = self.latencies
        if not lats:
            return {"min": 0, "avg": 0, "p95": 0, "max": 0}
        lats_sorted = sorted(lats)
        p95_idx = int(len(lats_sorted) * 0.95)
        return {
            "min": round(lats_sorted[0], 2),
            "avg": round(statistics.mean(lats_sorted), 2),
            "p95": round(lats_sorted[min(p95_idx, len(lats_sorted) - 1)], 2),
            "max": round(lats_sorted[-1], 2),
        }

    def issup_distribution(self) -> dict:
        dist = {"fully_supported": 0, "partially_supported": 0, "no_support": 0, "N/A": 0}
        for r in self.results:
            if r.success and r.issup:
                dist[r.issup] = dist.get(r.issup, 0) + 1
            elif r.success:
                dist["N/A"] += 1
        return dist

    def retry_stats(self) -> dict:
        retries = [r.retries for r in self.results if r.success]
        rewrites = [r.rewrite_tries for r in self.results if r.success]
        queries_with_retries = sum(1 for r in retries if r > 0)
        queries_with_rewrites = sum(1 for r in rewrites if r > 0)

        # Latency comparison: queries with vs without retries
        lat_with_retries = [r.latency_s for r in self.results if r.success and r.retries > 0]
        lat_without_retries = [r.latency_s for r in self.results if r.success and r.retries == 0]

        return {
            "queries_with_retries": queries_with_retries,
            "queries_with_rewrites": queries_with_rewrites,
            "total_retries": sum(retries),
            "total_rewrites": sum(rewrites),
            "avg_latency_with_retries": round(statistics.mean(lat_with_retries), 2) if lat_with_retries else 0,
            "avg_latency_without_retries": round(statistics.mean(lat_without_retries), 2) if lat_without_retries else 0,
        }


def run_single_query(rag: SelfRAG, query: str, query_idx: int) -> QueryResult:
    """Execute a single query through the Self-RAG pipeline and capture metrics."""
    start = time.perf_counter()
    try:
        result = rag.app.invoke({"question": query})
        elapsed = time.perf_counter() - start

        return QueryResult(
            query=query,
            latency_s=round(elapsed, 3),
            success=True,
            issup=result.get("issup"),
            retries=result.get("retries", 0),
            rewrite_tries=result.get("rewrite_tries", 0),
            need_retrieval=result.get("need_retrieval"),
            isuse=result.get("isuse"),
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return QueryResult(
            query=query,
            latency_s=round(elapsed, 3),
            success=False,
            error=str(e),
        )


async def run_concurrency_level(
    rag: SelfRAG,
    concurrency: int,
    executor: ThreadPoolExecutor,
) -> ConcurrencyLevelResult:
    """Fire `concurrency` queries simultaneously using the thread pool."""
    loop = asyncio.get_event_loop()

    # Build the query list: cycle through TEST_QUERIES to fill `concurrency` slots
    queries = [TEST_QUERIES[i % len(TEST_QUERIES)] for i in range(concurrency)]

    print(f"\n{'='*70}")
    print(f"  CONCURRENCY LEVEL: {concurrency} queries")
    print(f"{'='*70}")

    wall_start = time.perf_counter()

    # Fire all queries concurrently via thread pool
    tasks = [
        loop.run_in_executor(executor, run_single_query, rag, q, i)
        for i, q in enumerate(queries)
    ]

    # Gather results as they complete
    results = []
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        status = "✓" if result.success else "✗"
        retry_info = f" [retries={result.retries}]" if result.retries > 0 else ""
        print(f"  [{i+1:2d}/{concurrency}] {status} {result.latency_s:6.2f}s | "
              f"issup={result.issup or 'N/A':20s}{retry_info} | "
              f"{result.query[:45]}...")
        results.append(result)

    wall_elapsed = time.perf_counter() - wall_start

    return ConcurrencyLevelResult(
        concurrency=concurrency,
        total_time_s=round(wall_elapsed, 2),
        results=results,
    )


def print_summary(all_results: List[ConcurrencyLevelResult]):
    """Print a formatted summary table of all concurrency levels."""
    print("\n\n" + "=" * 90)
    print("  LOAD TEST SUMMARY")
    print("=" * 90)

    # ── Latency Table ──
    print("\n┌─────────────┬──────────┬──────────┬──────────┬──────────┬────────────┬──────────┐")
    print("│ Concurrency │   Min(s) │   Avg(s) │   P95(s) │   Max(s) │ Throughput │ Failures │")
    print("├─────────────┼──────────┼──────────┼──────────┼──────────┼────────────┼──────────┤")
    for r in all_results:
        stats = r.latency_stats()
        print(f"│ {r.concurrency:>11d} │ {stats['min']:>8.2f} │ {stats['avg']:>8.2f} │ "
              f"{stats['p95']:>8.2f} │ {stats['max']:>8.2f} │ {r.throughput:>8.2f}/s │ "
              f"{r.failure_count:>8d} │")
    print("└─────────────┴──────────┴──────────┴──────────┴──────────┴────────────┴──────────┘")

    # ── IsSUP Grounding Distribution ──
    print("\n  IsSUP Grounding Distribution:")
    print("┌─────────────┬─────────────────┬─────────────────────┬────────────┬──────┐")
    print("│ Concurrency │ Fully Supported │ Partially Supported │ No Support │  N/A │")
    print("├─────────────┼─────────────────┼─────────────────────┼────────────┼──────┤")
    for r in all_results:
        dist = r.issup_distribution()
        print(f"│ {r.concurrency:>11d} │ {dist['fully_supported']:>15d} │ "
              f"{dist['partially_supported']:>19d} │ {dist['no_support']:>10d} │ "
              f"{dist['N/A']:>4d} │")
    print("└─────────────┴─────────────────┴─────────────────────┴────────────┴──────┘")

    # ── Revision Loop Backpressure ──
    print("\n  Revision Loop Backpressure:")
    print("┌─────────────┬──────────┬──────────┬───────────────────┬──────────────────────┐")
    print("│ Concurrency │ w/Retry  │ w/Rewrt  │ Avg Lat (w/retry) │ Avg Lat (no retry)   │")
    print("├─────────────┼──────────┼──────────┼───────────────────┼──────────────────────┤")
    for r in all_results:
        rs = r.retry_stats()
        print(f"│ {r.concurrency:>11d} │ {rs['queries_with_retries']:>8d} │ "
              f"{rs['queries_with_rewrites']:>8d} │ {rs['avg_latency_with_retries']:>15.2f}s  │ "
              f"{rs['avg_latency_without_retries']:>18.2f}s  │")
    print("└─────────────┴──────────┴──────────┴───────────────────┴──────────────────────┘")


async def main():
    print("=" * 70)
    print("  Self-RAG Concurrent Load Test")
    print("=" * 70)
    print("\nInitializing Self-RAG pipeline...")

    rag = SelfRAG(documents_folder="documents")

    # Concurrency levels to test
    levels = [1, 5, 10, 20, 30]

    # Thread pool: sized to max concurrency
    max_workers = max(levels)
    executor = ThreadPoolExecutor(max_workers=max_workers)

    all_results: List[ConcurrencyLevelResult] = []

    for level in levels:
        result = await run_concurrency_level(rag, level, executor)
        all_results.append(result)

        # Brief cooldown between levels to avoid rate limit carryover
        if level != levels[-1]:
            cooldown = 5
            print(f"\n  ⏳ Cooling down {cooldown}s before next level...")
            await asyncio.sleep(cooldown)

    executor.shutdown(wait=False)

    # Print summary
    print_summary(all_results)

    # Save detailed results to JSON
    output = {
        "test_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test_queries": TEST_QUERIES,
        "concurrency_levels": [],
    }
    for r in all_results:
        level_data = {
            "concurrency": r.concurrency,
            "total_wall_time_s": r.total_time_s,
            "latency_stats": r.latency_stats(),
            "throughput_qps": round(r.throughput, 2),
            "issup_distribution": r.issup_distribution(),
            "retry_stats": r.retry_stats(),
            "success_count": r.success_count,
            "failure_count": r.failure_count,
            "individual_results": [asdict(qr) for qr in r.results],
        }
        output["concurrency_levels"].append(level_data)

    with open("load_test_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Detailed results saved to load_test_results.json")
    print(f"   Run completed at {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    # Suppress Self-RAG's verbose prints during load test
    import io
    import contextlib

    # Run without suppressing output so we can see progress
    asyncio.run(main())
