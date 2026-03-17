"""
Performance Benchmark — runs the Speaklar assessment scenario 100 times.
Reports: mean, median, p95, p99 latencies per step.
"""
import time
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline import RAGPipeline


def run_benchmark(iterations: int = 100) -> None:
    """Run the Q1→Q2 scenario and measure Q2 latency."""
    print("=" * 60)
    print("Speaklar Bangla RAG - Performance Benchmark")
    print("=" * 60)

    pipeline = RAGPipeline()
    pipeline.initialize()

    # Warm-up (first run is always slower due to caching)
    print("\nWarming up...")
    pipeline.reset()
    pipeline.process_query("আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?", use_llm=False)
    pipeline.process_query("দাম কত টাকা?", use_llm=False)
    pipeline.reset()

    q2_totals = []
    q2_enrichments = []
    q2_embeddings = []
    q2_searches = []
    q2_responses = []
    correct_count = 0

    print(f"\nRunning {iterations} iterations...\n")

    for i in range(iterations):
        pipeline.reset()

        pipeline.process_query(
            "আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?",
            use_llm=False,
        )

        response, results, metrics = pipeline.process_query(
            "দাম কত টাকা?",
            use_llm=False,
            force_template=True,
        )

        q2_totals.append(metrics.total_ms)
        q2_enrichments.append(metrics.enrichment_ms)
        q2_embeddings.append(metrics.embedding_ms)
        q2_searches.append(metrics.search_ms)
        q2_responses.append(metrics.response_ms)

        # Check correctness: response should mention নুডুলস
        if "নুডুলস" in response.lower() or "নুডুলস" in response:
            correct_count += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{iterations}] Latest Q2: {metrics.total_ms:.2f}ms")

    def stats(values):
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": sorted(values)[int(len(values) * 0.95)],
            "p99": sorted(values)[int(len(values) * 0.99)],
            "min": min(values),
            "max": max(values),
        }

    total = stats(q2_totals)
    enrich = stats(q2_enrichments)
    embed = stats(q2_embeddings)
    search = stats(q2_searches)
    resp = stats(q2_responses)

    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS — Q2 Latency (ms)")
    print("=" * 60)
    print(f"\n{'Step':<20} {'Mean':>8} {'Median':>8} {'P95':>8} {'P99':>8} {'Min':>8} {'Max':>8}")
    print("-" * 80)

    for name, s in [
        ("Enrichment", enrich),
        ("Embedding", embed),
        ("FAISS Search", search),
        ("Response Gen", resp),
        ("TOTAL", total),
    ]:
        print(
            f"{name:<20} {s['mean']:>8.2f} {s['median']:>8.2f} "
            f"{s['p95']:>8.2f} {s['p99']:>8.2f} {s['min']:>8.2f} {s['max']:>8.2f}"
        )

    print(f"\nCorrectness: {correct_count}/{iterations} ({correct_count/iterations*100:.1f}%)")
    print(f"Median Q2 latency: {total['median']:.2f}ms")

    if total['median'] < 100:
        print(f"PASS - Median under 100ms (target)")
    else:
        print(f"FAIL - Median exceeds 100ms")

    if total['p99'] < 100:
        print(f"PASS - P99 under 100ms")
    else:
        print(f"NOTE - P99 exceeds 100ms ({total['p99']:.2f}ms)")

    print()


if __name__ == "__main__":
    run_benchmark(iterations=100)
