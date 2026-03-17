"""End-to-end pipeline test — the exact Speaklar assessment scenario."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline import RAGPipeline


def test_assessment_scenario():
    """
    The exact test from the job posting:
    Q1: আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?
    Q2: দাম কত টাকা?

    Q2 must:
    1. Resolve coreference (understand Q2 refers to নুডুলস)
    2. Return correct noodle prices
    3. Complete in <100ms
    """
    pipeline = RAGPipeline()
    pipeline.initialize()

    # Q1: Ask about noodles
    q1 = "আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?"
    response1, results1, metrics1 = pipeline.process_query(q1, use_llm=False)

    print(f"\nQ1: {q1}")
    print(f"A1: {response1}")
    print(f"Q1 Metrics: {metrics1.summary()}")

    assert len(results1) > 0, "Q1 returned no results"
    assert any("নুডুলস" in r["name_bn"] for r in results1), "Q1 didn't find noodles"

    # Verify topic was tracked
    assert pipeline.state.current_topic is not None, "Topic not tracked after Q1"
    print(f"Tracked topic: {pipeline.state.current_topic}")

    # Q2: Ask about price (coreference!)
    q2 = "দাম কত টাকা?"
    response2, results2, metrics2 = pipeline.process_query(q2, use_llm=False, force_template=True)

    print(f"\nQ2: {q2}")
    print(f"A2: {response2}")
    print(f"Q2 Metrics: {metrics2.summary()}")

    # Verify coreference resolution
    assert metrics2.was_enriched, "Q2 was NOT enriched — coreference failed"
    assert "নুডুলস" in metrics2.enriched_query, (
        f"Enriched query missing নুডুলস: {metrics2.enriched_query}"
    )

    # Verify correct products returned
    assert len(results2) > 0, "Q2 returned no results"
    has_noodle = any("নুডুলস" in r["name_bn"] for r in results2)
    assert has_noodle, f"Q2 didn't return noodle products: {[r['name_bn'] for r in results2]}"

    # Verify response mentions price
    assert "৳" in response2 or "টাকা" in response2 or "মূল্য" in response2, (
        f"Q2 response doesn't mention price: {response2}"
    )

    # THE CRITICAL CHECK: <100ms
    assert metrics2.total_ms < 100, (
        f"Q2 took {metrics2.total_ms:.2f}ms — EXCEEDS 100ms limit!"
    )

    print(f"\n✅ Assessment PASSED — Q2 in {metrics2.total_ms:.2f}ms (<100ms)")


def test_multi_turn_topic_switch():
    """Test that topic switches correctly across multiple turns."""
    pipeline = RAGPipeline()
    pipeline.initialize()

    # Ask about noodles
    pipeline.process_query("নুডুলস আছে?", use_llm=False)
    assert "নুডুলস" in (pipeline.state.current_topic or "")

    # Ask about phones
    pipeline.process_query("স্মার্টফোন দেখান", use_llm=False)
    assert "স্মার্টফোন" in (pipeline.state.current_topic or "")

    # Follow-up about price should reference phones, not noodles
    _, _, metrics = pipeline.process_query("দাম কত?", use_llm=False, force_template=True)
    assert "স্মার্টফোন" in metrics.enriched_query or "ফোন" in metrics.enriched_query, (
        f"Topic didn't switch: {metrics.enriched_query}"
    )


if __name__ == "__main__":
    test_assessment_scenario()
    print("\n" + "=" * 40)
    test_multi_turn_topic_switch()
    print("✅ test_multi_turn_topic_switch")
    print("\n✅ All pipeline tests passed!")
