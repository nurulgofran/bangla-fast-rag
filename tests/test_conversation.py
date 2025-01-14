"""Tests for coreference resolution via conversation state & query enrichment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.conversation import ConversationState, enrich_query


def test_no_enrichment_on_first_query():
    """First query should not be enriched."""
    state = ConversationState()
    enriched, was_enriched = enrich_query("আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?", state)
    assert not was_enriched
    assert "নুডুলস" in enriched


def test_enrichment_with_topic():
    """Follow-up query should be enriched with current topic."""
    state = ConversationState()
    state.add_turn("user", "আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?")
    state.add_turn("assistant", "হ্যাঁ, আমাদের কাছে নুডুলস আছে।")
    state.update_topic_from_results([
        {"name_bn": "ম্যাগি নুডুলস", "price_bdt": 35, "category_bn": "খাদ্য",
         "description_bn": "সুস্বাদু নুডুলস", "id": 1}
    ])

    enriched, was_enriched = enrich_query("দাম কত টাকা?", state)
    assert was_enriched, f"Expected enrichment, got: {enriched}"
    assert "নুডুলস" in enriched, f"Expected 'নুডুলস' in enriched query: {enriched}"


def test_no_enrichment_when_product_mentioned():
    """Query that already mentions a product should not be enriched."""
    state = ConversationState()
    state.add_turn("user", "নুডুলস আছে?")
    state.add_turn("assistant", "হ্যাঁ আছে।")
    state.update_topic_from_results([
        {"name_bn": "ম্যাগি নুডুলস", "price_bdt": 35, "category_bn": "খাদ্য",
         "description_bn": "test", "id": 1}
    ])

    enriched, was_enriched = enrich_query("ম্যাগি নুডুলস কত টাকা?", state)
    assert not was_enriched


def test_topic_tracking():
    """Topic should update based on retrieval results."""
    state = ConversationState()
    assert state.current_topic is None

    state.update_topic_from_results([
        {"name_bn": "ম্যাগি নুডুলস", "price_bdt": 35},
    ])
    assert state.current_topic == "ম্যাগি নুডুলস"

    state.update_topic_from_results([
        {"name_bn": "স্মার্টফোন", "price_bdt": 15000},
    ])
    assert state.current_topic == "স্মার্টফোন"


def test_history_trimming():
    """History should be trimmed to max turns."""
    state = ConversationState()
    for i in range(20):
        state.add_turn("user", f"Question {i}")
        state.add_turn("assistant", f"Answer {i}")

    # Default MAX_HISTORY_TURNS=5, so 10 entries max
    assert len(state.history) == 10


def test_reset():
    """Reset should clear all state."""
    state = ConversationState()
    state.add_turn("user", "test")
    state.current_topic = "test"
    state.reset()
    assert len(state.history) == 0
    assert state.current_topic is None


if __name__ == "__main__":
    test_no_enrichment_on_first_query()
    print("✅ test_no_enrichment_on_first_query")

    test_enrichment_with_topic()
    print("✅ test_enrichment_with_topic")

    test_no_enrichment_when_product_mentioned()
    print("✅ test_no_enrichment_when_product_mentioned")

    test_topic_tracking()
    print("✅ test_topic_tracking")

    test_history_trimming()
    print("✅ test_history_trimming")

    test_reset()
    print("✅ test_reset")

    print("\n✅ All conversation tests passed!")
