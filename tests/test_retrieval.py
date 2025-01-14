"""Tests for hybrid retrieval accuracy."""
import sys
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.embeddings import embedding_model
from core.indexer import product_index


def setup():
    embedding_model.load()
    product_index.load_index()


def test_noodle_retrieval():
    """Searching for নুডুলস should return noodle products."""
    query = "নুডুলস"
    embedding = embedding_model.encode_query(query)
    results = product_index.search(embedding, top_k=5, query_text=query)

    assert len(results) > 0, "No results returned for নুডুলস"
    names = [r["name_bn"] for r in results]
    has_noodle = any("নুডুলস" in name for name in names)
    assert has_noodle, f"No noodle product found. Got: {names}"
    print(f"  → Top results: {names}")


def test_price_retrieval():
    """Searching for নুডুলসের দাম should return noodle products with prices."""
    query = "নুডুলসের দাম কত টাকা?"
    embedding = embedding_model.encode_query(query)
    results = product_index.search(embedding, top_k=3, query_text=query)

    assert len(results) > 0, "No results returned"
    names = [r["name_bn"] for r in results]
    has_noodle = any("নুডুলস" in name for name in names)
    assert has_noodle, f"নুডুলস not in results: {names}"
    print(f"  → Top results: {[(r['name_bn'], r['price_bdt']) for r in results]}")


def test_electronics_retrieval():
    """Searching for electronics should return electronics products."""
    query = "স্মার্টফোন"
    embedding = embedding_model.encode_query(query)
    results = product_index.search(embedding, top_k=3, query_text=query)

    assert len(results) > 0, "No results returned"
    names = [r["name_bn"] for r in results]
    has_phone = any("স্মার্টফোন" in name or "ফোন" in name for name in names)
    assert has_phone, f"No phone product found. Got: {names}"
    print(f"  → Top results: {names}")


def test_search_speed():
    """Hybrid search should complete in <5ms."""
    query = "নুডুলস"
    embedding = embedding_model.encode_query(query)
    product_index.search(embedding, query_text=query)  # warm-up

    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        product_index.search(embedding, query_text=query)
        times.append((time.perf_counter() - t0) * 1000)

    median_ms = statistics.median(times)
    print(f"  → Hybrid search median: {median_ms:.3f}ms")
    assert median_ms < 5, f"Search too slow: {median_ms:.3f}ms"


if __name__ == "__main__":
    setup()
    test_noodle_retrieval()
    print("✅ test_noodle_retrieval")
    test_price_retrieval()
    print("✅ test_price_retrieval")
    test_electronics_retrieval()
    print("✅ test_electronics_retrieval")
    test_search_speed()
    print("✅ test_search_speed")
    print("\n✅ All retrieval tests passed!")
