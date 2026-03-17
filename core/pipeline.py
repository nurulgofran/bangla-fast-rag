"""
Main RAG Pipeline Orchestrator.
Wires together: query enrichment → embedding → FAISS search → response generation.
Includes timing instrumentation for benchmarking.
"""
import time
from dataclasses import dataclass, field

from core.embeddings import embedding_model
from core.indexer import product_index
from core.conversation import ConversationState, enrich_query
from core.responder import classify_query, template_response, llm_response


@dataclass
class QueryMetrics:
    """Timing metrics for a single query."""
    enrichment_ms: float = 0.0
    embedding_ms: float = 0.0
    search_ms: float = 0.0
    response_ms: float = 0.0
    total_ms: float = 0.0
    was_enriched: bool = False
    enriched_query: str = ""
    response_type: str = ""  # 'template' or 'llm'

    def summary(self) -> str:
        return (
            f"⏱️ Enrichment: {self.enrichment_ms:.2f}ms | "
            f"Embedding: {self.embedding_ms:.2f}ms | "
            f"Search: {self.search_ms:.2f}ms | "
            f"Response: {self.response_ms:.2f}ms | "
            f"Total: {self.total_ms:.2f}ms "
            f"[{self.response_type}]"
        )


class RAGPipeline:
    """
    Context-aware Bangla RAG pipeline.

    For follow-up queries (Q2), the hot path is:
        enrichment (0ms) → embedding (~5ms) → FAISS (<1ms) → template (<1ms)
        Total: ~6-12ms
    """

    def __init__(self):
        self.state = ConversationState()
        self._initialized = False

    def initialize(self) -> None:
        """Load models and index. Call once at startup."""
        if self._initialized:
            return
        embedding_model.load()
        product_index.load_index()

        # Warm up the embedding model (eliminates cold-start latency)
        # First query after load is ~180ms, subsequent are ~10ms
        print("🔄 Warming up embedding model...")
        for _ in range(5):
            embedding_model.encode_query("পরীক্ষা নুডুলস দাম কত টাকা")
        print("✅ Model warmed up")

        self._initialized = True
        print("✅ RAG Pipeline initialized")

    def process_query(
        self,
        query: str,
        use_llm: bool = True,
        force_template: bool = False,
    ) -> tuple[str, list[dict], QueryMetrics]:
        """
        Process a user query through the RAG pipeline.

        Args:
            query: User's input query in Bangla
            use_llm: If True, use LLM for Q1/conversational queries
            force_template: If True, always use template response (for benchmarking)

        Returns:
            (response_text, retrieved_products, metrics)
        """
        if not self._initialized:
            self.initialize()

        metrics = QueryMetrics()
        total_start = time.perf_counter()

        # Step 1: Query Enrichment (coreference resolution)
        t0 = time.perf_counter()
        enriched_query, was_enriched = enrich_query(query, self.state)
        metrics.enrichment_ms = (time.perf_counter() - t0) * 1000
        metrics.was_enriched = was_enriched
        metrics.enriched_query = enriched_query

        # Step 2: Embed the query
        t0 = time.perf_counter()
        query_embedding = embedding_model.encode_query(enriched_query)
        metrics.embedding_ms = (time.perf_counter() - t0) * 1000

        # Step 3: FAISS Search
        t0 = time.perf_counter()
        results = product_index.search(query_embedding, query_text=enriched_query)
        metrics.search_ms = (time.perf_counter() - t0) * 1000

        # Step 4: Response Generation
        t0 = time.perf_counter()
        query_type = classify_query(enriched_query)

        is_first_query = len(self.state.history) == 0
        is_structured = query_type in ("price", "availability", "detail")

        if force_template or (is_structured and not is_first_query):
            # HOT PATH: Template response (<1ms)
            response = template_response(query_type, results)
            metrics.response_type = "template"
        elif use_llm:
            # SLOW PATH: LLM response (not time-critical)
            try:
                response = llm_response(enriched_query, results, self.state.history)
                metrics.response_type = "llm"
            except Exception as e:
                # Fallback to template if LLM fails
                response = template_response(query_type, results)
                metrics.response_type = "template_fallback"
        else:
            response = template_response(query_type, results)
            metrics.response_type = "template"

        metrics.response_ms = (time.perf_counter() - t0) * 1000

        # Step 5: Update conversation state
        self.state.add_turn("user", query)
        self.state.add_turn("assistant", response)
        self.state.update_topic_from_results(results, user_query=query)

        metrics.total_ms = (time.perf_counter() - total_start) * 1000

        return response, results, metrics

    def reset(self) -> None:
        """Reset conversation state."""
        self.state.reset()


# Global instance
rag_pipeline = RAGPipeline()
