"""
Hybrid product search — keyword matching + FAISS semantic search.
Keyword matching is the primary strategy for Bangla (better accuracy).
FAISS provides semantic fallback for complex queries.
"""
import json
import re
import numpy as np
import faiss
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FAISS_INDEX_FILE, EMBEDDINGS_FILE, PRODUCTS_FILE, TOP_K_RESULTS
from core.embeddings import embedding_model


class ProductIndex:
    """Hybrid product search index: keyword + FAISS."""

    def __init__(self):
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.products: list[dict] = []
        self.product_texts: list[str] = []

    @staticmethod
    def product_to_text(product: dict) -> str:
        """Convert product dict to searchable text."""
        return (
            f"{product['name_bn']} | {product['category_bn']} | "
            f"{product['description_bn']} | মূল্য: {product['price_bdt']} টাকা"
        )

    def build_index(self) -> None:
        """Build FAISS index from products.json. Run once offline."""
        print("Loading products...")
        with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
            self.products = json.load(f)

        self.product_texts = [self.product_to_text(p) for p in self.products]

        print(f"Encoding {len(self.products)} products...")
        embedding_model.load()
        embeddings = embedding_model.encode(self.product_texts)

        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)

        faiss.write_index(self.faiss_index, str(FAISS_INDEX_FILE))
        np.save(str(EMBEDDINGS_FILE), embeddings)

        print(f"FAISS index built: {self.faiss_index.ntotal} vectors ({dimension}-dim)")

    def load_index(self) -> None:
        """Load pre-built FAISS index and products."""
        if self.faiss_index is not None:
            return

        print("Loading FAISS index...")
        self.faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))

        with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
            self.products = json.load(f)

        self.product_texts = [self.product_to_text(p) for p in self.products]
        print(f"FAISS index loaded: {self.faiss_index.ntotal} vectors")

    def _keyword_search(self, query: str, top_k: int) -> list[dict]:
        """
        Fast keyword-based search with Bangla suffix handling.
        Runs in <2ms for 5000 products.
        """
        query_words = set(re.split(r'\s+', query.strip()))
        stop_words = {"কি", "কী", "কত", "কোন", "আপনাদের", "কোম্পানি", "বিক্রি", "করে",
                      "আছে", "দাম", "টাকা", "এর", "এটা", "সেটা", "মূল্য", "পাওয়া",
                      "যায়", "হয়", "একটি", "একটা", "দিন", "বলুন", "জানান", "দেখান"}
        search_words = query_words - stop_words

        if not search_words:
            search_words = query_words

        # Expand search words with Bangla suffix stripping
        expanded_words = set()
        bangla_suffixes = ["ের", "ে", "তে", "র", "গুলো", "গুলি", "সমূহ", "টি", "টা", "খানা"]
        for word in search_words:
            expanded_words.add(word)
            for suffix in bangla_suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    stem = word[:-len(suffix)]
                    expanded_words.add(stem)

        scored = []
        for i, product in enumerate(self.products):
            score = 0
            name = product["name_bn"]
            text = self.product_texts[i]

            for word in expanded_words:
                if len(word) > 2:
                    # Exact substring match in name (highest priority)
                    if word in name:
                        score += 10
                    # Substring match in full text
                    elif word in text:
                        score += 3
                    # Check if product name contains this word
                    elif any(word in pword for pword in name.split()):
                        score += 8

            # Category match
            for word in expanded_words:
                if len(word) > 2 and word in product.get("category_bn", ""):
                    score += 5

            if score > 0:
                result = product.copy()
                result["similarity_score"] = score / 10.0
                scored.append(result)

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored[:top_k]

    def _faiss_search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        """FAISS semantic search (fallback)."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.faiss_index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            product = self.products[idx].copy()
            product["similarity_score"] = float(score)
            results.append(product)
        return results

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K_RESULTS,
        query_text: str = "",
    ) -> list[dict]:
        """
        Hybrid search: keyword first, FAISS fallback.
        - If keyword search finds matches: use keyword results
        - If no keyword matches: fallback to FAISS semantic search
        """
        if self.faiss_index is None:
            self.load_index()

        # Try keyword search first (fast, accurate for Bangla)
        if query_text:
            keyword_results = self._keyword_search(query_text, top_k)
            if keyword_results:
                return keyword_results

        # Fallback to FAISS
        return self._faiss_search(query_embedding, top_k)


# Global instance
product_index = ProductIndex()
