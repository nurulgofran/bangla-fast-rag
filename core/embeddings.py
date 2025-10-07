"""
Embedding model management — ONNX Runtime backend for fast, thread-safe inference.
Uses paraphrase-multilingual-MiniLM-L12-v2 with ONNX optimization.
"""
import json
import numpy as np
import torch
import os

# Optimize PyTorch CPU threading to prevent massive thread-contention
# inside Python ThreadPools (Gradio background workers)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION


class EmbeddingModel:
    """ONNX-optimized embedding model wrapper."""

    def __init__(self):
        self.model: SentenceTransformer | None = None
        self.model_name = EMBEDDING_MODEL_NAME
        self.dimension = EMBEDDING_DIMENSION

    def load(self) -> None:
        """Load the embedding model with ONNX backend. Call at startup."""
        if self.model is None:
            print(f"Loading embedding model (PyTorch): {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"Embedding model loaded ({self.dimension}-dim)")

    def encode(self, texts: str | list[str], batch_size: int = 128) -> np.ndarray:
        """Encode text(s) into normalized embeddings."""
        if self.model is None:
            self.load()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query — optimized for speed."""
        return self.encode(query)


# Global instance
embedding_model = EmbeddingModel()
