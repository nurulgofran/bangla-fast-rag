"""
Embedding model management — ONNX Runtime backend for fast, thread-safe inference.
Uses paraphrase-multilingual-MiniLM-L12-v2 with ONNX optimization.
"""
import numpy as np
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
            print(f"Loading embedding model (ONNX): {self.model_name}...")
            import platform
            try:
                # Use ARM64 quantized model on Apple Silicon for best speed
                file_name = "onnx/model.onnx"
                if platform.machine() == "arm64":
                    file_name = "onnx/model_qint8_arm64.onnx"
                    print(f"  → Using ARM64 INT8 quantized model")

                self.model = SentenceTransformer(
                    self.model_name,
                    backend="onnx",
                    model_kwargs={
                        "provider": "CPUExecutionProvider",
                        "file_name": file_name,
                    },
                )
                print(f"Embedding model loaded (ONNX, {self.dimension}-dim)")
            except Exception as e:
                print(f"ONNX failed ({e}), falling back to PyTorch...")
                self.model = SentenceTransformer(self.model_name)
                print(f"Embedding model loaded (PyTorch fallback, {self.dimension}-dim)")

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
