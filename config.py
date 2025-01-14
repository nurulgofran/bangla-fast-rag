"""
Configuration constants for the Speaklar Bangla RAG system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PRODUCTS_FILE = DATA_DIR / "products.json"
FAISS_INDEX_FILE = DATA_DIR / "products.index"
EMBEDDINGS_FILE = DATA_DIR / "products_embeddings.npy"

# Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384

# FAISS
TOP_K_RESULTS = 20

# Groq LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MAX_TOKENS = 256
GROQ_TEMPERATURE = 0.3

# Conversation
MAX_HISTORY_TURNS = 5

# Dataset Generation
NUM_PRODUCTS = 5000

# Server
GRADIO_PORT = 7860
