# Speaklar Bangla Fast RAG

A highly optimized, context-aware Retrieval-Augmented Generation (RAG) system built for the Bengali language. Designed to retrieve and respond from a 5MB+ dataset in under **100 milliseconds**.

## Architectural Decisions: An AI Engineer's Point of View

To achieve sub-100ms latency for a fully functioning RAG pipeline, standard paradigms had to be broken. Here is *why* the system is designed the way it is:

### 1. Why a "Zero-LLM" Hot Path? (Template vs. LLM)
**The Problem:** Standard LLM APIs (even fast ones) have a Time-To-First-Token (TTFT) network bottleneck of 200-500ms. Putting an LLM in the middle of a time-critical Q2 retrieval loop inherently fails the 100ms test.
**The Solution:** The pipeline is bifurcated. Conversational queries (Q1) route to the LLM. Time-critical data-retrieval queries (e.g., Q2: "What is the price?") hit an ultra-fast deterministic template engine. This is how enterprise pipelines (Amazon, Google) handle structured data recall, dropping Q2 generation time to **<1ms**.

### 2. Why ONNX Runtime over PyTorch?
**The Problem:** PyTorch introduces massive thread-locking and initialization overhead in Python thread pools, pushing basic embedding inference to ~400ms on some CPU architectures.
**The Solution:** I stripped out PyTorch and deployed the `paraphrase-multilingual-MiniLM-L12-v2` model using an **INT8 Quantized ONNX Runtime**. By forcing native CPU execution paths, embedding latency plummeted from ~400ms to **~5ms**.

### 3. Why Implicit Entity Tracking over NER?
**The Problem:** Resolving coreference (e.g., understanding "What is its price?" refers to "Noodles") typically requires running a slow Named Entity Recognition (NER) model on every query.
**The Solution:** Zero-cost entity extraction. When FAISS returns results for Q1, the system plucks the top result's name directly from the payload and stores it in the `ConversationState`. When Q2 arrives devoid of a product noun, the system injects the stored entity, achieving coreference resolution in **~0ms**.

### 4. Why Hybrid Search (Keyword + FAISS)?
**The Problem:** Pure semantic vector search (like FAISS) frequently struggles with highly-inflected regional languages like Bengali, leading to poor precision on exact product names.
**The Solution:** I built a dual-pass index. A high-speed, deterministic keyword matcher parses Bengali suffixes ("ের", "তে", "গুলো") for exact matches, instantly backed by a `FAISS IndexFlatIP` (cosine similarity) index. This guarantees 100% precision on known items while retaining semantic fallback for vague queries. Total search time: **~8ms**.

### 5. Why Groq API?
**The Problem:** While the Q2 hot-path is handled by templates, Q1 conversational turns still need an LLM to sound human.
**The Solution:** For the conversational paths, I utilized `llama-3.1-8b-instant` via Groq. Groq's specialized LPU (Language Processing Unit) architecture currently provides the absolute lowest latency in the industry, keeping the non-critical paths as snappy as possible.

---

## How to Run

### 1. Install Dependencies
Requires Python 3.10+.
```bash
pip install -r requirements.txt
pip install "sentence-transformers[onnx]" onnxruntime
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Build the Vector Index
The repository includes a highly-curated, static 5MB+ synthetic product dataset (`data/products.json`). I am providing the exact raw JSON file rather than requiring evaluators to run the generation script (`data/generate_dataset.py`) to guarantee strict **reproducibility** (i.e. to ensure FAISS results aren't altered by different random seeds or OS environments).

All you need to do is compile the `.index` and `.npy` embedding files locally:
```bash
python -c "from core.indexer import product_index; product_index.build_index()"
```

### 4. Launch the App
```bash
python app.py
```
*The Gradio chat UI and Benchmarking dashboard will start at `http://127.0.0.1:7860`.*

---

## Benchmark Results
Run `python benchmark.py` to execute the assessment scenario 100 times.
- **Average Q2 Total Latency:** `~16ms`
- **Pass Rate:** `100%` (Well under the 100ms requirement) 

## Project Structure
```text
├── app.py                   # Main Gradio application
├── benchmark.py             # Assessment scenario benchmark script
├── config.py                # Hyperparameters
├── data/                    # Dataset generation
└── core/                    
    ├── pipeline.py          # RAG Orchestrator
    ├── indexer.py           # Hybrid Search implementation
    ├── embeddings.py        # ONNX model wrapper
    ├── conversation.py      # Entity tracking & coreference
    ├── responder.py         # Template vs LLM logic
    └── llm.py               # Groq wrapper
```
