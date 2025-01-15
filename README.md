# Bangla Fast RAG System

A highly optimized context-aware Retrieval-Augmented Generation (RAG) system built for the Bengali language. This project was developed as a solution to the Speaklar NLP Engineer assessment, demonstrating the ability to retrieve from a 5MB+ dataset and generate context-aware responses in under 100 milliseconds.

## Key Features

- **Fast Retrieval (<20ms):** Achieves median processing times of ~12-16ms for follow-up queries, meeting the <100ms assessment requirement.
- **Context-Aware Coreference Resolution:** Automatically understands conversational context. (e.g., Q1: "Do you sell noodles?", Q2: "What is the price?" -> System understands Q2 refers to noodles).
- **5MB+ Synthetic Bangla Dataset:** Includes a custom script that generates 5,000 unique Bangla products (~5.4MB text data) across 15 categories.
- **Zero-LLM Hot Path Architecture:** Solved the "Time-To-First-Token" bottleneck of cloud LLMs (~300ms) by utilizing a pre-warmed ONNX embedding model (ARM64 INT8 quantized) and template-based deterministic responses for time-critical queries.
- **Gradio Chat & Benchmark UI:** Includes a full chat interface with real-time latency metrics and a built-in automated 20-product benchmark suite.

## The Challenge

> *"Develop a system capable of resolving coreference and accurately returning the price of noodles. The total processing time for Q2 including both retrieval and response generation must be under 100 milliseconds for the RAG response from a 5MB dataset."*

**The Problem with Standard RAG:** Standard LLM APIs (like Groq, OpenAI) have a TTFT (Time-To-First-Token) of 200-500ms. A naive approach of placing an LLM in the middle of a time-critical retrieval loop inherently fails the 100ms test.

**The Solution:** This system splits the pipeline:
1. **Conversational queries** are routed to an LLM asynchronously.
2. **Follow-up / Data-retrieval queries** (e.g., "What is the price?") are routed to an ultra-fast "Hot Path" that leverages proactive entity tracking, quantized purely-CPU ONNX embeddings (~5ms), and FAISS Hybrid Search (~8ms).

## Benchmark Results

Running the automated benchmark (`test_20_products.py`) across 20 completely random products from the 5,000 dataset:

| Metric | Result | Target |
| :--- | :--- | :--- |
| **Pass Rate** | **100%** (20/20) | - |
| **Average Q2 Latency** | **12.43ms** | < 100ms |
| **Max Q2 Latency** | **16.63ms** | < 100ms |
| **Dataset Size (Raw Text)** | **5.42 MB** | 5.0 MB |

## Tech Stack
- **Language:** Python 3.10+
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (Running natively via **ONNX Runtime** `qint8_arm64` for maximum CPU thread efficiency)
- **Vector Search:** `FAISS` (IndexFlatIP)
- **LLM:** `Groq API` (llama-3.1-8b-instant)
- **UI:** `Gradio`

## Getting Started

### Prerequisites
- Python 3.10+
- Groq API Key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bangla-fast-rag.git
cd bangla-fast-rag
```

2. Install dependencies:
```bash
# Core requirements including ONNX runtime for sentence-transformers
pip install -r requirements.txt
pip install "sentence-transformers[onnx]" onnxruntime
```

3. Set up environment variables:
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

4. Generate the dataset and build the vector index:
```bash
python data/generate_dataset.py
python -c "from core.indexer import product_index; product_index.build_index()"
```

5. Run the application:
```bash
python app.py
```
*The Gradio app will open at `http://127.0.0.1:7860`.*

## Project Structure

```text
├── app.py                   # Main Gradio application
├── benchmark.py             # Assessment scenario benchmark script
├── test_20_products.py      # Automated 20 random product benchmark
├── config.py                # Configuration and hyper-parameters
├── data/
│   └── generate_dataset.py  # 5MB/5,000 item Bangla dataset generator
└── core/
    ├── pipeline.py          # RAG Orchestrator mapping query to strategy
    ├── indexer.py           # FAISS Hybrid Search implementation
    ├── embeddings.py        # ONNX model wrapper for ~5ms embeddings
    ├── conversation.py      # Entity tracking & coreference resolution
    ├── responder.py         # Template vs LLM response logic
    └── llm.py               # Groq API wrapper
```

## Acknowledgments
Developed as a technical assessment for Speaklar NLP Engineer role.
