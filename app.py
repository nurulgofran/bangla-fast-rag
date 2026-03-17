"""
Gradio Chat UI for the Speaklar Bangla RAG System.
Features: multi-turn chat, real-time latency display, benchmark mode.
"""
# ─── CRITICAL: Fix PyTorch threading for Gradio ───
# PyTorch multi-threaded inference is SLOW in Gradio's thread pool on Apple Silicon.
# Single-threaded = ~10ms, multi-threaded in Gradio = ~400ms.
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress HuggingFace tokenizer warning
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gradio as gr
from core.pipeline import rag_pipeline


def initialize():
    """Initialize the RAG pipeline at startup."""
    rag_pipeline.initialize()


def chat(message: str, history: list[dict]) -> str:
    """Process a chat message and return response with metrics."""
    response, results, metrics = rag_pipeline.process_query(
        query=message,
        use_llm=True,
    )

    latency_info = (
        f"\n\n---\n"
        f"📊 **পারফরম্যান্স মেট্রিক্স:**\n"
        f"- কোয়েরি এনরিচমেন্ট: `{metrics.enrichment_ms:.2f}ms`"
    )
    if metrics.was_enriched:
        latency_info += f" ✅ (এনরিচড: \"{metrics.enriched_query}\")"
    latency_info += (
        f"\n- এমবেডিং: `{metrics.embedding_ms:.2f}ms`"
        f"\n- সার্চ: `{metrics.search_ms:.2f}ms`"
        f"\n- রেসপন্স: `{metrics.response_ms:.2f}ms` [{metrics.response_type}]"
        f"\n- **মোট সময়: `{metrics.total_ms:.2f}ms`**"
    )

    if metrics.total_ms < 100:
        latency_info += " ⚡ <100ms ✅"
    else:
        latency_info += " ⚠️ >100ms"

    return response + latency_info


def run_benchmark() -> str:
    """Run the exact job posting test scenario."""
    rag_pipeline.reset()

    output = "## 🧪 বেঞ্চমার্ক: Speaklar অ্যাসেসমেন্ট টেস্ট সিনারিও\n\n"

    q1 = "আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?"
    output += f"**Q1:** {q1}\n\n"
    response1, results1, metrics1 = rag_pipeline.process_query(q1, use_llm=False)
    output += f"**A1:** {response1}\n"
    output += f"⏱️ Q1 Total: `{metrics1.total_ms:.2f}ms`\n\n---\n\n"

    q2 = "দাম কত টাকা?"
    output += f"**Q2:** {q2}\n\n"
    response2, results2, metrics2 = rag_pipeline.process_query(q2, use_llm=False)
    output += f"**A2:** {response2}\n\n"
    output += f"### Q2 বিস্তারিত মেট্রিক্স:\n"
    output += f"- এনরিচড কোয়েরি: \"{metrics2.enriched_query}\"\n"
    output += f"- এনরিচমেন্ট: `{metrics2.enrichment_ms:.2f}ms`\n"
    output += f"- এমবেডিং: `{metrics2.embedding_ms:.2f}ms`\n"
    output += f"- সার্চ: `{metrics2.search_ms:.2f}ms`\n"
    output += f"- রেসপন্স: `{metrics2.response_ms:.2f}ms` [{metrics2.response_type}]\n"
    output += f"- **মোট: `{metrics2.total_ms:.2f}ms`**\n\n"

    if metrics2.total_ms < 100:
        output += "### ✅ পাস! Q2 ১০০ms এর নিচে সম্পন্ন হয়েছে!"
    else:
        output += "### ❌ ব্যর্থ! Q2 ১০০ms অতিক্রম করেছে।"

    rag_pipeline.reset()
    return output


# ─── Build Gradio App ───

def create_app() -> gr.Blocks:
    with gr.Blocks(title="Speaklar Bangla RAG System") as app:
        gr.Markdown(
            "# 🇧🇩 Speaklar Bangla RAG System\n"
            "Context-aware Bangla RAG with coreference resolution | <100ms retrieval + response"
        )

        with gr.Tab("💬 চ্যাট"):
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(
                placeholder="বাংলায় পণ্য সম্পর্কে জিজ্ঞাসা করুন...",
                label="আপনার প্রশ্ন",
            )
            with gr.Row():
                send_btn = gr.Button("পাঠান", variant="primary")
                reset_btn = gr.Button("রিসেট")

            gr.Examples(
                examples=[
                    "আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?",
                    "ইলেকট্রনিক্স পণ্য আছে?",
                    "স্মার্টফোন দেখান",
                ],
                inputs=msg,
            )

            def respond(message, chat_history):
                response = chat(message, chat_history)
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": response})
                return "", chat_history

            def do_reset(chat_history):
                rag_pipeline.reset()
                return [], "রিসেট সম্পন্ন"

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            send_btn.click(respond, [msg, chatbot], [msg, chatbot])
            reset_btn.click(do_reset, [chatbot], [chatbot, msg])

        with gr.Tab("🧪 বেঞ্চমার্ক"):
            gr.Markdown("Speaklar অ্যাসেসমেন্ট টেস্ট সিনারিও চালান (Q1 → Q2)")
            benchmark_btn = gr.Button("▶️ বেঞ্চমার্ক চালান", variant="primary")
            benchmark_output = gr.Markdown()
            benchmark_btn.click(fn=run_benchmark, outputs=benchmark_output)

        with gr.Tab("ℹ️ তথ্য"):
            gr.Markdown(
                "## আর্কিটেকচার\n"
                "1. **Query Enrichment** — কথোপকথনের প্রসঙ্গ থেকে entity tracking\n"
                "2. **Embedding** — paraphrase-multilingual-MiniLM-L12-v2\n"
                "3. **Hybrid Search** — Keyword + FAISS IndexFlatIP\n"
                "4. **Dual Response** — Template (<1ms) + LLM (Groq)\n\n"
                "## পারফরম্যান্স\n"
                "- Q2 (follow-up) hot path: **~17ms median**\n"
                "- 100ms এর নিচে ✅"
            )

    return app


if __name__ == "__main__":
    initialize()
    app = create_app()
    app.launch(server_port=7860, share=False)
