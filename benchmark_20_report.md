# 🧪 20-Product Benchmark Report

Testing 20 random products for Q2 (follow-up) latency under 100ms.

🔄 Loading embedding model (ONNX): sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2...
  → Using ARM64 INT8 quantized model
✅ Embedding model loaded (ONNX, 384-dim)
🔄 Loading FAISS index...
✅ FAISS index loaded: 5000 vectors
🔄 Warming up embedding model...
✅ Model warmed up
✅ RAG Pipeline initialized
**Test 01: স্কার্ফ**
- Q1 Time: 14.54ms
- Q2 Enriched Query: "স্কার্ফ দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 3.04ms | Search: 8.89ms
- Q2 Total Time: `11.96ms` ✅ PASS

**Test 02: পানির**
- Q1 Time: 17.55ms
- Q2 Enriched Query: "পানির দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 2.51ms | Search: 14.09ms
- Q2 Total Time: `16.63ms` ✅ PASS

**Test 03: জিন্স**
- Q1 Time: 11.68ms
- Q2 Enriched Query: "জিন্স দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 2.48ms | Search: 8.95ms
- Q2 Total Time: `11.45ms` ✅ PASS

**Test 04: সরিষার**
- Q1 Time: 16.17ms
- Q2 Enriched Query: "সরিষার দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 2.26ms | Search: 13.24ms
- Q2 Total Time: `15.52ms` ✅ PASS

**Test 05: প্লায়ার**
- Q1 Time: 15.34ms
- Q2 Enriched Query: "প্লায়ার দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 1.95ms | Search: 12.63ms
- Q2 Total Time: `14.60ms` ✅ PASS

**Test 06: টি-শার্ট**
- Q1 Time: 11.11ms
- Q2 Enriched Query: "টি শার্ট দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 2.20ms | Search: 8.92ms
- Q2 Total Time: `11.14ms` ✅ PASS

**Test 07: রাধুনী**
- Q1 Time: 11.23ms
- Q2 Enriched Query: "রাধুনী দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 1.98ms | Search: 8.51ms
- Q2 Total Time: `10.50ms` ✅ PASS

**Test 08: থালা**
- Q1 Time: 11.82ms
- Q2 Enriched Query: "থালা দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 1.92ms | Search: 9.32ms
- Q2 Total Time: `11.26ms` ✅ PASS

**Test 09: শো-কেস**
- Q1 Time: 11.73ms
- Q2 Enriched Query: "শো কেস দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 2.25ms | Search: 10.22ms
- Q2 Total Time: `12.49ms` ✅ PASS

**Test 10: স্কয়ার**
- Q1 Time: 16.15ms
- Q2 Enriched Query: "স্কয়ার দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 2.05ms | Search: 13.69ms
- Q2 Total Time: `15.76ms` ✅ PASS

**Test 11: হ্যামার**
- Q1 Time: 16.23ms
- Q2 Enriched Query: "হ্যামার দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 1.93ms | Search: 13.83ms
- Q2 Total Time: `15.78ms` ✅ PASS

**Test 12: ল্যাপটপ**
- Q1 Time: 11.99ms
- Q2 Enriched Query: "ল্যাপটপ দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 2.31ms | Search: 9.18ms
- Q2 Total Time: `11.50ms` ✅ PASS

**Test 13: ড্রেসিং**
- Q1 Time: 11.43ms
- Q2 Enriched Query: "ড্রেসিং দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 2.17ms | Search: 8.47ms
- Q2 Total Time: `10.66ms` ✅ PASS

**Test 14: স্প্রিং**
- Q1 Time: 11.23ms
- Q2 Enriched Query: "স্প্রিং দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 2.13ms | Search: 8.46ms
- Q2 Total Time: `10.61ms` ✅ PASS

**Test 15: ইয়ারফোন**
- Q1 Time: 10.99ms
- Q2 Enriched Query: "ইয়ারফোন দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 1.92ms | Search: 8.62ms
- Q2 Total Time: `10.56ms` ✅ PASS

**Test 16: প্রাণ**
- Q1 Time: 11.23ms
- Q2 Enriched Query: "প্রাণ দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 2.28ms | Search: 8.69ms
- Q2 Total Time: `10.98ms` ✅ PASS

**Test 17: চার্জার**
- Q1 Time: 15.53ms
- Q2 Enriched Query: "চার্জার দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 2.05ms | Search: 12.69ms
- Q2 Total Time: `14.75ms` ✅ PASS

**Test 18: শ্যাম্পু**
- Q1 Time: 11.20ms
- Q2 Enriched Query: "শ্যাম্পু দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 2.06ms | Search: 8.34ms
- Q2 Total Time: `10.42ms` ✅ PASS

**Test 19: ঝুড়ি**
- Q1 Time: 11.70ms
- Q2 Enriched Query: "ঝুড়ি দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 2.02ms | Search: 8.85ms
- Q2 Total Time: `10.90ms` ✅ PASS

**Test 20: সেলফি**
- Q1 Time: 12.59ms
- Q2 Enriched Query: "সেলফি দাম কত টাকা?"
- Q2 Coreference: 0.00ms | Embed: 2.25ms | Search: 8.86ms
- Q2 Total Time: `11.13ms` ✅ PASS

## 📊 Summary
- **Total Tests:** 20
- **Pass Rate (<100ms):** 20/20 (100%)
- **Average Q2 Latency:** 12.43ms
- **Max Q2 Latency:** 16.63ms
