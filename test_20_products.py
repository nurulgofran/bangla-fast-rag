import json
import random
import time
import os

# Configure threading for local benchmark to match app
import torch
torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from core.pipeline import rag_pipeline

def run_tests():
    # Load products
    with open('data/products.json', 'r', encoding='utf-8') as f:
        products = json.load(f)

    # Get 20 random names
    names = list(set([p['name_bn'].split()[0] for p in products]))
    random.seed(42)
    test_products = random.sample(names, 20)

    print('# 🧪 20-Product Benchmark Report\n')
    print(f'Testing 20 random products for Q2 (follow-up) latency under 100ms.\n')

    rag_pipeline.initialize()

    results_q2_ms = []

    for i, prod in enumerate(test_products, 1):
        q1 = f'আপনাদের কোম্পানি কি {prod} বিক্রি করে?'
        q2 = 'দাম কত টাকা?'
        
        rag_pipeline.reset()
        
        # Q1
        ans1, _, m1 = rag_pipeline.process_query(q1, use_llm=False)
        
        # Q2
        ans2, _, m2 = rag_pipeline.process_query(q2, use_llm=False)
        
        results_q2_ms.append(m2.total_ms)
        
        status = '✅ PASS' if m2.total_ms < 100 else '❌ FAIL'
        print(f'**Test {i:02d}: {prod}**')
        print(f'- Q1 Time: {m1.total_ms:.2f}ms')
        print(f'- Q2 Enriched Query: "{m2.enriched_query}"')
        print(f'- Q2 Coreference: {m2.enrichment_ms:.2f}ms | Embed: {m2.embedding_ms:.2f}ms | Search: {m2.search_ms:.2f}ms')
        print(f'- Q2 Total Time: `{m2.total_ms:.2f}ms` {status}\n')

    avg_q2 = sum(results_q2_ms) / len(results_q2_ms)
    max_q2 = max(results_q2_ms)
    pass_count = sum(1 for t in results_q2_ms if t < 100)

    print('## 📊 Summary')
    print(f'- **Total Tests:** 20')
    print(f'- **Pass Rate (<100ms):** {pass_count}/20 ({pass_count/20*100:.0f}%)')
    print(f'- **Average Q2 Latency:** {avg_q2:.2f}ms')
    print(f'- **Max Q2 Latency:** {max_q2:.2f}ms')

if __name__ == '__main__':
    run_tests()
