# test_decompose.py
import time
from generation.generator import answer

queries = [
    "what is LoRA and how does it compare to full fine-tuning and what are its limitations",
    "what is RLHF",  # single intent, should not decompose
]

for q in queries:
    print(f"\nQuery: {q}")
    print("-" * 60)
    t = time.time()
    result = answer(q)
    print(f"Time: {time.time()-t:.2f}s")
    print(f"Sources: {len(result['sources'])}")
    print(f"Answer: {result['answer']}")