# test_downloader.py
if __name__ == "__main__":
    from evaluation.harness import run_retrieval_eval, run_ragas_eval

    # Step 1 — Retrieval eval
    print("="*50)
    print("STEP 1: RETRIEVAL EVALUATION")
    print("="*50)
    retrieval = run_retrieval_eval(top_k=10)
    print(f"\n  Hit Rate @ 10 : {retrieval['hit_rate']:.3f}")
    print(f"  MRR           : {retrieval['mrr']:.3f}")

    # Step 2 — RAGAS eval (50 questions)
    print("\n" + "="*50)
    print("STEP 2: RAGAS GENERATION EVALUATION")
    print("="*50)
    ragas = run_ragas_eval(num_questions=50)
    if ragas:
        print(f"\n  Faithfulness      : {ragas['faithfulness']:.3f}")
        print(f"  Answer Relevancy  : {ragas['answer_relevancy']:.3f}")
        print(f"  Context Precision : {ragas['context_precision']:.3f}")