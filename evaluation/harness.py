# evaluation/harness.py
import json
import random
from datetime import datetime
from pathlib import Path
from loguru import logger
from config import DATA_DIR, OLLAMA_HOST, OLLAMA_MODEL
from evaluation.golden_dataset import load_golden_dataset
from retrieval.searcher import hybrid_search
from retrieval.reranker import rerank
from generation.generator import answer


EVAL_RESULTS_DIR = DATA_DIR / "eval_results"


# ── Retrieval Evaluation ───────────────────────────────────────────────────

def _hit_rate(retrieved_ids: list[str], expected_id: str) -> bool:
    return expected_id in retrieved_ids


def _reciprocal_rank(retrieved_ids: list[str], expected_id: str) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid == expected_id:
            return 1.0 / (i + 1)
    return 0.0


def run_retrieval_eval(top_k: int = 10) -> dict:
    dataset = load_golden_dataset()
    logger.info(f"Running retrieval eval on {len(dataset)} questions...")

    hit_rates  = []
    mrr_scores = []
    results    = []

    for item in dataset:
        question    = item["question"]
        expected_id = item["expected_arxiv_id"]

        candidates    = hybrid_search(question, top_k=top_k)
        reranked      = rerank(question, candidates, top_n=top_k)
        retrieved_ids = [c["arxiv_id"] for c in reranked]

        hit = _hit_rate(retrieved_ids, expected_id)
        mrr = _reciprocal_rank(retrieved_ids, expected_id)

        hit_rates.append(float(hit))
        mrr_scores.append(mrr)

        results.append({
            "question":    question,
            "expected_id": expected_id,
            "hit":         hit,
            "mrr":         mrr,
            "top_ids":     retrieved_ids[:5],
        })

        status = "HIT" if hit else "MISS"
        logger.info(f"  [{status}] MRR={mrr:.3f} | {question[:50]}")

    summary = {
        "timestamp":     datetime.now().isoformat(),
        "eval_type":     "retrieval",
        "num_questions": len(dataset),
        "top_k":         top_k,
        "hit_rate":      sum(hit_rates) / len(hit_rates),
        "mrr":           sum(mrr_scores) / len(mrr_scores),
        "details":       results,
    }

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_RESULTS_DIR / f"retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.success(f"Retrieval eval complete:")
    logger.success(f"  Hit Rate @ {top_k} : {summary['hit_rate']:.3f}")
    logger.success(f"  MRR              : {summary['mrr']:.3f}")
    logger.success(f"  Saved            : {out_path}")
    return summary


# ── RAGAS Evaluation ───────────────────────────────────────────────────────

def run_ragas_eval(num_questions: int = 10) -> dict:
    import requests as req

    def _ollama_judge(prompt: str) -> str:
        try:
            resp = req.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_ctx": 4096},
                },
                timeout=120,
            )
            return resp.json()["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"Judge call failed: {e}")
            return ""

    dataset = load_golden_dataset()
    subset  = random.sample(dataset, min(num_questions, len(dataset)))
    logger.info(f"Running RAGAS eval on {len(subset)} questions...")

    faithfulness_scores = []
    relevancy_scores    = []
    precision_scores    = []

    for i, item in enumerate(subset):
        question = item["question"]
        logger.info(f"  [{i+1}/{len(subset)}] {question[:60]}")

        try:
            result   = answer(question, top_k=20, rerank_top=3)  # was 5
            ans_text = result["answer"]
            contexts = [c["text"] for c in result["chunks"]]
            ctx      = "\n\n".join(contexts[:3])

            prompt = f"""Evaluate this RAG system output. Answer with ONLY three numbers on separate lines.

CONTEXT:
{ctx[:1500]}

QUESTION: {question}

ANSWER: {ans_text[:800]}

Rate each from 0.0 to 1.0:
Line 1 - Faithfulness: are all answer claims supported by the context?
Line 2 - Answer Relevancy: does the answer address the question?
Line 3 - Context Precision: are the retrieved chunks relevant to the question?

Output format (three numbers only, nothing else):
0.X
0.X
0.X"""

            result_text = _ollama_judge(prompt)
            lines       = [l.strip() for l in result_text.strip().split("\n") if l.strip()]
            scores_raw  = []
            for l in lines:
                try:
                    scores_raw.append(float(l))
                except:
                    continue

            f = scores_raw[0] if len(scores_raw) > 0 else 0.5
            r = scores_raw[1] if len(scores_raw) > 1 else 0.5
            p = scores_raw[2] if len(scores_raw) > 2 else 0.5

            faithfulness_scores.append(f)
            relevancy_scores.append(r)
            precision_scores.append(p)

            logger.info(f"    F={f:.2f} R={r:.2f} P={p:.2f}")

        except Exception as e:
            logger.warning(f"  Skipping due to error: {e}")
            continue

    if not faithfulness_scores:
        logger.error("No questions scored successfully.")
        return {}

    summary = {
        "timestamp":         datetime.now().isoformat(),
        "eval_type":         "manual_ragas",
        "num_questions":     len(faithfulness_scores),
        "faithfulness":      sum(faithfulness_scores) / len(faithfulness_scores),
        "answer_relevancy":  sum(relevancy_scores) / len(relevancy_scores),
        "context_precision": sum(precision_scores) / len(precision_scores),
    }

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_RESULTS_DIR / f"ragas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.success(f"Eval complete:")
    logger.success(f"  Faithfulness      : {summary['faithfulness']:.3f}")
    logger.success(f"  Answer Relevancy  : {summary['answer_relevancy']:.3f}")
    logger.success(f"  Context Precision : {summary['context_precision']:.3f}")
    logger.success(f"  Saved             : {out_path}")
    return summary


if __name__ == "__main__":
    run_ragas_eval(num_questions=10)