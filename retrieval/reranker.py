# retrieval/reranker.py
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import RERANKER_MODEL, RERANKER_TOP_N, DEVICE, MODELS_DIR


_tokenizer = None
_model     = None


def _get_reranker():
    global _tokenizer, _model
    if _model is None:
        logger.info(f"Loading re-ranker: {RERANKER_MODEL}")
        _tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_MODEL,
            cache_dir=str(MODELS_DIR),
        )
        _model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL,
            cache_dir=str(MODELS_DIR),
        ).to(DEVICE)
        _model.eval()
        logger.success(f"Re-ranker loaded on {DEVICE}")
    return _tokenizer, _model


def rerank(query: str, chunks: list[dict], top_n: int = RERANKER_TOP_N) -> list[dict]:
    """
    Re-rank chunks using a cross-encoder.
    Takes query + each chunk text, scores relevance jointly.
    Returns top_n chunks sorted by re-ranker score.
    """
    if not chunks:
        return []

    tokenizer, model = _get_reranker()

    pairs = [[query, chunk["text"]] for chunk in chunks]

    with torch.no_grad():
        encoded = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)

        scores = model(**encoded).logits.squeeze(-1).tolist()

    # Handle single result (squeeze turns it into float)
    if isinstance(scores, float):
        scores = [scores]

    # Attach scores and sort
    for chunk, score in zip(chunks, scores):
        chunk["_score_reranker"] = score

    reranked = sorted(chunks, key=lambda x: x["_score_reranker"], reverse=True)

    logger.info(f"Re-ranked {len(chunks)} → returning top {top_n}")
    return reranked[:top_n]