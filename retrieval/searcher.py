# retrieval/searcher.py
import json
from pathlib import Path
from loguru import logger
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from config import QDRANT_HOST, QDRANT_COLLECTION, CHUNKS_DIR
from embedding.embedder import embed_query


# ── Qdrant client ──────────────────────────────────────────────────────────
_client = None

def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_HOST)
    return _client


# ── BM25 index (built once from chunks on disk) ────────────────────────────
_bm25_index  = None
_bm25_chunks = None

def _build_bm25_index():
    global _bm25_index, _bm25_chunks
    if _bm25_index is not None:
        return

    logger.info("Building BM25 index from chunks...")
    all_chunks = []

    for fpath in sorted(CHUNKS_DIR.glob("*.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_chunks.append(json.loads(line))

    if not all_chunks:
        logger.warning("No chunks found for BM25 index.")
        return

    tokenized = [c["text"].lower().split() for c in all_chunks]
    _bm25_index  = BM25Okapi(tokenized)
    _bm25_chunks = all_chunks
    logger.success(f"BM25 index built: {len(all_chunks)} chunks")


# ── Dense search ───────────────────────────────────────────────────────────
def dense_search(
    query: str,
    top_k: int = 20,
    year_min: int = None,
    year_max: int = None,
    category: str = None,
    arxiv_id: str = None,          # ← new
) -> list[dict]:
    client       = _get_client()
    query_vector = embed_query(query)

    conditions = []
    if year_min:
        conditions.append(FieldCondition(key="year", range=Range(gte=year_min)))
    if year_max:
        conditions.append(FieldCondition(key="year", range=Range(lte=year_max)))
    if category:
        conditions.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if arxiv_id:                   # ← new
        conditions.append(FieldCondition(key="arxiv_id", match=MatchValue(value=arxiv_id)))

    search_filter = Filter(must=conditions) if conditions else None

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    ).points

    return [
        {**hit.payload, "_score_dense": hit.score}
        for hit in results
    ]


# ── BM25 search ────────────────────────────────────────────────────────────
def bm25_search(
    query: str,
    top_k: int = 20,
    arxiv_id: str = None,          # ← new
) -> list[dict]:
    _build_bm25_index()
    if _bm25_index is None:
        return []

    tokenized_query = query.lower().split()
    scores          = _bm25_index.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = [
        {**_bm25_chunks[i], "_score_bm25": scores[i]}
        for i in top_indices
        if scores[i] > 0
    ]

    # ← new: filter to specific paper if requested
    if arxiv_id:
        results = [c for c in results if c.get("arxiv_id") == arxiv_id]

    return results


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────
def _reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results:  list[dict],
    k: int = 60,
) -> list[dict]:
    scores = {}
    chunks = {}

    for rank, chunk in enumerate(dense_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunks[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        if cid not in chunks:
            chunks[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {**chunks[cid], "_score_rrf": score}
        for cid, score in ranked
    ]


# ── Hybrid search (main entry point) ──────────────────────────────────────
def hybrid_search(
    query:    str,
    top_k:    int  = 20,
    year_min: int  = None,
    year_max: int  = None,
    category: str  = None,
    arxiv_id: str  = None,         # ← new
) -> list[dict]:
    logger.info(f"Hybrid search: '{query[:60]}'")

    dense = dense_search(query, top_k=top_k, year_min=year_min,
                         year_max=year_max, category=category, arxiv_id=arxiv_id)
    bm25  = bm25_search(query, top_k=top_k, arxiv_id=arxiv_id)  # ← new
    fused = _reciprocal_rank_fusion(dense, bm25)

    logger.info(f"  Dense: {len(dense)} | BM25: {len(bm25)} | Fused: {len(fused)}")
    return fused[:top_k]