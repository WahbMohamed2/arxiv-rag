# generation/generator.py
import json
import requests
from loguru import logger
from config import OLLAMA_HOST, OLLAMA_MODEL
from generation.prompt import build_messages
from retrieval.searcher import hybrid_search
from retrieval.reranker import rerank
from retrieval.cache import get_cached, set_cached
from retrieval.decomposer import decompose_query


def _ollama_chat(messages: list[dict], stream: bool = False) -> str:
    url  = f"{OLLAMA_HOST}/api/chat"
    body = {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   stream,
        "options": {
            "temperature": 0.1,
            "num_ctx":     8192,
        }
    }

    try:
        resp = requests.post(url, json=body, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_HOST}. "
            "Make sure Ollama is running: 'ollama serve'"
        )
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        raise


def answer(
    query:       str,
    top_k:       int = 20,
    rerank_top:  int = 3,
    year_min:    int = None,
    year_max:    int = None,
    category:    str = None,
    arxiv_id:    str = None,    # ← new
) -> dict:
    logger.info(f"Query: {query[:80]}")

    # cache check — include arxiv_id in cache key so per-paper answers don't collide
    cache_key = f"{query}||{arxiv_id}" if arxiv_id else query   # ← new
    cached = get_cached(cache_key)                               # ← new
    if cached:
        logger.info("Cache hit — returning cached answer.")
        return cached

    # decompose into sub-queries
    sub_queries = decompose_query(query)

    # run hybrid search + rerank for each sub-query, then merge
    seen_ids   = set()
    all_chunks = []

    for sub_q in sub_queries:
        candidates = hybrid_search(
            sub_q,
            top_k=top_k,
            year_min=year_min,
            year_max=year_max,
            category=category,
            arxiv_id=arxiv_id,     # ← new
        )
        if not candidates:
            continue

        top = rerank(sub_q, candidates, top_n=rerank_top)

        for chunk in top:
            uid = (chunk["arxiv_id"], chunk.get("chunk_index", chunk["heading"]))
            if uid not in seen_ids:
                seen_ids.add(uid)
                all_chunks.append(chunk)

    if not all_chunks:
        return {
            "answer":  "No relevant papers found for this query.",
            "sources": [],
            "chunks":  [],
        }

    # cap total chunks passed to LLM to avoid context overflow
    all_chunks = all_chunks[:9]

    logger.info(f"Total chunks after decomposition merge: {len(all_chunks)}")

    # build prompt and generate
    messages = build_messages(query, all_chunks)

    logger.info(f"Calling Ollama ({OLLAMA_MODEL})...")
    response = _ollama_chat(messages)
    logger.success("Generation complete.")

    # format sources
    sources = []
    seen    = set()
    for chunk in all_chunks:
        key = (chunk["arxiv_id"], chunk["heading"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "arxiv_id": chunk["arxiv_id"],
                "title":    chunk["title"],
                "section":  chunk["heading"],
                "year":     chunk["year"],
                "authors":  chunk["authors"][:3],
            })

    result = {
        "answer":  response,
        "sources": sources,
        "chunks":  all_chunks,
    }

    set_cached(cache_key, result)   # ← new (was: set_cached(query, result))
    return result