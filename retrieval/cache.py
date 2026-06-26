# retrieval/cache.py
import json
import numpy as np
import redis
from loguru import logger
from config import REDIS_HOST, REDIS_PORT, CACHE_SIMILARITY_THRESHOLD
from embedding.embedder import embed_texts


r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)


def _embed(query: str) -> np.ndarray:
    return np.array(embed_texts([query])[0])


def get_cached(query: str) -> dict | None:
    vector = _embed(query)
    keys   = r.keys("cache:*")

    for key in keys:
        raw  = r.get(key)
        data = json.loads(raw)
        stored_vector = np.array(data["vector"])
        similarity = np.dot(vector, stored_vector) / (
            np.linalg.norm(vector) * np.linalg.norm(stored_vector)
        )
        if similarity >= CACHE_SIMILARITY_THRESHOLD:
            logger.info(f"Cache hit (similarity={similarity:.3f})")
            return data["result"]

    return None


def set_cached(query: str, result: dict) -> None:
    vector = _embed(query)
    r.set(f"cache:{query}", json.dumps({
        "vector": vector.tolist(),
        "result": result,
    }))
    logger.info(f"Cached query: {query[:60]}")