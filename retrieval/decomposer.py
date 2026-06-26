# retrieval/decomposer.py
import json
import requests
from loguru import logger
from config import OLLAMA_HOST, OLLAMA_MODEL


def decompose_query(query: str) -> list[str]:
    prompt = f"""Split the following question into a list of simple, independent sub-questions.
If the question is already simple and has only one intent, return a list with just that question.
Return ONLY a JSON array of strings, nothing else. No explanation, no markdown.

Question: {query}

Example output for a multi-intent question:
["what is LoRA", "how does LoRA compare to full fine-tuning", "what are the limitations of LoRA"]

Example output for a single-intent question:
["what is LoRA"]

Output:"""

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 2048},
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        sub_queries = json.loads(raw)

        if not isinstance(sub_queries, list) or len(sub_queries) == 0:
            return [query]

        sub_queries = [q.strip() for q in sub_queries if isinstance(q, str) and q.strip()]
        logger.info(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
        return sub_queries

    except Exception as e:
        logger.warning(f"Decomposition failed ({e}), using original query")
        return [query]