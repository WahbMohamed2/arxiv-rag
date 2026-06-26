# evaluation/golden_dataset.py
import json
import random
import requests
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from config import CHUNKS_DIR, DATA_DIR, OLLAMA_HOST, OLLAMA_MODEL


GOLDEN_DATASET_PATH = DATA_DIR / "golden_dataset.jsonl"


def _generate_questions_for_chunk(chunk: dict) -> list[str]:
    prompt = f"""You are a research assistant. Read the following excerpt from a research paper and generate exactly 2 specific, answerable questions that a researcher might ask about it.

Paper: {chunk['title']}
Section: {chunk['heading']}
Text: {chunk['text'][:800]}

Rules:
- Questions must be answerable from the text above
- Questions must be specific, not generic
- Do NOT include numbering or bullet points
- Output ONLY the 2 questions, one per line, nothing else

Questions:"""

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.7, "num_ctx": 2048},
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"].strip()

        questions = [
            q.strip()
            for q in content.split("\n")
            if q.strip() and len(q.strip()) > 15 and "?" in q
        ]
        return questions[:2]

    except Exception as e:
        logger.warning(f"Ollama call failed: {e}")
        return []


def build_golden_dataset(
    target: int = 300,
    chunks_per_paper: int = 1,
) -> list[dict]:
    chunk_files = list(CHUNKS_DIR.glob("*.jsonl"))
    if not chunk_files:
        logger.error("No chunks found. Run ingestion + chunking first.")
        return []

    logger.info(f"Found {len(chunk_files)} papers to sample from")

    papers_needed = target // (chunks_per_paper * 2)
    sampled_files = random.sample(
        chunk_files,
        min(papers_needed, len(chunk_files))
    )

    all_entries = []

    for fpath in tqdm(sampled_files, desc="Generating questions"):
        chunks = []
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        if not chunks:
            continue

        preferred = [
            c for c in chunks
            if any(kw in c["heading"].lower() for kw in
                   ["introduction", "method", "result", "conclusion", "approach"])
        ]
        pool     = preferred if preferred else chunks
        selected = random.sample(pool, min(chunks_per_paper, len(pool)))

        for chunk in selected:
            questions = _generate_questions_for_chunk(chunk)
            for q in questions:
                entry = {
                    "question":          q,
                    "expected_arxiv_id": chunk["arxiv_id"],
                    "expected_section":  chunk["heading"],
                    "title":             chunk["title"],
                    "year":              chunk["year"],
                }
                all_entries.append(entry)

        if len(all_entries) >= target:
            break

    random.shuffle(all_entries)

    GOLDEN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_DATASET_PATH, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.success(f"Golden dataset saved: {len(all_entries)} questions -> {GOLDEN_DATASET_PATH}")
    return all_entries


def load_golden_dataset() -> list[dict]:
    if not GOLDEN_DATASET_PATH.exists():
        logger.warning("Golden dataset not found. Building it now...")
        return build_golden_dataset()

    items = []
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    logger.info(f"Loaded golden dataset: {len(items)} questions")
    return items