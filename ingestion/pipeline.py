# ingestion/pipeline.py
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from config import (
    WORKER_COUNT,
    ARXIV_CATEGORIES,
    ARXIV_MAX_RESULTS,
    FAILURE_LOG,
    DATA_DIR,
)
from ingestion.downloader import fetch_papers, download_pdf
from ingestion.parser import parse_paper


def _process_one(paper: dict) -> dict:
    """
    Single unit of work: download → parse → return status.
    This is what each worker thread runs.
    """
    arxiv_id = paper["arxiv_id"]

    # Step 1 — Download
    pdf_path = download_pdf(paper)
    if pdf_path is None:
        return {**paper, "pipeline_status": "download_failed"}

    # Step 2 — Parse
    record = parse_paper(pdf_path, paper)
    if record is None:
        return {**paper, "pipeline_status": "parse_failed"}

    return {**paper, "pipeline_status": "done"}


def _log_failure(result: dict):
    """Append failed paper to failures.jsonl for later retry."""
    FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILURE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def run_pipeline(
    categories: list[str] = ARXIV_CATEGORIES,
    max_results: int       = ARXIV_MAX_RESULTS,
    max_workers: int       = WORKER_COUNT,
):
    """
    Full ingestion pipeline:
      1. Fetch paper metadata from arXiv for each category
      2. Download PDFs in parallel
      3. Parse each PDF (Grobid + PyMuPDF)
      4. Log failures to failures.jsonl
      5. Print summary
    """
    logger.info("=" * 60)
    logger.info("Starting ingestion pipeline")
    logger.info(f"Categories : {categories}")
    logger.info(f"Max results: {max_results} per category")
    logger.info(f"Workers    : {max_workers}")
    logger.info("=" * 60)

    # ── Step 1: Fetch all metadata ─────────────────────────────
    all_papers = []
    for cat in categories:
        papers = fetch_papers(cat, max_results)
        all_papers.extend(papers)

    # Deduplicate by arxiv_id (a paper can appear in multiple categories)
    seen = set()
    unique_papers = []
    for p in all_papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            unique_papers.append(p)

    logger.info(f"Total unique papers to process: {len(unique_papers)}")

    # ── Step 2 & 3: Download + Parse in parallel ───────────────
    done_count    = 0
    failed_count  = 0
    skipped_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, p): p for p in unique_papers}

        with tqdm(total=len(unique_papers), desc="Processing papers") as pbar:
            for future in as_completed(futures):
                result = future.result()
                status = result["pipeline_status"]

                if status == "done":
                    done_count += 1
                elif status in ("download_failed", "parse_failed"):
                    failed_count += 1
                    _log_failure(result)
                    logger.warning(f"Failed [{status}]: {result['arxiv_id']}")

                pbar.update(1)
                pbar.set_postfix({
                    "done":   done_count,
                    "failed": failed_count,
                })

    # ── Step 4: Summary ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Pipeline complete")
    logger.info(f"  ✅ Done   : {done_count}")
    logger.info(f"  ❌ Failed : {failed_count}")
    logger.info(f"  📁 Parsed : {DATA_DIR / 'parsed'}")
    if failed_count > 0:
        logger.info(f"  📋 Failures logged to: {FAILURE_LOG}")
    logger.info("=" * 60)


def retry_failures():
    """
    Re-run the pipeline on previously failed papers.
    Reads from failures.jsonl, retries each one.
    """
    if not FAILURE_LOG.exists():
        logger.info("No failures log found. Nothing to retry.")
        return

    failed_papers = []
    with open(FAILURE_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                failed_papers.append(json.loads(line))

    if not failed_papers:
        logger.info("Failures log is empty. Nothing to retry.")
        return

    logger.info(f"Retrying {len(failed_papers)} failed papers...")

    # Clear the log before retry — it'll be rewritten for any that still fail
    FAILURE_LOG.unlink()

    with ThreadPoolExecutor(max_workers=WORKER_COUNT) as pool:
        futures = {pool.submit(_process_one, p): p for p in failed_papers}
        for future in as_completed(futures):
            result = future.result()
            if result["pipeline_status"] != "done":
                _log_failure(result)
                logger.warning(f"Still failing: {result['arxiv_id']}")
            else:
                logger.success(f"Recovered: {result['arxiv_id']}")



def process_single_paper(arxiv_id: str) -> bool:
    """
    Download, parse, chunk, embed, and index a single paper by arXiv ID.
    Returns True if successful.
    """
    from chunking.splitter import chunk_parsed_paper
    from embedding.embedder import embed_texts
    from embedding.indexer import index_chunks

    # Construct minimal paper metadata
    paper = {
        "arxiv_id": arxiv_id,
        "pdf_url":  f"https://arxiv.org/pdf/{arxiv_id}",
        "title":    arxiv_id,
        "authors":  [],
        "abstract": "",
        "year":     2025,
        "category": "cs.AI",
    }

    # Try to enrich metadata from arXiv API
    try:
        from ingestion.downloader import fetch_papers
        results = fetch_papers("cs.AI", max_results=1)
        match = next((p for p in results if p["arxiv_id"] == arxiv_id), None)
        if match:
            paper = match
    except Exception as e:
        logger.warning(f"Could not fetch metadata for {arxiv_id}: {e} — using defaults.")

    # Download + parse
    result = _process_one(paper)
    if result["pipeline_status"] != "done":
        logger.error(f"Ingestion failed for {arxiv_id}: {result['pipeline_status']}")
        return False

    # Chunk
    parsed_path = DATA_DIR / "parsed" / f"{arxiv_id}.json"
    if not parsed_path.exists():
        logger.error(f"Parsed file not found: {parsed_path}")
        return False

    chunks = chunk_parsed_paper(parsed_path)
    if not chunks:
        logger.error(f"No chunks produced for {arxiv_id}")
        return False

    # Embed + index
    texts   = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    index_chunks(chunks, vectors)

    logger.success(f"Paper {arxiv_id} fully processed and indexed.")
    return True

if __name__ == "__main__":
    run_pipeline()