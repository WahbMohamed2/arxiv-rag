# ingestion/downloader.py
import arxiv
import httpx
from pathlib import Path
from loguru import logger
from config import RAW_DIR, ARXIV_CATEGORIES, ARXIV_MAX_RESULTS


def fetch_papers(category: str, max_results: int = ARXIV_MAX_RESULTS) -> list[dict]:
    """
    Query arXiv for a given category.
    Returns a list of paper metadata dicts.
    """
    logger.info(f"Fetching papers for category: {category}")
    client = arxiv.Client()
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers = []
    for result in client.results(search):
        papers.append({
            "arxiv_id": result.entry_id.split("/")[-1],
            "title":    result.title.strip(),
            "authors":  [a.name for a in result.authors],
            "abstract": result.summary.strip(),
            "year":     result.published.year,
            "month":    result.published.month,
            "category": category,
            "pdf_url":  result.pdf_url,
        })

    logger.info(f"Fetched {len(papers)} papers from {category}")
    return papers


def download_pdf(paper: dict) -> Path | None:
    """
    Download a single PDF to RAW_DIR.
    - Skips if already downloaded
    - Returns local Path on success, None on failure
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{paper['arxiv_id']}.pdf"

    if dest.exists():
        logger.info(f"Already exists, skipping: {paper['arxiv_id']}")
        return dest

    try:
        with httpx.Client(follow_redirects=True, timeout=30) as client:
            resp = client.get(paper["pdf_url"])
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        logger.success(f"Downloaded: {paper['arxiv_id']} → {dest}")
        return dest

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error for {paper['arxiv_id']}: {e.response.status_code}")
        return None
    except httpx.TimeoutException:
        logger.error(f"Timeout downloading {paper['arxiv_id']}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error for {paper['arxiv_id']}: {e}")
        return None


def fetch_and_download(category: str, max_results: int = ARXIV_MAX_RESULTS) -> list[dict]:
    """
    Convenience function: fetch metadata + download PDFs for a category.
    Returns list of paper dicts with added 'local_pdf_path' key.
    """
    papers = fetch_papers(category, max_results)
    results = []

    for paper in papers:
        path = download_pdf(paper)
        results.append({
            **paper,
            "local_pdf_path": str(path) if path else None,
            "download_status": "ok" if path else "failed",
        })

    return results