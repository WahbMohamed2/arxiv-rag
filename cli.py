# cli.py
import sys
import time
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from config import QDRANT_HOST, QDRANT_COLLECTION
from ingestion.pipeline import process_single_paper
from generation.generator import answer


def _extract_arxiv_id(user_input: str) -> str:
    """Accept bare ID (2301.12345) or full URL."""
    user_input = user_input.strip()
    if "arxiv.org" in user_input:
        user_input = user_input.rstrip("/").split("/")[-1]
    # strip version suffix e.g. v2
    parts = user_input.lower().rsplit("v", 1)
    if len(parts) == 2 and parts[-1].isdigit():
        user_input = parts[0]
    return user_input


def _paper_already_indexed(arxiv_id: str) -> bool:
    client = QdrantClient(url=QDRANT_HOST)
    results = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="arxiv_id", match=MatchValue(value=arxiv_id))
        ]),
        limit=1,
        with_payload=False,
    )
    return len(results[0]) > 0


def _load_paper(arxiv_id: str) -> bool:
    """Index the paper if not already done. Returns True on success."""
    if _paper_already_indexed(arxiv_id):
        print(f"  ✓ Already indexed — skipping download.\n")
        return True

    print("  → Downloading PDF...")
    print("  → Parsing (Grobid + Marker)...")
    print("  → Chunking + embedding + indexing...")

    success = process_single_paper(arxiv_id)

    if not success:
        print(f"\n  ✗ Failed to process {arxiv_id}.")
        return False

    print(f"  ✓ Paper ready.\n")
    return True


def _qa_loop(arxiv_id: str):
    """Interactive Q&A loop scoped to one paper."""
    print("─" * 50)
    print("Ask anything about this paper.")
    print("Commands:  'switch' → load another paper  |  'exit' → quit")
    print("─" * 50 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            sys.exit(0)

        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye.")
            sys.exit(0)
        if question.lower() == "switch":
            return  # bubble up to main loop

        print()
        t = time.time()

        try:
            result = answer(question, arxiv_id=arxiv_id)
        except ConnectionError as e:
            print(f"  ✗ {e}\n")
            continue
        except Exception as e:
            logger.error(f"Generation error: {e}")
            print("  ✗ Something went wrong — see logs.\n")
            continue

        elapsed = time.time() - t
        print(f"Answer ({elapsed:.1f}s):\n")
        print(result["answer"])
        print()

        if result["sources"]:
            print("Sources:")
            for s in result["sources"]:
                print(f"  [{s['arxiv_id']}] {s['title'][:60]}")
                print(f"    Section : {s['section']}")
                print(f"    Authors : {', '.join(s['authors'][:3])}")
                print(f"    Year    : {s['year']}")
            print()


def main():
    print("\n" + "=" * 50)
    print("   AI Papers Assistant")
    print("=" * 50 + "\n")

    while True:
        raw = input("Enter arXiv ID or URL (or 'exit'): ").strip()

        if not raw or raw.lower() == "exit":
            print("Goodbye.")
            sys.exit(0)

        arxiv_id = _extract_arxiv_id(raw)
        print(f"\nPaper: {arxiv_id}\n")

        ok = _load_paper(arxiv_id)
        if ok:
            _qa_loop(arxiv_id)
        # if switch or load failed → loop back to paper prompt


if __name__ == "__main__":
    main()