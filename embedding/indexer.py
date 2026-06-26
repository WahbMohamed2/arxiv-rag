# embedding/indexer.py
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    PayloadSchemaType,
)

from config import (
    QDRANT_HOST,
    QDRANT_COLLECTION,
    CHUNKS_DIR,
    EMBEDDING_DIM,
)
from embedding.embedder import embed_texts


# ── Qdrant client ──────────────────────────────────────────────────────────
_client = None

def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_HOST)
        logger.info(f"Connected to Qdrant at {QDRANT_HOST}")
    return _client


def setup_collection(recreate: bool = False):
    """
    Create the Qdrant collection if it doesn't exist.
    Set recreate=True to wipe and start fresh.
    """
    client     = _get_client()
    collection = QDRANT_COLLECTION

    existing = [c.name for c in client.get_collections().collections]

    if collection in existing:
        if recreate:
            client.delete_collection(collection)
            logger.warning(f"Deleted existing collection: {collection}")
        else:
            logger.info(f"Collection already exists: {collection}")
            return

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )

    # Index payload fields for fast filtering
    for field, schema in [
        ("arxiv_id",  PayloadSchemaType.KEYWORD),
        ("year",      PayloadSchemaType.INTEGER),
        ("category",  PayloadSchemaType.KEYWORD),
        ("heading",   PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=schema,
        )

    logger.success(f"Created collection: {collection} (dim={EMBEDDING_DIM})")


def index_all_chunks(batch_size: int = 64):
    """
    Read all JSONL chunk files from CHUNKS_DIR,
    embed them, upload to Qdrant.
    Skips papers already indexed (checks by arxiv_id).
    """
    client = _get_client()
    setup_collection()

    chunk_files = list(CHUNKS_DIR.glob("*.jsonl"))
    if not chunk_files:
        logger.warning("No chunk files found. Run chunking first.")
        return

    logger.info(f"Indexing {len(chunk_files)} papers into Qdrant...")
    total_indexed = 0

    for fpath in tqdm(chunk_files, desc="Indexing papers"):
        arxiv_id = fpath.stem

        # Check if already indexed
        results = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter={
                "must": [{"key": "arxiv_id", "match": {"value": arxiv_id}}]
            },
            limit=1,
        )
        if results[0]:
            logger.debug(f"Already indexed: {arxiv_id}")
            total_indexed += 1
            continue

        # Load chunks
        chunks = []
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        if not chunks:
            continue

        # Embed in batches
        texts      = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        # Build Qdrant points
        points = []
        for chunk, vector in zip(chunks, embeddings):
            point_id = abs(hash(chunk["chunk_id"])) % (2**63)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=chunk,
                )
            )

        # Upload in batches
        for i in range(0, len(points), batch_size):
            batch = points[i: i + batch_size]
            client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=batch,
            )

        total_indexed += 1
        logger.success(f"Indexed {arxiv_id} → {len(chunks)} vectors")

    logger.info(f"Done. Total papers indexed: {total_indexed}")


def index_chunks(chunks: list[dict], vectors: list[list[float]], batch_size: int = 64):
    """
    Upload pre-computed chunks + vectors to Qdrant.
    Used by process_single_paper() for on-demand indexing.
    """
    client = _get_client()
    setup_collection()

    points = []
    for chunk, vector in zip(chunks, vectors):
        point_id = abs(hash(chunk["chunk_id"])) % (2**63)
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=chunk,
            )
        )

    for i in range(0, len(points), batch_size):
        batch = points[i: i + batch_size]
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=batch,
        )

    logger.success(f"Indexed {len(chunks)} chunks into Qdrant.")


def get_collection_stats() -> dict:
    """Return basic stats about the current collection."""
    client = _get_client()
    info   = client.get_collection(QDRANT_COLLECTION)
    return {
        "total_vectors": info.points_count,
        "collection":    QDRANT_COLLECTION,
        "dimension":     EMBEDDING_DIM,
    }