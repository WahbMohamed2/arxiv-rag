# embedding/embedder.py
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger
from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, DEVICE, MODELS_DIR


# Load model once at module level — stays in memory across calls
_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=DEVICE,
            cache_folder=str(MODELS_DIR),
        )
        logger.success(f"Embedding model loaded on {DEVICE}")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings.
    Returns a list of float vectors, one per input text.
    """
    if not texts:
        return []

    model = _get_model()

    all_embeddings = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i: i + EMBEDDING_BATCH_SIZE]
        with torch.no_grad():
            embeddings = model.encode(
                batch,
                normalize_embeddings=True,   # cosine similarity ready
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        all_embeddings.extend(embeddings.tolist())
        logger.debug(f"Embedded batch {i // EMBEDDING_BATCH_SIZE + 1} "
                     f"({len(batch)} texts)")

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.
    BGE models need a prefix for queries.
    """
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    return embed_texts([prefixed])[0]