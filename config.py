# config.py
import os
from pathlib import Path

# ── Project root (on C: — code lives here) ────────────────────────────────
BASE_DIR = Path(r"C:\CODING\ai-papers-assistant")

# ── All data lives on E: ───────────────────────────────────────────────────
DATA_DIR   = Path(r"E:\ai-papers-data")
RAW_DIR    = DATA_DIR / "raw"        # downloaded PDFs
PARSED_DIR = DATA_DIR / "parsed"     # Grobid + Marker JSON outputs
CHUNKS_DIR = DATA_DIR / "chunks"     # final chunked records
MODELS_DIR = DATA_DIR / "models"     # HuggingFace models cached here

# ── GPU ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_DEVICE"]         = "cuda"
os.environ["MARKER_DEVICE"]        = "cuda"  # your RTX 4060 — falls back to cpu if CUDA not available

# ── arXiv ──────────────────────────────────────────────────────────────────
ARXIV_CATEGORIES  = ["cs.AI", "cs.LG", "cs.CL"]
ARXIV_MAX_RESULTS = 500   # per category per run, raise later

# ── Grobid (runs as Docker container) ─────────────────────────────────────
GROBID_HOST = "http://localhost:8070"

# ── Marker ────────────────────────────────────────────────────────────────
MARKER_BATCH_SIZE = 4   # tune up if VRAM allows (4060 has 8GB)

# ── Embedding model (free, HuggingFace, fits 4060) ────────────────────────
EMBEDDING_MODEL      = "BAAI/bge-base-en-v1.5"
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_DIM        = 768

# ── Re-ranker model ────────────────────────────────────────────────────────
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_TOP_N = 3   # how many chunks to re-rank, return top 5

# ── Qdrant (runs as Docker container, stores data on E:) ───────────────────
QDRANT_HOST           = "http://localhost:6333"
QDRANT_COLLECTION     = "ai_papers"
QDRANT_STORAGE_PATH   = r"E:\ai-papers-data\qdrant"  # mapped in Docker

# ── Ollama (local LLM, free) ───────────────────────────────────────────────
OLLAMA_HOST  = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE       = 768    # tokens
CHUNK_OVERLAP    = 80     # ~10% overlap
MIN_CHUNK_TOKENS = 80     # discard chunks smaller than this

# ── Pipeline ───────────────────────────────────────────────────────────────
WORKER_COUNT = 4   # parallel download/parse workers

# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
FAILURE_LOG = DATA_DIR / "failures.jsonl"

# ── HuggingFace cache → E: (so models don't fill C:) ──────────────────────
os.environ["HF_HOME"]            = str(MODELS_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)
os.environ["HF_DATASETS_CACHE"]  = str(MODELS_DIR / "datasets")

REDIS_HOST = "localhost"
REDIS_PORT = 6379
CACHE_SIMILARITY_THRESHOLD = 0.90