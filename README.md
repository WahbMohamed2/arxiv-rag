# AI Papers Assistant

A fully local, no-cloud RAG system that ingests arXiv research papers and lets you ask questions against them with cited, grounded answers. Every component runs on your own hardware — no API keys, no paid services, no data leaving your machine.

Stack: Python, PyMuPDF, Grobid (Docker), BAAI/bge-base-en-v1.5, Qdrant (Docker), rank-bm25, BAAI/bge-reranker-base, Ollama + llama3.1:8b, RAGAS.

---

## The Journey: How Each Block Was Built

This project was built component by component. Each addition solved a concrete problem that the previous state couldn't handle. This section explains what was added, why, and what it changed about the system's capability.

---

### Component 1 — Ingestion and Parsing

The first problem: PDFs are not documents. They are drawing instructions. Academic PDFs are especially hostile — two-column layouts, inline math, figure captions that bleed into body text, and footnotes that break naive text extractors. Running a standard PDF reader over an arXiv paper produces garbage that no downstream model can reason over.

The solution was to use two tools in combination. Grobid runs as a Docker container and returns structured TEI-XML for each paper: titles, authors, abstracts, section headers, and reference lists. Marker handles the actual body text extraction — it is layout-aware, runs on the RTX 4060, and returns clean section-separated markdown rather than a flat dump of characters.

These two outputs are merged into a single JSON record per paper and saved to `data/parsed/`. If either tool fails on a given PDF (roughly 5-10% of papers at scale), the failure is logged to `failures.jsonl` and the pipeline continues. Nothing crashes. Failed papers can be retried independently.

The pipeline itself uses `concurrent.futures.ThreadPoolExecutor` with configurable worker count so downloading and parsing happen in parallel. On the test run, 5 papers were processed in 23 seconds — around 4-5 seconds per paper — which projects to roughly 40 minutes for 500 papers running unattended.

What this component made possible: the system could now ingest arbitrary arXiv papers and produce clean, structured text records. Without this, there is no data.

---

### Component 2 — Chunking

A full paper is 8,000 to 30,000 tokens. No embedding model can take that as a single input, and even if it could, the resulting vector would be so diluted it would be useless for retrieval. The paper needs to be split into pieces that are small enough to embed meaningfully and large enough to contain a complete thought.

The naive approach — split every N characters — was rejected immediately. It produces chunks that start and end in the middle of sentences, strips structural context, and loses the relationship between a chunk and its section. Retrieving a chunk with no section header means the generation layer has no idea whether it came from the introduction, the methodology, or the results.

The chunking implementation splits along section boundaries identified by the parser. Each section is then split into 512-768 token chunks using paragraph boundaries first, falling back to sentence boundaries when a single paragraph exceeds the limit. A 10-15% token overlap is maintained between adjacent chunks to prevent meaning from being lost at the boundary between two chunks.

Every chunk carries its full metadata: paper ID, title, authors, year, month, arXiv category, abstract, section heading, section index, chunk index, total chunk count, and token count. This metadata is stored alongside the vector in Qdrant and is available for filtering at query time. A noise filter removes sections that are structurally present but semantically useless — reference lists, appendices, figure captions detected as headings, acknowledgements.

On the 5-paper test set, 252 chunks were produced in under one second.

What this component made possible: the system could now produce embeddable units that carry enough context to be retrieved accurately and cited precisely. The chunk is the unit of retrieval and the unit of citation — its design determines the ceiling on everything downstream.

---

### Component 3 — Embedding and Indexing

With clean chunks available, the next problem is how to find the right ones given a query. The answer is vector similarity search: embed every chunk as a dense vector, store the vectors in a database, and at query time embed the question and find the nearest chunks by cosine distance.

The embedding model chosen is `BAAI/bge-base-en-v1.5`, a 768-dimension model from the BAAI group. It is available for free on HuggingFace, fits comfortably in the RTX 4060's 8GB VRAM, and performs strongly on semantic similarity tasks involving scientific text. Models are cached to `E:\ai-papers-data\models` so they are not re-downloaded on each run.

Embedding runs in batches of 64 chunks at a time using `torch.no_grad()` to minimise VRAM pressure. The model is loaded once at module level and stays resident across the indexing run.

Qdrant is used as the vector database. It runs as a Docker container with its storage volume mounted to `E:\ai-papers-data\qdrant`, so the index persists across restarts. The collection is created with named vectors of dimension 768, HNSW indexing for fast approximate nearest-neighbour search, and payload indexes on the metadata fields most likely to be used as filters: year, category, and arxiv_id.

What this component made possible: the system could now answer the question "which chunks are semantically similar to this query?" in milliseconds. This is the foundation of retrieval.

---

### Component 4 — Hybrid Retrieval and Re-ranking

Dense vector search has a well-known blind spot: exact terms. If a user asks about "QLoRA" or "RWKV" or a specific author name, a semantic model may return chunks that are thematically related to the general topic but do not contain the exact term. For academic queries, where model names, method acronyms, and author identifiers are load-bearing, this is a serious failure mode.

The solution is hybrid retrieval: run BM25 keyword search and dense vector search in parallel, then merge the results. BM25 is a classical term-frequency ranking algorithm that scores documents by exact term overlap with the query. It handles the specific terminology that dense search misses. Dense search handles the conceptual and paraphrased queries that BM25 misses.

The two result sets are merged using Reciprocal Rank Fusion (RRF). RRF takes the rank position of each chunk in each result list and combines them into a single score without requiring the two ranking systems to be on the same scale. A chunk that ranks highly in both lists scores very well; a chunk that ranks highly in only one still gets credit.

The merged candidate set is then passed through a cross-encoder re-ranker (`BAAI/bge-reranker-base`). Unlike the bi-encoder used for embedding — which encodes query and chunk separately — the cross-encoder reads the query and each chunk together in a single forward pass and produces a precise relevance score. This is computationally expensive, which is why it is applied only to the top 8-12 candidates from the fusion step rather than the full index. The re-ranked top 5 chunks are what get passed to the generation layer.

The re-ranker is the single highest-leverage component in the system for quality per unit of computation. In practice it consistently surfaces the right chunks even when the initial retrieval order was imperfect.

What this component made possible: the system could now handle both semantic queries and exact-term queries, and could distinguish between a chunk that is topically related and one that is actually the right answer. Retrieval quality increased substantially compared to dense search alone.

---

### Component 5 — Generation

With the right chunks in hand, the final step is producing an answer. The LLM receives the retrieved chunks as context and must generate a response that is grounded in that context and cites its sources explicitly.

The generation layer uses Ollama running locally with `llama3.1:8b`. The system prompt instructs the model to answer only from the provided context, to never introduce claims that are not supported by retrieved text, and to format citations using the paper ID and section name of the chunk the claim came from. When structured output is needed — for programmatic consumption — the prompt requests a JSON object with an `answer` field and a `citations` array.

Context packing is handled explicitly: chunks are ordered by re-rank score, each is prefixed with its paper ID and section heading so the model can form accurate references, and if the total token count of the top chunks approaches the context window limit, lower-ranked chunks are dropped rather than truncated mid-sentence.

A CLI wrapper (`cli.py`) was added that accepts an arXiv ID, runs the full single-paper pipeline (download, parse, chunk, embed, index), and then enters an interactive query loop. This makes it possible to add a new paper and immediately interrogate it without running the full batch ingestion.

What this component made possible: the system could now produce natural language answers with source citations from the indexed corpus. The full pipeline from paper URL to cited answer was closed.

---

### Component 6 — Evaluation Harness

A RAG system without evaluation is guesswork. It is impossible to know whether a change to the chunking strategy improved retrieval, or whether a different re-ranker threshold helped or hurt, without a way to measure it.

Evaluation is split across two layers because retrieval and generation can fail independently and need to be diagnosed separately.

Retrieval is measured with Hit Rate at K (was the correct chunk in the top K results?) and Mean Reciprocal Rank (how high in the list did the correct chunk appear?). These metrics require a golden dataset: a set of questions for which the correct source chunks are known in advance. The golden dataset is built first, before any component is tuned, and treated as an immutable test fixture.

Generation is measured using the RAGAS framework across three metrics. Faithfulness measures whether every claim in the generated answer is supported by the retrieved chunks — a faithfulness score below 1.0 indicates hallucination. Answer Relevance measures whether the answer actually addresses the question asked. Context Precision measures whether the retrieved chunks were useful or mostly noise.

Production targets: Hit Rate above 80%, Faithfulness above 0.85, Answer Relevance above 0.80.

The evaluation suite is run every time a component changes. Regression catches go here, not in manual spot-checking.

What this component made possible: the system could be improved systematically. Every change to chunking, retrieval, or generation had a measurable impact. The difference between a component that helps and one that hurts became observable rather than guessed.

---

## End-to-End Flow

```
[Ingest + Parse PDFs]          arxiv API, Grobid, Marker
         |
[Chunk + Attach Metadata]      structure-aware, 512-768 tokens, overlapping
         |
[Embed + Index in Qdrant]      BAAI/bge-base-en-v1.5, HNSW, payload indexes
         |
[Hybrid Retrieval + Re-rank]   BM25 + dense + RRF + bge-reranker-base
         |                         ^ user question enters here
[Generate Answer + Citations]  Ollama llama3.1:8b, grounded, cited
         |
[Return to User]
         |
[Evaluation Harness]           RAGAS + retrieval metrics, run on every change
```

---

## Project Structure

```
ai-papers-assistant/
|
+-- ingestion/
|   +-- downloader.py        arXiv API fetch and PDF download
|   +-- parser.py            Grobid + Marker orchestration
|   +-- pipeline.py          Parallel worker, failure logging, retry
|   +-- storage.py           Parsed record persistence
|
+-- chunking/
|   +-- schema.py            Chunk dataclass with full metadata
|   +-- splitter.py          Structure-aware, overlap-preserving splitting
|
+-- embedding/
|   +-- embedder.py          Batch embedding on GPU
|   +-- indexer.py           Qdrant collection setup and upload
|
+-- retrieval/
|   +-- searcher.py          BM25 + dense hybrid search with RRF
|   +-- reranker.py          Cross-encoder re-ranking
|
+-- generation/
|   +-- prompt.py            System prompt and context packing
|   +-- generator.py         Ollama call and structured output
|
+-- evaluation/
|   +-- golden_dataset.py    Dataset builder and loader
|   +-- harness.py           RAGAS and retrieval metrics runner
|
+-- config.py                All settings, paths, model names
+-- cli.py                   Single-paper ingest and interactive query
+-- requirements.txt
```

---

## Local Stack

Every tool in this project is free and runs locally.

| Layer | Tool |
|---|---|
| PDF download | arxiv Python library |
| PDF parsing — metadata | Grobid via Docker |
| PDF parsing — body text | PyMuPDF / Marker |
| Chunking | custom (tiktoken tokenization) |
| Embedding | BAAI/bge-base-en-v1.5 (HuggingFace) |
| Vector store | Qdrant via Docker |
| Keyword search | rank-bm25 |
| Re-ranking | BAAI/bge-reranker-base (HuggingFace) |
| Generation | Ollama + llama3.1:8b |
| Evaluation | RAGAS |

Data is stored on a dedicated partition (`E:\ai-papers-data`). Model weights are cached to the same drive. Code lives in `C:\CODING\ai-papers-assistant`. Nothing requires an internet connection after the initial model downloads.

---

## Setup

**Prerequisites:** Python 3.11+, Docker, Ollama, CUDA drivers for your GPU.

**1. Start Grobid**
```
docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0
```

**2. Start Qdrant**
```
docker run -p 6333:6333 -v E:\ai-papers-data\qdrant:/qdrant/storage qdrant/qdrant
```

**3. Pull the generation model**
```
ollama pull llama3.1:8b
```

**4. Install dependencies**
```
pip install -r requirements.txt
```

**5. Run the ingestion pipeline**
```
python -m ingestion.pipeline
```

**6. Query a single paper by arXiv ID**
```
python cli.py
```

---

## Evaluation Targets

| Metric | Target |
|---|---|
| Hit Rate @ 5 | > 80% |
| Mean Reciprocal Rank | > 0.70 |
| Faithfulness (RAGAS) | > 0.85 |
| Answer Relevance (RAGAS) | > 0.80 |
| Context Precision (RAGAS) | > 0.75 |
