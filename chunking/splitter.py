# chunking/splitter.py
import re
import json
import tiktoken
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from config import PARSED_DIR, CHUNKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_TOKENS
from chunking.schema import Chunk


# Tokenizer — same one used by most embedding models
_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text, disallowed_special=()))


def _split_text_into_chunks(text: str, heading: str) -> list[str]:
    """
    Split a section's text into token-sized chunks with overlap.
    Splits at paragraph boundaries where possible.
    """
    # Split into paragraphs first
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    chunks      = []
    current     = []
    current_tok = 0

    for para in paragraphs:
        para_tok = _count_tokens(para)

        # Single paragraph too big — split by sentence
        if para_tok > CHUNK_SIZE:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_tok = _count_tokens(sent)
                if current_tok + sent_tok > CHUNK_SIZE and current:
                    chunks.append(" ".join(current))
                    # Overlap: keep last few sentences
                    overlap_sents = []
                    overlap_tok   = 0
                    for s in reversed(current):
                        t = _count_tokens(s)
                        if overlap_tok + t > CHUNK_OVERLAP:
                            break
                        overlap_sents.insert(0, s)
                        overlap_tok += t
                    current     = overlap_sents
                    current_tok = overlap_tok
                current.append(sent)
                current_tok += sent_tok
            continue

        # Normal paragraph — add to current chunk
        if current_tok + para_tok > CHUNK_SIZE and current:
            chunks.append("\n\n".join(current))
            # Overlap: keep last paragraph if it fits
            if para_tok <= CHUNK_OVERLAP:
                current     = [current[-1]]
                current_tok = _count_tokens(current[-1])
            else:
                current     = []
                current_tok = 0

        current.append(para)
        current_tok += para_tok

    # Don't forget the last chunk
    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _is_noise(heading: str) -> bool:
    """
    Filter out sections that are noise — references, tables rows,
    figure captions mistakenly detected as headings.
    """
    heading_lower = heading.lower().strip()
    noise_patterns = [
        r'^\d+\.\d+\s+\d',        # looks like a table row e.g. "59.83 Missing"
        r'^references$',
        r'^acknowledgements?$',
        r'^appendix',
        r'^table\s+\d',
        r'^figure\s+\d',
        r'^fig\.\s+\d',
    ]
    for pattern in noise_patterns:
        if re.match(pattern, heading_lower):
            return True
    return False


def chunk_paper(record: dict) -> list[Chunk]:
    """
    Take a parsed paper record and produce a flat list of Chunk objects.
    Each chunk is one embeddable unit with full metadata attached.
    """
    arxiv_id = record["arxiv_id"]
    sections = record.get("sections", [])

    if not sections:
        logger.warning(f"No sections found for {arxiv_id}, skipping.")
        return []

    all_chunks  = []
    section_idx = 0

    for section in sections:
        heading = section.get("heading", "body")
        text    = section.get("text", "").strip()

        # Skip noise sections
        if _is_noise(heading):
            continue

        if not text or _count_tokens(text) < MIN_CHUNK_TOKENS:
            continue

        # Split section into chunks
        raw_chunks = _split_text_into_chunks(text, heading)

        for chunk_idx, chunk_text in enumerate(raw_chunks):
            tok = _count_tokens(chunk_text)
            if tok < MIN_CHUNK_TOKENS:
                continue

            chunk = Chunk(
                chunk_id      = f"{arxiv_id}_{section_idx}_{chunk_idx}",
                arxiv_id      = arxiv_id,
                text          = chunk_text,
                heading       = heading,
                title         = record.get("title", ""),
                authors       = record.get("authors", []),
                year          = record.get("year", 0),
                month         = record.get("month", 0),
                category      = record.get("category", ""),
                abstract      = record.get("abstract", ""),
                section_idx   = section_idx,
                chunk_idx     = chunk_idx,
                total_chunks  = 0,   # filled in below
                token_count   = tok,
                reference_count = record.get("reference_count", 0),
            )
            all_chunks.append(chunk)

        section_idx += 1

    # Fill in total_chunks now that we know the count
    for chunk in all_chunks:
        chunk.total_chunks = len(all_chunks)

    return all_chunks


def chunk_all_papers() -> int:
    """
    Read every parsed JSON from PARSED_DIR,
    chunk it, save results to CHUNKS_DIR as JSONL.
    Returns total chunks produced.
    """
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    parsed_files = list(PARSED_DIR.glob("*.json"))

    if not parsed_files:
        logger.warning("No parsed files found. Run ingestion first.")
        return 0

    logger.info(f"Chunking {len(parsed_files)} papers...")
    total_chunks = 0

    for fpath in tqdm(parsed_files, desc="Chunking"):
        arxiv_id = fpath.stem
        out_path = CHUNKS_DIR / f"{arxiv_id}.jsonl"

        if out_path.exists():
            # Count existing chunks
            with open(out_path, encoding="utf-8") as f:
                n = sum(1 for _ in f)
            total_chunks += n
            continue

        record = json.loads(fpath.read_text(encoding="utf-8"))
        chunks = chunk_paper(record)

        if not chunks:
            logger.warning(f"No chunks produced for {arxiv_id}")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        total_chunks += len(chunks)
        logger.success(f"{arxiv_id} → {len(chunks)} chunks")

    logger.info(f"Total chunks produced: {total_chunks}")
    return total_chunks


def chunk_parsed_paper(parsed_path: Path) -> list[dict]:
    """
    Load a parsed JSON file and return chunks as plain dicts.
    Wrapper around chunk_paper() for use in the single-paper pipeline.
    """
    record = json.loads(parsed_path.read_text(encoding="utf-8"))
    chunks = chunk_paper(record)
    return [c.to_dict() for c in chunks]

if __name__ == "__main__":
    chunk_all_papers()