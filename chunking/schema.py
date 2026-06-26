# chunking/schema.py
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Chunk:
    """
    A single chunk of text ready for embedding.
    Every field here becomes a Qdrant payload — filterable at query time.
    """
    # Identity
    chunk_id:    str   # "{arxiv_id}_{section_idx}_{chunk_idx}"
    arxiv_id:    str
    
    # Content
    text:        str
    heading:     str   # section this chunk came from
    
    # Paper metadata (for filtering)
    title:       str
    authors:     list[str]
    year:        int
    month:       int
    category:    str
    abstract:    str
    
    # Position info
    section_idx: int   # which section in the paper
    chunk_idx:   int   # which chunk within that section
    total_chunks: int  # total chunks in this paper
    
    # Quality signals
    token_count:      int
    reference_count:  int

    def to_dict(self) -> dict:
        return asdict(self)