"""doc-to-vectors: structure-aware chunking and hybrid retrieval for RAG.

>>> from docvec import chunk_document
>>> chunks = chunk_document("# Title\\n\\nHello there.")
>>> chunks[0].path
'Title'
"""

from .blocks import Block, outline, parse, split_front_matter
from .bm25 import BM25, tokenize
from .chunk import (
    Chunk,
    ChunkConfig,
    chunk_document,
    chunk_text,
    estimate_tokens,
    split_sentences,
)
from .search import Hit, Index, cosine, reciprocal_rank_fusion

__all__ = [
    "BM25",
    "Block",
    "Chunk",
    "ChunkConfig",
    "Hit",
    "Index",
    "chunk_document",
    "chunk_text",
    "cosine",
    "estimate_tokens",
    "outline",
    "parse",
    "reciprocal_rank_fusion",
    "split_front_matter",
    "split_sentences",
    "tokenize",
]

__version__ = "1.0.0"
