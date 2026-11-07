"""Hybrid retrieval: BM25 plus optional dense vectors, fused with RRF.

Lexical and dense scores are not comparable numbers. BM25 is unbounded and
depends on corpus statistics; cosine similarity lives in [-1, 1] and depends on
the embedding model. Adding them, or min-max normalising them, means the mix
changes every time the corpus changes, which makes ranking regressions
impossible to diagnose.

Reciprocal Rank Fusion ignores the magnitudes entirely: each list contributes
``1 / (k + rank)``. It needs no calibration, no per-corpus tuning, and no
assumption that the two retrievers agree on scale. That is why Elasticsearch,
Weaviate and Azure AI Search all ship it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .bm25 import BM25
from .chunk import Chunk

__all__ = [
    "Hit",
    "Index",
    "cosine",
    "reciprocal_rank_fusion",
]

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must share a dimension")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """Fuse ranked id lists into one ranking.

    ``k`` damps the influence of the very top of each list. 60 is the value from
    the original paper and the default in Elasticsearch; it makes the difference
    between rank 1 and rank 2 matter less than agreement between retrievers.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings length")

    fused: Dict[str, float] = {}
    for weight, ranking in zip(weights, rankings):
        seen = set()
        for position, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + position)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass
class Hit:
    chunk: Chunk
    score: float
    lex_rank: Optional[int] = None
    vec_rank: Optional[int] = None

    def to_dict(self) -> dict:
        payload = self.chunk.to_dict()
        payload.update(
            {
                "score": round(self.score, 6),
                "lex_rank": self.lex_rank,
                "vec_rank": self.vec_rank,
            }
        )
        return payload


@dataclass
class Index:
    """An in-memory index over chunks, serialisable to JSON."""

    bm25: BM25 = field(default_factory=BM25)
    chunks: Dict[str, Chunk] = field(default_factory=dict)
    vectors: Dict[str, List[float]] = field(default_factory=dict)

    # -- building ---------------------------------------------------------

    def add(self, chunks: Iterable[Chunk], vectors: Optional[Mapping[str, Sequence[float]]] = None) -> int:
        added = 0
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
            self.bm25.add(chunk.id, chunk.embed_text)
            if vectors and chunk.id in vectors:
                self.vectors[chunk.id] = [float(x) for x in vectors[chunk.id]]
            added += 1
        return added

    def embed_all(self, embedder: Embedder) -> int:
        """Compute and store vectors for every chunk that lacks one."""
        pending = [c for cid, c in self.chunks.items() if cid not in self.vectors]
        if not pending:
            return 0
        vectors = embedder([c.embed_text for c in pending])
        if len(vectors) != len(pending):
            raise ValueError("embedder returned the wrong number of vectors")
        for chunk, vector in zip(pending, vectors):
            self.vectors[chunk.id] = [float(x) for x in vector]
        return len(pending)

    # -- searching --------------------------------------------------------

    def _lexical(self, query: str, depth: int) -> List[Tuple[str, float]]:
        return self.bm25.search(query, limit=depth)

    def _dense(self, query: str, embedder: Embedder, depth: int) -> List[Tuple[str, float]]:
        if not self.vectors:
            return []
        query_vec = list(embedder([query]))[0]
        scored = [
            (chunk_id, cosine(query_vec, vector))
            for chunk_id, vector in self.vectors.items()
            if chunk_id in self.chunks
        ]
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:depth]

    def search(
        self,
        query: str,
        limit: int = 10,
        *,
        embedder: Optional[Embedder] = None,
        depth: Optional[int] = None,
        k: int = 60,
        weights: Optional[Sequence[float]] = None,
    ) -> List[Hit]:
        """Return fused hits. Without ``embedder`` this is pure BM25."""
        pool = depth or max(limit * 5, 50)
        lexical = self._lexical(query, pool)

        if embedder is None or not self.vectors:
            return [
                Hit(chunk=self.chunks[doc_id], score=score, lex_rank=rank)
                for rank, (doc_id, score) in enumerate(lexical[:limit], start=1)
            ]

        dense = self._dense(query, embedder, pool)
        if not dense:
            return [
                Hit(chunk=self.chunks[doc_id], score=score, lex_rank=rank)
                for rank, (doc_id, score) in enumerate(lexical[:limit], start=1)
            ]

        lex_rank = {doc_id: i for i, (doc_id, _) in enumerate(lexical, start=1)}
        vec_rank = {doc_id: i for i, (doc_id, _) in enumerate(dense, start=1)}
        fused = reciprocal_rank_fusion(
            [[doc_id for doc_id, _ in lexical], [doc_id for doc_id, _ in dense]],
            k=k,
            weights=weights,
        )
        hits: List[Hit] = []
        for doc_id, score in fused[:limit]:
            chunk = self.chunks.get(doc_id)
            if chunk is None:
                continue
            hits.append(
                Hit(
                    chunk=chunk,
                    score=score,
                    lex_rank=lex_rank.get(doc_id),
                    vec_rank=vec_rank.get(doc_id),
                )
            )
        return hits

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "bm25": self.bm25.to_dict(),
            "chunks": [chunk.to_dict() for chunk in self.chunks.values()],
            "vectors": self.vectors,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Index":
        index = cls(bm25=BM25.from_dict(payload.get("bm25", {})))
        for raw in payload.get("chunks", []):
            chunk = Chunk.from_dict(raw)
            index.chunks[chunk.id] = chunk
        index.vectors = {
            cid: [float(x) for x in vec] for cid, vec in payload.get("vectors", {}).items()
        }
        return index

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "Index":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    # -- introspection ----------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.chunks)

    def stats(self) -> dict:
        sources = sorted({c.source for c in self.chunks.values()})
        tokens = [c.tokens for c in self.chunks.values()]
        return {
            "chunks": len(self.chunks),
            "sources": len(sources),
            "vocabulary": len(self.bm25.postings),
            "vectors": len(self.vectors),
            "avg_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0.0,
            "max_tokens": max(tokens) if tokens else 0,
            "oversized": sum(1 for c in self.chunks.values() if c.oversized),
        }
