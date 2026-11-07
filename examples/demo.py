"""End-to-end demo: chunk, index, search, hybrid retrieval.

    python examples/demo.py
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docvec import ChunkConfig, Index, chunk_document  # noqa: E402

DOCS = Path(__file__).resolve().parent / "docs"
DIM = 1024
PUNCT = ".,:;!?()[]`\"'"


def embed(texts: Sequence[str]) -> List[List[float]]:
    """Deterministic feature-hashing embedder: a stand-in for a real model.

    In production you pass a sentence-transformer here. Hashing keeps this demo
    dependency-free while still producing meaningful cosine similarities,
    because the same word always lands in the same dimension. Sublinear term
    frequency and L2 normalisation keep long chunks from dominating.
    """
    vectors = []
    for text in texts:
        vector = [0.0] * DIM
        for word in (w.strip(PUNCT).lower() for w in text.split()):
            if len(word) < 3:
                continue
            bucket = int.from_bytes(hashlib.md5(word.encode("utf-8")).digest()[:4], "big") % DIM
            vector[bucket] += 1.0
        vector = [math.log1p(v) for v in vector]
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


def main() -> int:
    config = ChunkConfig(max_tokens=110, overlap=0)
    index = Index()
    print(f"chunking {DOCS}")
    for path in sorted(DOCS.glob("*.md")):
        chunks = chunk_document(path.read_text(encoding="utf-8"), path.name, config=config)
        index.add(chunks)
        print(f"  {path.name}: {len(chunks)} chunks")
    print()

    print("chunk map")
    print("-" * 78)
    for chunk in index.chunks.values():
        flag = " OVERSIZED" if chunk.oversized else ""
        print(f"  {chunk.ordinal:>2}  {chunk.tokens:>4}tok  {chunk.path or 'root'}{flag}")
    print()

    index.embed_all(embed)

    for query in (
        "rotate signing key",
        "who gets paged for a SEV1",
        "rollback across a migration",
    ):
        print(f'query: "{query}"')
        print("-" * 78)
        for rank, hit in enumerate(index.search(query, limit=3, embedder=embed), start=1):
            snippet = " ".join(hit.chunk.text.split())[:96]
            print(f"  {rank}. score={hit.score:.4f} lex={hit.lex_rank} vec={hit.vec_rank}")
            print(f"     {hit.chunk.path}")
            print(f"     {snippet}...")
        print()

    print("index stats")
    print("-" * 78)
    for key, value in index.stats().items():
        print(f"  {key:<12} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
