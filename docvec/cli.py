"""Command line interface for doc-to-vectors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .chunk import ChunkConfig, chunk_document
from .search import Hit, Index

SUFFIXES = {".md", ".markdown", ".mdx", ".txt", ".rst"}


def collect(paths: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(
                p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in SUFFIXES
            )
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(raw)
    return found


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def build(paths: Sequence[str], config: ChunkConfig) -> Index:
    index = Index()
    for path in collect(paths):
        source = path.as_posix()
        index.add(chunk_document(read_text(path), source, config=config))
    return index


def _print_hits(hits: Sequence[Hit], width: int) -> None:
    for position, hit in enumerate(hits, start=1):
        snippet = " ".join(hit.chunk.text.split())
        if len(snippet) > width:
            snippet = snippet[: width - 1] + "\u2026"
        ranks = [f"lex={hit.lex_rank}"] if hit.lex_rank else []
        if hit.vec_rank:
            ranks.append(f"vec={hit.vec_rank}")
        rank_text = f"  [{', '.join(ranks)}]" if ranks else ""
        print(f"{position}. {hit.chunk.source}  ({hit.chunk.path or 'root'})")
        print(f"   score {hit.score:.4f}{rank_text}  {snippet}")
        print()


def cmd_chunks(args: argparse.Namespace) -> int:
    path = Path(args.path)
    chunks = chunk_document(
        read_text(path),
        path.as_posix(),
        config=ChunkConfig(max_tokens=args.max_tokens, overlap=args.overlap),
    )
    if args.json:
        print(json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False))
        return 0
    for chunk in chunks:
        flag = " OVERSIZED" if chunk.oversized else ""
        print(
            f"{chunk.ordinal:>3}  {chunk.tokens:>5}tok  "
            f"[{','.join(chunk.kinds)}]{flag}  {chunk.path or 'root'}"
        )
        print(f"      id={chunk.id}  chars={len(chunk.text)}")
    print(f"\n{len(chunks)} chunks")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    config = ChunkConfig(max_tokens=args.max_tokens, overlap=args.overlap)
    index = build(args.paths, config)
    index.save(args.out)
    stats = index.stats()
    print(json.dumps(stats, indent=2))
    print(f"wrote {args.out}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    index = Index.load(args.index)
    hits = index.search(args.query, limit=args.limit)
    if args.json:
        print(json.dumps([h.to_dict() for h in hits], indent=2, ensure_ascii=False))
    else:
        _print_hits(hits, args.width)
    return 0 if hits else 1


def cmd_stats(args: argparse.Namespace) -> int:
    print(json.dumps(Index.load(args.index).stats(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docvec",
        description="Structure-aware chunking and hybrid retrieval for RAG.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    chunks = sub.add_parser("chunks", help="inspect how a document chunks")
    chunks.add_argument("path")
    chunks.add_argument("--max-tokens", type=int, default=400)
    chunks.add_argument("--overlap", type=int, default=0)
    chunks.add_argument("--json", action="store_true")
    chunks.set_defaults(func=cmd_chunks)

    index = sub.add_parser("index", help="chunk files or directories into a JSON index")
    index.add_argument("paths", nargs="+")
    index.add_argument("--out", default="docvec-index.json")
    index.add_argument("--max-tokens", type=int, default=400)
    index.add_argument("--overlap", type=int, default=0)
    index.set_defaults(func=cmd_index)

    search = sub.add_parser("search", help="query a saved index")
    search.add_argument("query")
    search.add_argument("--index", default="docvec-index.json")
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--width", type=int, default=140)
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=cmd_search)

    stats = sub.add_parser("stats", help="summarise a saved index")
    stats.add_argument("--index", default="docvec-index.json")
    stats.set_defaults(func=cmd_stats)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: not found: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
