"""Structure-aware chunking.

The rules, in priority order:

1. A chunk never splits a fenced code block or a table. If one is larger than
   the budget it becomes an oversized chunk and says so, because a broken code
   sample is worse than a large one.
2. Every chunk carries the heading path it came from (``Guide > Install >
   macOS``). Embeddings of bare paragraphs lose the topic that the outline was
   carrying for free.
3. Chunk ids are content hashes of ``source + breadcrumbs + text``, so
   re-indexing an unchanged document produces identical ids and the vector
   store only has to embed what actually changed.
4. Overlap is prose-only and counted inside the budget, so it cannot silently
   push a chunk past the limit you set.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .blocks import (
    KIND_CODE,
    KIND_HEADING,
    KIND_LIST,
    KIND_PARAGRAPH,
    KIND_TABLE,
    Block,
    parse,
)

__all__ = [
    "Chunk",
    "ChunkConfig",
    "chunk_document",
    "chunk_text",
    "estimate_tokens",
    "split_sentences",
]

# Characters per token. Deliberately conservative: overestimating keeps chunks
# inside real embedding limits, underestimating produces silent truncation.
RATIOS = {"prose": 4.0, "code": 3.2, "table": 3.4}

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])(?=\s+[\"'(\[]?[A-Z0-9])|(?<=[。！？])")


def _ratio_for(kind: str) -> float:
    if kind == KIND_CODE:
        return RATIOS["code"]
    if kind == KIND_TABLE:
        return RATIOS["table"]
    return RATIOS["prose"]


def estimate_tokens(text: str, kind: str = KIND_PARAGRAPH) -> int:
    """Conservative token estimate with no tokenizer dependency."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / _ratio_for(kind)))


def split_sentences(text: str) -> List[str]:
    """Split prose on sentence boundaries, keeping the terminators.

    Abbreviations such as "e.g." will split. Chunking only needs boundaries to
    be roughly right, and a wrong boundary costs a little cohesion, while a
    long unsplit block costs a truncated chunk.
    """
    parts = [p.strip() for p in _SENTENCE_BREAK.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _hard_split(text: str, budget_tokens: int, kind: str) -> List[str]:
    """Break an over-long sentence on word boundaries."""
    limit = max(1, int(budget_tokens * _ratio_for(kind)))
    if len(text) <= limit:
        return [text]
    out: List[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        out.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        out.append(remaining)
    return out


def _tail(text: str, overlap_chars: int) -> str:
    """Last ``overlap_chars`` characters, snapped forward to a word boundary."""
    if overlap_chars <= 0 or not text:
        return ""
    if len(text) <= overlap_chars:
        return ""
    idx = len(text) - overlap_chars
    space = text.find(" ", idx)
    if space < 0 or space >= len(text) - 1:
        return ""
    return text[space + 1:].strip()


@dataclass
class ChunkConfig:
    max_tokens: int = 400
    overlap: int = 0  # characters of prose carried into the next chunk
    breadcrumbs: bool = True
    separator: str = "\n\n"

    def __post_init__(self) -> None:
        if self.max_tokens < 16:
            raise ValueError("max_tokens must be >= 16")
        if self.overlap < 0:
            raise ValueError("overlap must be >= 0")

    @property
    def overlap_tokens(self) -> int:
        return math.ceil(self.overlap / RATIOS["prose"]) if self.overlap else 0


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    ordinal: int
    breadcrumbs: Tuple[str, ...]
    text: str
    kinds: Tuple[str, ...]
    tokens: int
    oversized: bool = False

    @property
    def path(self) -> str:
        return " > ".join(self.breadcrumbs)

    @property
    def embed_text(self) -> str:
        """What to embed and what to show: context first, then content."""
        if not self.breadcrumbs:
            return self.text
        return f"{self.path}\n\n{self.text}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "ordinal": self.ordinal,
            "breadcrumbs": list(self.breadcrumbs),
            "path": self.path,
            "text": self.text,
            "kinds": list(self.kinds),
            "tokens": self.tokens,
            "oversized": self.oversized,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Chunk":
        crumbs = payload.get("breadcrumbs")
        if crumbs is None:
            path = payload.get("path") or ""
            crumbs = tuple(p for p in path.split(" > ") if p)
        return cls(
            id=payload["id"],
            source=payload.get("source", ""),
            ordinal=int(payload.get("ordinal", 0)),
            breadcrumbs=tuple(crumbs),
            text=payload.get("text", ""),
            kinds=tuple(payload.get("kinds", ())),
            tokens=int(payload.get("tokens", 0)),
            oversized=bool(payload.get("oversized", False)),
        )


def _chunk_id(source: str, breadcrumbs: Sequence[str], text: str) -> str:
    payload = "\x00".join(["v1", source, ">".join(breadcrumbs), text])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _segments(blocks: Iterable[Block]) -> List[Tuple[Tuple[str, ...], List[Block]]]:
    """Group blocks under their heading path."""
    segments: List[Tuple[Tuple[str, ...], List[Block]]] = []
    stack: List[Tuple[int, str]] = []
    crumbs: Tuple[str, ...] = ()
    current: List[Block] = []

    def flush() -> None:
        nonlocal current
        if current:
            segments.append((crumbs, current))
        current = []

    for block in blocks:
        if block.kind == KIND_HEADING:
            flush()
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.text))
            crumbs = tuple(title for _, title in stack)
            current = [Block(KIND_HEADING, f"{'#' * block.level} {block.text}", block.level)]
            continue
        current.append(block)

    flush()
    return segments


def _render(block: Block) -> str:
    if block.kind == KIND_HEADING:
        return block.text
    if block.kind == KIND_CODE:
        return block.text
    return block.text


def _split_block(block: Block, budget: int) -> List[Block]:
    """Split a non-atomic block into budget-sized pieces."""
    text = block.text
    if block.kind == KIND_LIST:
        items: List[str] = []
        for line in text.split("\n"):
            if re.match(r"^\s*([-*+]|\d{1,9}[.)])\s+", line) and items:
                items.append(line)
            elif items:
                items[-1] = items[-1] + "\n" + line
            else:
                items.append(line)
        pieces: List[str] = []
        buf = ""
        for item in items:
            trial = f"{buf}\n{item}" if buf else item
            if buf and estimate_tokens(trial, KIND_LIST) > budget:
                pieces.append(buf)
                buf = item
            else:
                buf = trial
        if buf:
            pieces.append(buf)
        out: List[Block] = []
        for piece in pieces:
            if estimate_tokens(piece, KIND_LIST) <= budget:
                out.append(Block(KIND_LIST, piece))
            else:
                out.extend(Block(KIND_LIST, p) for p in _hard_split(piece, budget, KIND_LIST))
        return out

    sentences = split_sentences(text)
    pieces = []
    buf = ""
    for sentence in sentences:
        trial = f"{buf} {sentence}" if buf else sentence
        if buf and estimate_tokens(trial, block.kind) > budget:
            pieces.extend(_hard_split(buf, budget, block.kind))
            buf = sentence
        else:
            buf = trial
    if buf:
        pieces.extend(_hard_split(buf, budget, block.kind))
    return [Block(block.kind, p) for p in pieces if p.strip()]


def _pack(
    crumbs: Tuple[str, ...],
    blocks: List[Block],
    source: str,
    config: ChunkConfig,
    start: int,
) -> List[Chunk]:
    budget = config.max_tokens
    pieces: List[Tuple[Block, bool]] = []
    reserve = config.overlap_tokens
    # Leave two tokens of slack so the separator and ceiling arithmetic can
    # never push a packed chunk one token over the caller's limit.
    effective = max(8, budget - reserve - 2)

    for block in blocks:
        if estimate_tokens(block.text, block.kind) <= effective:
            pieces.append((block, False))
        elif block.atomic:
            pieces.append((block, True))
        else:
            pieces.extend((b, False) for b in _split_block(block, effective))

    chunks: List[Chunk] = []
    buf: List[Block] = []
    ordinal = start
    carry = ""

    def flush() -> str:
        nonlocal buf, carry, ordinal
        if not buf:
            return ""
        body = config.separator.join(_render(b) for b in buf)
        text = f"{carry}{config.separator}{body}" if carry else body
        kinds = tuple(dict.fromkeys(b.kind for b in buf))
        chunks.append(
            Chunk(
                id=_chunk_id(source, crumbs, text),
                source=source,
                ordinal=ordinal,
                breadcrumbs=crumbs,
                text=text,
                kinds=kinds,
                tokens=estimate_tokens(text, kinds[-1] if len(kinds) == 1 else KIND_PARAGRAPH),
                oversized=any(
                    b.atomic and estimate_tokens(b.text, b.kind) > effective for b in buf
                ),
            )
        )
        ordinal += 1
        carry = ""
        buf = []
        return body

    for block, oversized in pieces:
        if oversized:
            flush()
            buf = [block]
            flush()
            continue

        trial = buf + [block]
        text = config.separator.join(_render(b) for b in trial)
        if carry:
            text = f"{carry}{config.separator}{text}"
        if buf and estimate_tokens(text, KIND_PARAGRAPH) > budget:
            body = flush()
            if config.overlap and body:
                carry = _tail(body, config.overlap)
            buf = [block]
            continue
        buf = trial

    flush()
    return chunks


def chunk_document(
    text: str,
    source: str = "document",
    *,
    config: Optional[ChunkConfig] = None,
) -> List[Chunk]:
    """Chunk a Markdown or plain-text document."""
    cfg = config or ChunkConfig()
    blocks = parse(text)
    if not blocks:
        return []
    chunks: List[Chunk] = []
    ordinal = 0
    for crumbs, segment in _segments(blocks):
        produced = _pack(crumbs, segment, source, cfg, ordinal)
        chunks.extend(produced)
        ordinal += len(produced)
    return chunks


def chunk_text(text: str, source: str = "document", **kwargs) -> List[Chunk]:
    return chunk_document(text, source, **kwargs)
