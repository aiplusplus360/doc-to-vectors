"""Markdown block parser.

Chunking quality is decided here, not in the splitter. If you do not know which
lines are a fenced code block, a table, or a nested list item, every split is a
guess, and retrieval quality is capped by that guess. So this parser is
deliberately conservative and structural: it classifies blocks and never
rewrites their text.

Supported inputs: Markdown (ATX headings, fenced code, pipe tables, lists,
block quotes) and plain text (everything becomes paragraphs). For PDFs, HTML or
DOCX, extract text with the right tool first and pass it here: this module is
about structure, not container formats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["Block", "parse", "split_front_matter"]

_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*([^`]*)$")
_LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d{1,9}[.)])[ \t]+(.*)$")
_QUOTE = re.compile(r"^ {0,3}>[ ]?(.*)$")
_TABLE_SEP = re.compile(r"^ {0,3}\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$")
_FRONT_MATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.DOTALL)

KIND_HEADING = "heading"
KIND_PARAGRAPH = "paragraph"
KIND_CODE = "code"
KIND_TABLE = "table"
KIND_LIST = "list"
KIND_QUOTE = "quote"


@dataclass
class Block:
    """One structural unit of a document."""

    kind: str
    text: str
    level: int = 0

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("empty block")
        if self.kind == KIND_HEADING and not 1 <= self.level <= 6:
            raise ValueError("heading level must be 1..6")

    @property
    def atomic(self) -> bool:
        """True when splitting this block would destroy its meaning."""
        return self.kind in (KIND_CODE, KIND_TABLE)


def split_front_matter(text: str):
    """Separate a leading YAML front matter block from the body."""
    match = _FRONT_MATTER.match(text)
    if not match:
        return "", text
    return match.group(1), text[match.end():]


def _is_blank(line: str) -> bool:
    return not line.strip()


def _fence_closes(line: str, char: str, length: int) -> bool:
    stripped = line.lstrip(" ")
    if len(line) - len(stripped) > 3:
        return False
    if not stripped.startswith(char * length):
        return False
    rest = stripped[length:].strip()
    return rest == ""


def parse(text: str) -> List[Block]:
    """Classify ``text`` into blocks, preserving every original character."""
    _, body = split_front_matter(text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = body.split("\n")
    blocks: List[Block] = []
    i = 0
    total = len(lines)

    while i < total:
        line = lines[i]
        if _is_blank(line):
            i += 1
            continue

        heading = _HEADING.match(line)
        if heading:
            blocks.append(Block(KIND_HEADING, heading.group(2).strip(), len(heading.group(1))))
            i += 1
            continue

        fence = _FENCE.match(line)
        if fence:
            char, length = fence.group(1)[0], len(fence.group(1))
            chunk = [line]
            i += 1
            while i < total:
                chunk.append(lines[i])
                if _fence_closes(lines[i], char, length):
                    i += 1
                    break
                i += 1
            blocks.append(Block(KIND_CODE, "\n".join(chunk)))
            continue

        if "|" in line and i + 1 < total and _TABLE_SEP.match(lines[i + 1]) and "|" in lines[i + 1]:
            chunk = [line, lines[i + 1]]
            i += 2
            while i < total and not _is_blank(lines[i]) and "|" in lines[i]:
                chunk.append(lines[i])
                i += 1
            blocks.append(Block(KIND_TABLE, "\n".join(chunk)))
            continue

        if _QUOTE.match(line):
            chunk = []
            while i < total and _QUOTE.match(lines[i]):
                chunk.append(_QUOTE.match(lines[i]).group(1))
                i += 1
            blocks.append(Block(KIND_QUOTE, "\n".join(chunk)))
            continue

        if _LIST_ITEM.match(line):
            chunk = [line]
            i += 1
            while i < total:
                current = lines[i]
                if _is_blank(current):
                    # A blank line only ends the list when the next line is not
                    # an indented continuation or another item.
                    look = i + 1
                    while look < total and _is_blank(lines[look]):
                        look += 1
                    if look >= total:
                        i = look
                        break
                    nxt = lines[look]
                    if _LIST_ITEM.match(nxt) or nxt.startswith((" ", "\t")):
                        chunk.extend(lines[i:look])
                        i = look
                        continue
                    i = look
                    break
                if _LIST_ITEM.match(current) or current.startswith((" ", "\t")):
                    chunk.append(current)
                    i += 1
                    continue
                break
            blocks.append(Block(KIND_LIST, "\n".join(chunk).strip("\n")))
            continue

        chunk = [line]
        i += 1
        while i < total and not _is_blank(lines[i]):
            nxt = lines[i]
            if _HEADING.match(nxt) or _FENCE.match(nxt) or _LIST_ITEM.match(nxt) or _QUOTE.match(nxt):
                break
            if "|" in nxt and i + 1 < total and _TABLE_SEP.match(lines[i + 1]):
                break
            chunk.append(nxt)
            i += 1
        blocks.append(Block(KIND_PARAGRAPH, "\n".join(chunk)))

    return blocks


def outline(blocks: List[Block]) -> List[tuple]:
    """Return ``(level, title)`` for every heading, for CLI display."""
    return [(b.level, b.text) for b in blocks if b.kind == KIND_HEADING]
