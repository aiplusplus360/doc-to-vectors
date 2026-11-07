"""Okapi BM25 that behaves itself on short chunks.

Chunks are short, which is exactly where naive tf-idf misbehaves: a term that
appears twice in a 40-word chunk looks more important than the same term twice
in a 400-word chunk, and idf goes negative once a term appears in more than half
the corpus. So:

* the tf component saturates (``k1``), and length normalisation uses ``b``
  against the *actual* average chunk length in this index;
* idf uses the Lucene form ``log(1 + (N - df + 0.5) / (df + 0.5))``, which is
  strictly positive, so a term in every chunk contributes ~0 instead of
  actively penalising.

Tokenisation emits character unigrams and bigrams for CJK runs, because those
scripts have no spaces and a single token per sentence matches nothing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["BM25", "tokenize", "DEFAULT_K1", "DEFAULT_B"]

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, with CJK expanded to unigrams and bigrams."""
    tokens: List[str] = []
    for raw in _WORD.findall(text.lower()):
        if _CJK.search(raw):
            chars = list(raw)
            tokens.extend(chars)
            tokens.extend(a + b for a, b in zip(chars, chars[1:]))
        else:
            tokens.append(raw)
    return tokens


@dataclass
class BM25:
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    tokenizer: Callable[[str], List[str]] = tokenize
    postings: Dict[str, Dict[str, int]] = field(default_factory=dict)
    lengths: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("k1 must be > 0")
        if not 0.0 <= self.b <= 1.0:
            raise ValueError("b must be in [0, 1]")

    # -- mutation ---------------------------------------------------------

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self.lengths:
            self.remove(doc_id)
        tokens = self.tokenizer(text)
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, tf in counts.items():
            self.postings.setdefault(token, {})[doc_id] = tf
        self.lengths[doc_id] = len(tokens)

    def add_many(self, documents: Iterable[Tuple[str, str]]) -> None:
        for doc_id, text in documents:
            self.add(doc_id, text)

    def remove(self, doc_id: str) -> bool:
        if doc_id not in self.lengths:
            return False
        for token, docs in list(self.postings.items()):
            if doc_id in docs:
                del docs[doc_id]
                if not docs:
                    del self.postings[token]
        del self.lengths[doc_id]
        return True

    # -- statistics -------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.lengths)

    @property
    def avg_length(self) -> float:
        if not self.lengths:
            return 0.0
        return sum(self.lengths.values()) / len(self.lengths)

    def idf(self, token: str) -> float:
        df = len(self.postings.get(token, {}))
        if df == 0:
            return 0.0
        n = self.size
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    # -- scoring ----------------------------------------------------------

    def scores(self, query: str) -> Dict[str, float]:
        """Score every document that shares at least one term with the query."""
        terms = [t for t in dict.fromkeys(self.tokenizer(query)) if t in self.postings]
        if not terms or not self.lengths:
            return {}
        avg = self.avg_length or 1.0
        candidates: Dict[str, float] = {}
        for token in terms:
            weight = self.idf(token)
            if weight <= 0:
                continue
            for doc_id, tf in self.postings[token].items():
                norm = 1.0 - self.b + self.b * (self.lengths[doc_id] / avg)
                candidates[doc_id] = candidates.get(doc_id, 0.0) + (
                    weight * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
                )
        return candidates

    def search(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        ranked = sorted(self.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:limit] if limit >= 0 else ranked

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "postings": self.postings,
            "lengths": self.lengths,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BM25":
        return cls(
            k1=float(payload.get("k1", DEFAULT_K1)),
            b=float(payload.get("b", DEFAULT_B)),
            postings={t: dict(d) for t, d in payload.get("postings", {}).items()},
            lengths={d: int(n) for d, n in payload.get("lengths", {}).items()},
        )
