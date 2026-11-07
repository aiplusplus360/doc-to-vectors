"""Tests for doc-to-vectors. Pure stdlib, no network, no tokenizer download."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from docvec import (
    BM25,
    ChunkConfig,
    Index,
    chunk_document,
    cosine,
    estimate_tokens,
    parse,
    reciprocal_rank_fusion,
    split_front_matter,
    split_sentences,
    tokenize,
)
from docvec.bm25 import BM25 as BM25Class
from docvec.blocks import KIND_CODE, KIND_HEADING, KIND_LIST, KIND_PARAGRAPH, KIND_TABLE

FENCE = "```"


class BlockParsingTests(unittest.TestCase):
    def test_headings_and_paragraphs(self):
        blocks = parse("# One\n\nFirst para.\n\n## Two\n\nSecond para.")
        self.assertEqual([b.kind for b in blocks],
                         [KIND_HEADING, KIND_PARAGRAPH, KIND_HEADING, KIND_PARAGRAPH])
        self.assertEqual(blocks[0].level, 1)
        self.assertEqual(blocks[2].level, 2)
        self.assertEqual(blocks[0].text, "One")

    def test_fenced_code_is_one_block(self):
        doc = f"Intro.\n\n{FENCE}python\nx = 1\ny = 2\n{FENCE}\n\nOutro."
        blocks = parse(doc)
        code = [b for b in blocks if b.kind == KIND_CODE]
        self.assertEqual(len(code), 1)
        self.assertEqual(code[0].text, f"{FENCE}python\nx = 1\ny = 2\n{FENCE}")
        self.assertTrue(code[0].atomic)

    def test_unclosed_fence_runs_to_eof(self):
        blocks = parse(f"{FENCE}\nnever closed")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].kind, KIND_CODE)
        self.assertIn("never closed", blocks[0].text)

    def test_tilde_fence(self):
        blocks = parse("~~~\nplain\n~~~")
        self.assertEqual(blocks[0].kind, KIND_CODE)

    def test_table_block(self):
        doc = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n\nAfter."
        blocks = parse(doc)
        self.assertEqual(blocks[0].kind, KIND_TABLE)
        self.assertEqual(blocks[0].text.count("\n"), 3)
        self.assertEqual(blocks[1].kind, KIND_PARAGRAPH)

    def test_list_block_and_continuation(self):
        doc = "- one\n- two\n  continued\n\n- three\n\nParagraph."
        blocks = parse(doc)
        self.assertEqual(blocks[0].kind, KIND_LIST)
        self.assertIn("three", blocks[0].text)
        self.assertEqual(blocks[1].kind, KIND_PARAGRAPH)

    def test_ordered_list(self):
        blocks = parse("1. first\n2. second\n")
        self.assertEqual(blocks[0].kind, KIND_LIST)

    def test_blockquote(self):
        blocks = parse("> quoted line\n> second line")
        self.assertEqual(blocks[0].text, "quoted line\nsecond line")

    def test_front_matter_stripped(self):
        meta, body = split_front_matter("---\ntitle: x\n---\n# Hi\n")
        self.assertEqual(meta, "title: x")
        self.assertEqual(parse(body)[0].kind, KIND_HEADING)

    def test_paragraph_ends_at_heading(self):
        blocks = parse("para one\n# H\npara two")
        self.assertEqual([b.kind for b in blocks],
                         [KIND_PARAGRAPH, KIND_HEADING, KIND_PARAGRAPH])

    def test_crlf_normalised(self):
        blocks = parse("# A\r\n\r\ntext\r\n")
        self.assertEqual(blocks[0].kind, KIND_HEADING)
        self.assertEqual(blocks[1].text, "text")


class ChunkingTests(unittest.TestCase):
    def test_code_block_never_split(self):
        body = "\n".join(f"print({i})" for i in range(200))
        doc = f"# Code\n\nIntro.\n\n{FENCE}python\n{body}\n{FENCE}\n\nOutro."
        chunks = chunk_document(doc, "doc.md", config=ChunkConfig(max_tokens=100))
        code_chunks = [c for c in chunks if "code" in c.kinds]
        self.assertEqual(len(code_chunks), 1)
        self.assertIn(body, code_chunks[0].text)
        self.assertTrue(code_chunks[0].text.startswith(FENCE + "python"))
        self.assertTrue(code_chunks[0].text.endswith(FENCE))
        self.assertTrue(code_chunks[0].oversized)

    def test_no_partial_fence_anywhere(self):
        body = "\n".join(f"line {i}" for i in range(120))
        doc = f"{FENCE}\n{body}\n{FENCE}\n\ntail paragraph"
        for chunk in chunk_document(doc, "d", config=ChunkConfig(max_tokens=60)):
            self.assertIn(chunk.text.count(FENCE) % 2, (0, 2 % 2), chunk.text[:40])

    def test_table_never_split(self):
        rows = "\n".join(f"| r{i} | v{i} |" for i in range(40))
        doc = f"| a | b |\n| --- | --- |\n{rows}"
        chunks = chunk_document(doc, "t.md", config=ChunkConfig(max_tokens=50))
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].oversized)
        self.assertEqual(chunks[0].text.count("\n"), 41)

    def test_breadcrumbs(self):
        doc = (
            "# Guide\n\nIntro.\n\n"
            "## Install\n\nSteps.\n\n"
            "### macOS\n\nUse brew.\n\n"
            "## Linux\n\nUse apt.\n"
        )
        paths = [c.path for c in chunk_document(doc, "g.md")]
        self.assertIn("Guide", paths)
        self.assertIn("Guide > Install > macOS", paths)
        self.assertIn("Guide > Linux", paths)
        self.assertNotIn("Guide > Install > Linux", paths)

    def test_heading_line_is_in_body(self):
        chunks = chunk_document("# Deployment\n\nRun it.", "d")
        self.assertIn("# Deployment", chunks[0].text)
        self.assertEqual(chunks[0].embed_text.split("\n")[0], "Deployment")

    def test_budget_respected(self):
        doc = " ".join(f"Sentence {i} contains several ordinary words here." for i in range(120))
        chunks = chunk_document(doc, "d", config=ChunkConfig(max_tokens=48))
        self.assertGreater(len(chunks), 3)
        for chunk in chunks:
            self.assertLessEqual(chunk.tokens, 48, chunk.text[:40])
            self.assertFalse(chunk.oversized)

    def test_ids_are_stable(self):
        doc = "# A\n\nSome text here.\n\n## B\n\nMore text here."
        first = [c.id for c in chunk_document(doc, "x.md")]
        second = [c.id for c in chunk_document(doc, "x.md")]
        self.assertEqual(first, second)

    def test_ids_depend_on_source(self):
        doc = "# A\n\nSome text here."
        a = chunk_document(doc, "a.md")[0].id
        b = chunk_document(doc, "b.md")[0].id
        self.assertNotEqual(a, b)

    def test_ids_change_when_text_changes(self):
        a = chunk_document("# A\n\nfirst version.", "d")[0].id
        b = chunk_document("# A\n\nsecond version.", "d")[0].id
        self.assertNotEqual(a, b)

    def test_overlap_carries_prose(self):
        doc = " ".join(f"Sentence number {i} has a few useful words in it." for i in range(40))
        chunks = chunk_document(doc, "d", config=ChunkConfig(max_tokens=40, overlap=60))
        self.assertGreater(len(chunks), 2)
        for previous, following in zip(chunks, chunks[1:]):
            carried = following.text.split("\n\n")[0]
            self.assertTrue(previous.text.endswith(carried), carried)

    def test_no_overlap_means_no_shared_prefix(self):
        doc = " ".join(f"Sentence number {i} has a few useful words in it." for i in range(40))
        chunks = chunk_document(doc, "d", config=ChunkConfig(max_tokens=40))
        second = chunks[1].text.split("\n\n")[0]
        self.assertFalse(chunks[0].text.endswith(second))

    def test_empty_document(self):
        self.assertEqual(chunk_document("", "d"), [])
        self.assertEqual(chunk_document("\n\n   \n", "d"), [])

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            ChunkConfig(max_tokens=4)
        with self.assertRaises(ValueError):
            ChunkConfig(overlap=-1)

    def test_estimate_tokens(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("abcd", KIND_PARAGRAPH), 1)
        self.assertGreater(estimate_tokens("a" * 100, KIND_CODE),
                           estimate_tokens("a" * 100, KIND_PARAGRAPH))

    def test_split_sentences(self):
        parts = split_sentences("One. Two! Three?")
        self.assertEqual(len(parts), 3)
        self.assertEqual(split_sentences("no terminator"), ["no terminator"])

    def test_chunk_round_trip(self):
        chunk = chunk_document("# T\n\nBody text.", "s.md")[0]
        again = type(chunk).from_dict(chunk.to_dict())
        self.assertEqual(chunk, again)


class BM25Tests(unittest.TestCase):
    def test_idf_is_positive_for_common_terms(self):
        index = BM25()
        for doc_id in ("a", "b", "c"):
            index.add(doc_id, "common word here")
        self.assertGreater(index.idf("common"), 0.0)

    def test_idf_zero_for_unknown_terms(self):
        index = BM25()
        index.add("a", "hello")
        self.assertEqual(index.idf("absent"), 0.0)

    def test_ranks_focused_document_first(self):
        index = BM25()
        index.add("focused", "python python python testing")
        index.add("buried", "python " + " ".join(f"filler{i}" for i in range(99)))
        ranked = index.search("python")
        self.assertEqual(ranked[0][0], "focused")
        self.assertGreater(ranked[0][1], ranked[1][1])

    def test_empty_query_and_empty_index(self):
        self.assertEqual(BM25().search("anything"), [])
        index = BM25()
        index.add("a", "hello")
        self.assertEqual(index.search("nothing-matches"), [])
        self.assertEqual(index.search(""), [])

    def test_cjk_tokenisation(self):
        tokens = tokenize("\u673a\u5668\u5b66\u4e60")
        self.assertEqual(len(tokens), 7)  # 4 unigrams + 3 bigrams
        self.assertIn("\u673a\u5668", tokens)

    def test_mixed_script_tokenisation(self):
        tokens = tokenize("Python \u5b66\u4e60 guide")
        self.assertEqual(tokens[0], "python")
        self.assertIn("guide", tokens)

    def test_remove(self):
        index = BM25()
        index.add("a", "hello world")
        self.assertTrue(index.remove("a"))
        self.assertFalse(index.remove("a"))
        self.assertEqual(index.size, 0)
        self.assertEqual(index.postings, {})

    def test_re_add_replaces(self):
        index = BM25()
        index.add("a", "hello world")
        index.add("a", "goodbye")
        self.assertEqual(index.search("hello"), [])
        self.assertEqual(index.search("goodbye")[0][0], "a")

    def test_persistence(self):
        index = BM25()
        index.add("a", "alpha beta")
        index.add("b", "beta gamma")
        restored = BM25Class.from_dict(json.loads(json.dumps(index.to_dict())))
        self.assertEqual(restored.search("beta"), index.search("beta"))
        self.assertEqual(restored.size, 2)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            BM25(k1=0)
        with self.assertRaises(ValueError):
            BM25(b=1.5)

    def test_length_normalisation_uses_real_average(self):
        index = BM25()
        index.add("short", "term")
        index.add("long", "term " + " ".join(f"w{i}" for i in range(60)))
        self.assertGreater(index.avg_length, 10)


class FusionAndIndexTests(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine([0, 0], [1, 1]), 0.0)
        with self.assertRaises(ValueError):
            cosine([1, 0], [1])

    def test_rrf_prefers_agreement(self):
        fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
        self.assertEqual(fused[0][0], "b")
        self.assertGreater(fused[0][1], fused[1][1])

    def test_rrf_weights(self):
        unweighted = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
        weighted = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], weights=[2.0, 1.0])
        self.assertEqual(unweighted[0][0], "a")  # deterministic tie-break by id
        self.assertEqual(weighted[0][0], "a")
        self.assertGreater(weighted[0][1], unweighted[0][1])

    def test_rrf_rejects_bad_k(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["a"]], k=0)

    def build(self):
        index = Index()
        index.add(chunk_document("# Install\n\nRun pip install docvec.", "install.md"))
        index.add(chunk_document("# Deploy\n\nUse docker compose up.", "deploy.md"))
        index.add(chunk_document("# Troubleshoot\n\nIf docker fails, check logs.", "trouble.md"))
        return index

    def test_lexical_search_finds_right_source(self):
        index = self.build()
        hits = index.search("pip install", limit=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.source, "install.md")
        self.assertEqual(hits[0].lex_rank, 1)

    def test_empty_index_search(self):
        self.assertEqual(Index().search("anything"), [])

    def test_hybrid_sets_both_ranks(self):
        index = self.build()
        ids = list(index.chunks)
        index.bm25 = _FakeBM25([ids[0], ids[1], ids[2]])
        index.vectors = {
            ids[0]: [1.0, 0.0],
            ids[1]: [0.0, 1.0],
            ids[2]: [1.0, 0.0],
        }

        def embedder(texts):
            return [[1.0, 0.0] for _ in texts]

        hits = index.search("anything", limit=3, embedder=embedder)
        self.assertEqual(len(hits), 3)
        for hit in hits:
            self.assertIsNotNone(hit.lex_rank)
            self.assertIsNotNone(hit.vec_rank)

    def test_hybrid_ranks_vector_favourite_first(self):
        index = self.build()
        ids = list(index.chunks)
        index.bm25 = _FakeBM25([ids[0], ids[1], ids[2]])
        index.vectors = {
            ids[0]: [0.0, 1.0],
            ids[1]: [0.0, 1.0],
            ids[2]: [1.0, 0.0],
        }

        def embedder(texts):
            return [[1.0, 0.0] for _ in texts]

        hits = index.search("anything", limit=3, embedder=embedder)
        self.assertEqual(hits[0].chunk.id, ids[2])
        self.assertEqual(hits[0].vec_rank, 1)
        self.assertEqual(hits[0].lex_rank, 3)

    def test_embed_all(self):
        index = self.build()

        def embedder(texts):
            return [[float(len(t)), 1.0] for t in texts]

        self.assertEqual(index.embed_all(embedder), 3)
        self.assertEqual(index.embed_all(embedder), 0)
        self.assertEqual(len(index.vectors), 3)

    def test_embed_all_validates_length(self):
        index = self.build()
        with self.assertRaises(ValueError):
            index.embed_all(lambda texts: [])

    def test_save_and_load(self):
        index = self.build()
        handle, path = tempfile.mkstemp(suffix=".json")
        os.close(handle)
        try:
            index.save(path)
            restored = Index.load(path)
            self.assertEqual(restored.stats()["chunks"], index.stats()["chunks"])
            self.assertEqual(
                [h.chunk.id for h in restored.search("docker", limit=2)],
                [h.chunk.id for h in index.search("docker", limit=2)],
            )
        finally:
            os.unlink(path)

    def test_stats(self):
        stats = self.build().stats()
        self.assertEqual(stats["chunks"], 3)
        self.assertEqual(stats["sources"], 3)
        self.assertEqual(stats["vectors"], 0)
        self.assertGreater(stats["vocabulary"], 0)

    def test_hit_serialisation(self):
        index = self.build()
        payload = index.search("docker", limit=1)[0].to_dict()
        for key in ("id", "source", "path", "text", "score", "lex_rank"):
            self.assertIn(key, payload)


class _FakeBM25:
    """Deterministic lexical ranking used to test fusion in isolation."""

    def __init__(self, order):
        self.order = list(order)

    def search(self, query, limit=10):
        return [(doc_id, 1.0 / (position + 1)) for position, doc_id in enumerate(self.order)][:limit]


if __name__ == "__main__":
    unittest.main()
