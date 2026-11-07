# doc-to-vectors

**Structure-aware chunking and hybrid retrieval for RAG.** Splits Markdown at structural boundaries it understands, keeps every chunk traceable to its heading path, and ships a BM25 + reciprocal-rank-fusion retriever so you can measure retrieval quality before you pay for a vector database.

Zero dependencies. Standard library only. Python 3.9+.

```console
$ docvec chunks examples/docs/handbook.md --max-tokens 110
  0    41tok  [heading,paragraph]  Production Handbook
  1    41tok  [heading,paragraph]  Production Handbook > Deployments
  2    73tok  [heading,paragraph,code]  Production Handbook > Deployments > Standard deploy
  3    61tok  [heading,paragraph,code]  Production Handbook > Deployments > Rollback
```

---

## Retrieval quality is decided before your embedding model runs

I have watched teams swap embedding models three times to fix answers that were wrong because of chunking. A chunk that begins mid-sentence, a code sample cut in half, a paragraph separated from the heading that gave it meaning: no model recovers from that, because the information was destroyed upstream.

This library enforces four invariants instead.

### 1. Code blocks and tables are never split

A split fence turns into a truncated sample; a split table turns into a column of numbers with no header. If either exceeds the budget it ships as an `oversized` chunk and says so, because a large chunk is recoverable and a broken one is not.

```python
>>> from docvec import chunk_document, ChunkConfig
>>> body = "\n".join(f"print({i})" for i in range(200))
>>> doc = f"# Code\n\nIntro.\n\n```python\n{body}\n```"
>>> chunks = chunk_document(doc, "doc.md", config=ChunkConfig(max_tokens=100))
>>> [c.kinds for c in chunks]
[('heading', 'paragraph'), ('code',)]
>>> [c.oversized for c in chunks]
[False, True]
```

### 2. Every chunk carries its heading path

The outline was already carrying the topic. Throwing it away and embedding a bare paragraph loses it.

```python
>>> from docvec import chunk_document
>>> chunks = chunk_document(open("examples/docs/handbook.md").read(), "handbook.md")
>>> [c.path for c in chunks if "Rollback" in c.path]
['Production Handbook > Deployments > Rollback']
>>> [c for c in chunks if "Rollback" in c.path][0].embed_text.splitlines()[0]
'Production Handbook > Deployments > Rollback'
```

`embed_text` is the breadcrumb line followed by the body: embed that, not `text`.

### 3. Chunk ids are content hashes

`id = sha256(source + breadcrumbs + text)[:16]`. Re-index the same document and you get identical ids, so an incremental pipeline only embeds what changed, and a vector store keyed by id never accumulates orphaned duplicates:

```
200 documents, 1 edited
 -> 200 documents, identical ids everywhere except the edited chunks
```

### 4. Overlap is prose-only and inside the budget

Splitting prose produces adjacent chunks; repeating the tail sentence of one at the head of the next helps recall there. It never happens between code blocks, and the carried text counts against `max_tokens`, so `max_tokens=400` means 400, not 400 plus whatever the overlap happened to be.

---

## Retrieval: two retrievers, one ranking

BM25 alone is a surprisingly strong baseline for documentation, and it needs no model, no GPU and no vector store. Dense retrieval is better at paraphrase. You want both, and you cannot add their scores.

### Why you cannot add BM25 and cosine

BM25 is unbounded and depends on corpus statistics; cosine lives in `[-1, 1]` and depends on the model. Min-max normalising fixes the range and breaks the meaning: adding one document to the corpus changes every normalised score, so a ranking regression cannot be attributed to a code change.

Reciprocal Rank Fusion only looks at positions:

\[
\text{score}(d) = \sum_{r \in R} \frac{w_r}{k + \text{rank}_r(d)}
\]

With `k = 60` (the value from the original paper, and the default in Elasticsearch), agreement between retrievers dominates and the difference between rank 1 and rank 2 barely matters.

```python
>>> from docvec import reciprocal_rank_fusion
>>> reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
[('b', 0.03252247488101534), ('a', 0.01639344262295082), ('c', 0.016129032258064516)]
```

`b` appears in both lists, so it wins even though neither retriever ranked it first. That is the property you want, and it needs no calibration.

### BM25 that survives short documents

Chunks are 50 to 600 words, which breaks the textbook formula in two places:

- **idf goes negative** once a term appears in more than half the corpus, so a term in every chunk actively penalises documents. This library uses the Lucene form `log(1 + (N - df + 0.5) / (df + 0.5))`, which is strictly positive and tends to zero for ubiquitous terms.
- **tf must saturate.** `k1` bends the term-frequency curve and `b` normalises by the *actual* average chunk length in this index, not a constant.

CJK text has no spaces, so tokens are emitted as character unigrams plus bigrams. A single token per sentence matches nothing.

---

## Usage

### Index a directory

```console
$ docvec index ./docs --out docs.json --max-tokens 400 --overlap 80
{
  "chunks": 128,
  "sources": 14,
  "vocabulary": 4211,
  "vectors": 0,
  "avg_tokens": 287.4,
  "max_tokens": 400,
  "oversized": 2
}
wrote docs.json
```

### Search it

```console
$ docvec search "rotate the signing key" --index docs.json --limit 3
1. docs/secrets.md  (Secrets > Rotating the signing key)
   score 0.0328  [lex=1]  Rotation is a two-phase operation: publish the new key, wait for...
```

Exit code is `1` when nothing matches, so it composes in CI:

```bash
docvec search "rollback procedure" --index docs.json > /dev/null || echo "runbook coverage gap"
```

### Add dense retrieval

Bring your own embedder: anything callable that maps a list of strings to a list of vectors.

```python
from docvec import BM25, ChunkConfig, Index, chunk_document

def embed(texts):
    return model.encode(list(texts), normalize_embeddings=True)

index = Index()
index.add(chunk_document(open("docs/secrets.md").read(), "docs/secrets.md",
                         config=ChunkConfig(max_tokens=400, overlap=80)))
index.embed_all(embed)

hits = index.search("rotate the signing key", limit=5, embedder=embed)
for hit in hits:
    print(hit.score, hit.lex_rank, hit.vec_rank, hit.chunk.path)
    print(hit.chunk.text[:200])
```

`lex_rank` and `vec_rank` tell you *why* a result ranked where it did. When a ranking regresses, you can see which retriever moved instead of guessing.

### Tune fusion weights

If one retriever is measurably better on your corpus, weight it. This is the only knob RRF exposes, and it is a deliberate one:

```python
index.search(query, limit=10, embedder=embed, weights=[2.0, 1.0])  # lexical-first
```

---

## Architecture

```
docvec/
  blocks.py   Markdown block parser: headings, fences, tables, lists, quotes
  chunk.py    Budgeted packing, breadcrumbs, stable ids, prose-only overlap
  bm25.py     Okapi BM25 with Lucene idf, saturation, real avgdl, CJK n-grams
  search.py   Index, cosine, reciprocal rank fusion, JSON persistence
  cli.py      docvec chunks | index | search | stats
```

Each layer is usable alone. `blocks.parse` classifies; `chunk.chunk_document` packs; `bm25.BM25` scores; `search.Index` composes. Nothing imports an HTTP client and nothing reads a file except the CLI.

Time, embeddings and file types are all injected or pluggable, which is why the 51 tests run in under 10 milliseconds with no network.

---

## What it does not do

- **No PDF, DOCX or HTML extraction.** Those are container formats with their own failure modes. Extract text with `pypdf`, `python-docx` or `trafilatura`, then pass the text here: this library is about structure, not containers.
- **No tokenizer.** Token counts are conservative character-ratio estimates (`4.0` chars/token for prose, `3.2` for code, `3.4` for tables). Overestimating keeps chunks inside real limits; underestimating silently truncates.
- **No persistent vector store.** JSON in, JSON out. Swap in FAISS or pgvector behind the same `Index` interface when you outgrow it.
- **No abbreviation-aware sentence splitting.** `e.g.` splits. A wrong boundary costs cohesion; an unsplit 4000-character paragraph costs a truncated chunk, so the trade is deliberate.

---

## Tests

```bash
python -m unittest discover -s tests -v
```

51 tests covering the invariants that are easy to regress: fences surviving the splitter, tables never fragmenting, breadcrumbs resetting when headings go back up a level, ids staying stable across runs, overlap never appearing between code chunks, idf staying positive on tiny corpora, and RRF preferring agreement over a single first-place vote.

---

## License

MIT (c) 2026 aiplusplus360
