---
title: "Hybrid Retrieval Done Right: Fusing BM25 and Vectors"
subtitle: "Why semantic search alone loses, and the quiet bugs that appear when you add two score systems that were never meant to be added."
tags: [Information Retrieval, BM25, Vector Search, Hybrid Search, Search Relevance]
series: "Semantic Core Service"
part: 2
---

# Hybrid Retrieval Done Right: Fusing BM25 and Vectors

In [Part 1](01-why-a-retrieval-platform-not-a-vector-demo.md) I argued for treating retrieval as
layers. This post is about the second layer — **hybrid retrieval** — and it starts with a
confession: pure semantic search, the thing everyone is excited about, is *bad* at a whole class
of queries, and pretending otherwise is how you ship a search box people quietly stop using.

The fix is well known in principle (combine vectors with classic keyword search) and full of
sharp edges in practice. Let me show you both.

---

## Where each method fails, precisely

**Vector search** embeds text into a dense vector and ranks by cosine similarity. It is
extraordinary at *meaning*: it matches "mind-bending space thriller" to *Interstellar* even
though they share no words. But it has a characteristic failure — it is **fuzzy about exactness**.
Search a unique token (a SKU, an error code, a movie title, a username) and the vector model,
which was trained to smooth meaning, will surface three plausibly-related-but-wrong results above
the one you literally named.

**BM25** — the lexical workhorse behind decades of search — is the mirror image. It scores by
term frequency and inverse document frequency, so it is razor-sharp on exact tokens and rare
words, and it ranks the document you literally named first. But it is **blind to meaning**: a
query and a document that share zero tokens score zero, no matter how related they are.

These aren't competing options. They are complementary failure modes:

| Query | Vector search | BM25 |
| --- | --- | --- |
| "uplifting movie about second chances" | ✅ strong | ❌ no shared terms |
| "The Shawshank Redemption" | ⚠️ fuzzy | ✅ exact |
| "error E3 hybrid retrieval" | ⚠️ drifts | ✅ pins the token |
| "tracking workouts without internet" | ✅ paraphrase match | ⚠️ partial |

**Hybrid retrieval runs both and fuses the results**, so a query that either method would have
sunk gets rescued by the other.

---

## The fusion, and the trap inside it

The fusion formula in Semantic Core Service is a normalized weighted sum
(`app/hybrid/fusion.py`):

```python
score = alpha * vector_score + (1 - alpha) * bm25_score
```

with `alpha` (default `0.7`) tilting toward semantics. Looks trivial. Here is the trap: **you
cannot add a cosine score and a BM25 score directly.** Cosine similarity lives in roughly
`[0, 1]`. BM25 is an unbounded sum of IDF-weighted term contributions — a rare-term match can
score `9.0`. Add them raw and BM25's scale silently bulldozes the vector signal; your "hybrid"
search is secretly just BM25 with extra steps.

The fix is to **normalize each result set into `[0, 1]` before blending**:

```python
def _normalize(results):
    scores = [s for _, s in results]
    min_s, max_s = min(scores), max(scores)
    span = max_s - min_s if max_s > min_s else 1.0   # guard against divide-by-zero
    return {id_: (s - min_s) / span for id_, s in results}
```

Two details that look like nitpicks but are load-bearing:

1. **The `span` guard.** When every candidate has the same score (one result, or a tie), `span`
   would be zero and you'd divide by zero. Defaulting it to `1.0` keeps the function total.
2. **The union, not the intersection.** Fusion scores over `set(vec_ids) | set(bm25_ids)`, with a
   missing id contributing `0` from the method that didn't find it. That is the entire point —
   a result only vector search found still gets a fused score and a chance to rank.

```python
all_ids = set(vec_norm) | set(bm25_norm)
for id_ in all_ids:
    score = alpha * vec_norm.get(id_, 0.0) + (1 - alpha) * bm25_norm.get(id_, 0.0)
```

Min-max normalization is the simplest reasonable choice. It is also imperfect — it is sensitive
to outliers and it discards absolute score information (a top BM25 score of `0.5` and one of
`50` both normalize to `1.0`). That's a deliberate trade for predictability and zero
dependencies; the architecture leaves room to swap in reciprocal-rank fusion or a learned
combiner later, behind the same call site.

---

## The BM25 nobody imports

A small thing I care about: the BM25 here is ~80 lines with **no external dependency**
(`app/hybrid/bm25.py`). Standard parameters, `k1=1.5, b=0.75`, and the textbook scoring:

```python
idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
score += idf * tf_norm
```

Why not pull in a library? Because for a local-first platform, a transparent 80-line index you
can read in one sitting is worth more than a dependency you can't reason about — and it keeps the
"runs with nothing installed" promise from Part 1. The `b * dl / avg_dl` term is the part worth
internalizing: it is **length normalization**, stopping a long document from scoring highly just
because it contains a query term many times.

---

## The real lesson: make relevance bugs *observable*

Here is the thing about fusion that the formula hides: when it goes wrong, it goes wrong
*silently*. The endpoint still returns ten results. No exception, no 500. The results are just…
worse. That is the worst kind of bug — invisible until a user complains, and impossible to
reproduce from a stack trace.

So the most valuable code in this layer isn't the math. It is the **retrieval warnings** the
search path emits when the *shape* of the results looks pathological (`app/api/routes.py`):

```python
# the two methods found completely disjoint result sets
if vec_ids and bm25_ids and not (vec_ids & bm25_ids):
    logger.warning("retrieval_warning", extra={"type": "result_disagreement", "query": q})

# fusion produced nothing
if not vector_hits and not bm25_hits:
    logger.warning("retrieval_warning", extra={"type": "empty_fusion", "query": q})

# suspiciously wide score spread before reranking
if scores and (max(scores) - min(scores)) > 0.8:
    logger.warning("retrieval_warning", extra={"type": "large_score_divergence", "query": q})
```

Plus the post-hoc checks: `empty_results`, `low_score` (top hit below `0.3`), and `high_latency`
(over `200 ms`). None of these are errors. They are **structured signals** that turn "search
feels off lately" into a grep:

```bash
grep retrieval_warning logs.jsonl | jq 'select(.type=="result_disagreement") | .query'
```

`result_disagreement` is especially useful: when vectors and BM25 return *zero* overlapping ids,
it usually means the query is either a pure-keyword lookup the embeddings can't place, or a pure
semantic query with no lexical anchor — exactly the cases where your `alpha` is mistuned for that
traffic. The warning doesn't fix it; it *tells you it happened*, with the query attached, so you
can.

This is the broader philosophy of the whole project, surfacing here: **a retrieval system you
can't observe is a retrieval system you can't improve.** Every request also carries a request id
and per-stage latency (`embedding_ms`, `search_ms`), so a slow query tells you *which stage* was
slow, not just that something was.

---

## Try it on the golden examples

With the bundled datasets seeded (`python scripts/load_golden_examples.py`), turn hybrid on:

```bash
HYBRID_ENABLED=true uvicorn app.main:app --reload
```

Then compare a paraphrase query against an exact-token query in the same namespace:

```bash
# semantic win — no shared words with the target doc, still ranks it #1
curl -X POST :8000/similar -d '{"query":"adaptive video streaming quality","k":3,"namespace":"netflix_clone"}'
#   → "Adaptive bitrate playback with an ABR ladder"   (top score ~0.60)

# exact-token win — BM25 pins the milestone token that vectors would smear
curl -X POST :8000/similar -d '{"query":"E3 Hybrid Retrieval","k":3,"namespace":"semantic_core"}'
```

The first query is carried by the vector side; the second is rescued by BM25. Fusion is the
layer that lets one search box serve both kinds of user without you having to guess which kind
they are.

---

## Takeaways

- **Vectors and BM25 fail in opposite directions.** Hybrid retrieval exists to cover both.
- **Never add raw scores.** Normalize each result set first, or BM25's scale eats your semantics.
- **Fuse over the union.** A result only one method found still deserves to rank.
- **Instrument the failure modes.** Disagreement, empty fusion, and score divergence are signals,
  not errors — log them with the query attached.

Next up, [Part 3](03-teaching-search-to-learn-feedback-and-l2r.md): hybrid retrieval gives you a
good static ordering, but the best ranking signal you have is the one your users generate every
time they click. Here's how the platform learns from feedback — and an adaptive planner that
picks a retrieval strategy per query.

---

*Part 2 of the Semantic Core Service series. Code references: `app/hybrid/fusion.py`,
`app/hybrid/bm25.py`, `app/api/routes.py`.*
