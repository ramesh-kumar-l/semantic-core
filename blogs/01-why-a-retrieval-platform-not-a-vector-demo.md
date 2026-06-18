---
title: "Why I Built a Retrieval *Platform*, Not a Vector-Search Demo"
subtitle: "Most semantic-search projects die at the first real requirement. Here is the architecture that doesn't."
tags: [Semantic Search, System Design, FastAPI, Information Retrieval, Software Architecture]
series: "Semantic Core Service"
part: 1
---

# Why I Built a Retrieval *Platform*, Not a Vector-Search Demo

There is a version of semantic search that fits in a tweet:

```python
emb = model.encode(docs)
hits = cosine(model.encode(query), emb).topk(5)
```

It is genuinely magical the first time it works. It is also the exact point where most
projects stop — and the exact point where every real requirement starts to break it.

I wanted to build the thing that comes *after* that tweet. Not a bigger demo, but a small,
honest **retrieval platform**: something you can run on a laptop with zero infrastructure, and
that can grow toward a production search stack one feature flag at a time, without a rewrite.

This is the first post in a series about that system — **Semantic Core Service** — and it is
about the decision that shaped everything else: treating retrieval as **layers**, not a
function call.

---

## The problem: the demo-to-production cliff

Watch what happens to the five-line demo as soon as it meets reality.

- **"Search for the exact title."** Pure vector search fuzzes everything into a vibe. A user
  who types an exact product code or movie title wants *that row*, and cosine similarity will
  happily rank three thematically-similar-but-wrong results above it.
- **"Keep tenant A's data away from tenant B."** Now you need isolation, and a single global
  index doesn't have it.
- **"The results are bad for this one query."** With one opaque scoring function, you have no
  knob to turn and no signal telling you *why*.
- **"Make it better over time."** A static index never improves. Real search learns from the
  clicks it receives.
- **"Show me everything related to this person across all content."** That is a graph question,
  and a vector index cannot answer it.

Each of these is a reasonable ask. Each one is a brick wall for the five-line version. Teams
respond in one of two bad ways: they bolt hacks onto the toy until it's an unmaintainable ball
of special cases, or they panic and adopt a heavyweight search platform on day one — a
distributed cluster to serve 200 documents.

Semantic Core Service is a bet on the **middle ground**: start local and simple, but with the
*shape* of a real system, so growth is additive rather than a rebuild.

---

## The design: four layers, one app, every layer optional

The whole platform is four stacked capabilities behind a single FastAPI app:

```
                  POST /content   POST /similar   POST /feedback
                        │               │               │
        ┌───────────────┴───────────────┴───────────────┴─────────────┐
        │  1. Vector retrieval     embed → nearest neighbours          │
        │  2. Hybrid retrieval     fuse vectors + BM25 keyword search  │
        │  3. Feedback ranking     learn from clicks, reorder results  │
        │  4. Semantic graph       extract entities, traverse relations│
        └──────────────────────────────────────────────────────────────┘
```

The non-obvious part is that **every layer is feature-flagged**. The system runs with nothing
but vector search, and you opt into hybrid search, learned ranking, planning, query
intelligence, and the graph independently. From `app/core/config.py`:

```python
"hybrid":   {"enabled": os.getenv("HYBRID_ENABLED", "false").lower() == "true"},
"ranking":  {"enabled": os.getenv("RANKING_ENABLED", "false").lower() == "true"},
"learning": {"enabled": os.getenv("LEARNING_ENABLED", "true").lower() == "true"},
"planner":  {"enabled": os.getenv("PLANNER_ENABLED", "false").lower() == "true"},
"graph":    {"enabled": os.getenv("GRAPH_ENABLED", "true").lower() == "true"},
```

This matters for two reasons that go beyond tidiness:

1. **Debuggability.** When relevance looks wrong, you bisect the *pipeline*, not the code. Turn
   off reranking and learning; is fusion the problem? Turn off hybrid; is it the embeddings?
   Each flag is a controlled experiment you can run in seconds.
2. **Honest defaults.** The system is useful at every level of investment. You are never forced
   to stand up Qdrant and Neo4j to see value, but the path to both is already wired in.

---

## Stable interfaces are the actual product

The layers stay swappable because the seams between them are tiny, explicit contracts. The
entire storage abstraction is two methods:

```python
class VectorStore:
    def add(self, id: str, vector: list[float], text: str, metadata: dict) -> None: ...
    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]: ...
```

`FlatIndex` implements that with an in-memory cosine scan for instant local iteration.
`QdrantStore` implements the *same* contract against a production HNSW vector database. The API
layer never imports Qdrant directly — switching backends is one line in a factory:

```python
def get_vector_store(config):
    if config["vector_store"]["type"] == "qdrant":
        return QdrantStore(config)
    return FlatIndex()
```

The graph layer follows the same discipline: a `GraphDB` interface with `SQLiteGraph` for
local-first use and `Neo4jGraph` for when you outgrow it. This is just **dependency inversion**,
but applied where it pays off most — at the boundary between your business logic and the
infrastructure you haven't committed to yet. Business logic depends on the *interface*; the
infrastructure is a runtime choice.

---

## Graceful degradation is a feature, not a fallback

The single design rule I'm proudest of: **optional dependencies never hard-fail the service.**

- No `sentence-transformers`? Embeddings fall back to a deterministic hash vector. Quality
  drops; the API still serves.
- No Neo4j? The graph falls back to SQLite.
- No trained ML reranker? Ranking falls back to a rule-based reranker.
- LLM query rewrite times out? The original query is used.

Here is the embedding service making that choice at import time:

```python
try:
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    _USE_REAL = True
except Exception:
    _USE_REAL = False   # deterministic hash-embedding fallback
```

The payoff is that **`git clone` → `pip install` → `uvicorn` just works**, with no API keys, no
Docker, no database. "Local-first" stops being a marketing word and becomes a property you can
verify on a fresh machine. And in production, the same rule means a single dependency outage
degrades quality instead of taking the service down.

---

## Eating the platform's own dog food

A retrieval platform is only convincing if you can *see* it retrieve. So the repo ships three
curated datasets — and instead of the usual "lorem ipsum about beaches," they describe **real
engineering work**:

- `semantic_core` — the design decisions behind this very platform (13 documents)
- `netflix_clone` — system design for a Netflix-style streaming service (12 documents)
- `fitness_tracker` — building an offline-first Android fitness app (12 documents)

Seed them into a running instance:

```bash
python scripts/load_golden_examples.py
```

Then ask a vague, human question and watch semantic retrieval earn its keep:

```bash
curl -X POST http://localhost:8000/similar \
  -d '{"query":"how does hybrid search combine keyword and vector","k":3,"namespace":"semantic_core"}'
# top result: "Hybrid retrieval: fusing BM25 and vector search"  (score 0.66, ~16 ms)
```

No keyword in that query matches the document title, yet it comes back first. That is the whole
promise of semantic search — and across the seeded set, queries return in **11–21 ms** on the
in-memory backend. The three datasets reappear throughout this series as concrete, runnable
examples.

---

## What this buys you

The layered, flag-driven design turns a pile of features into a system with a spine:

- **You can reason about it.** Every behaviour traces to a flag and a module.
- **You can grow it.** Local FlatIndex → Qdrant; SQLite → Neo4j; rules → learned models — each
  is an additive change behind a stable interface.
- **You can trust it to start.** Nothing external is required to run.

In the next post I go down one level into the second layer — **hybrid retrieval** — and the
surprisingly nasty bugs that show up when you fuse two scoring systems that were never meant to
be added together.

---

*This is Part 1 of the Semantic Core Service series. The code is open source; every snippet
here maps directly to a file in the repo (`app/main.py`, `app/core/config.py`,
`app/vector_store/`, `app/services/embedding.py`). Follow along by seeding the golden examples
and turning the flags on one at a time.*
