# Quick Starter Guide

> The fastest path from `git clone` to "I understand this project and I can run it."
> If you are opening this repository for the first time, read this top to bottom once.
> It answers the questions a new engineer actually asks.

---

## 1. What is this project, in one paragraph?

**Semantic Core Service** is a modular **retrieval platform** built on FastAPI. It is *not*
a single embed-and-cosine demo — it is four stacked, independently toggle-able layers
behind one app:

1. **Vector retrieval** — embed text, find nearest neighbours.
2. **Hybrid retrieval** — fuse semantic vectors with BM25 keyword search.
3. **Feedback-driven ranking** — learn from clicks to reorder results.
4. **Semantic graph** — extract entities, build a graph, and traverse relationships.

It runs **locally with zero external infrastructure** (no database, no API keys required),
and every advanced capability degrades gracefully when its optional dependency is missing.

If you remember one sentence: *it begins as a local semantic-search tool and grows into a
hybrid, graph-aware retrieval platform without a rewrite.*

---

## 2. The 5-minute quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell / Git Bash)
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the API (defaults: in-memory vector store, namespaced memory + graph enabled)
uvicorn app.main:app --reload
```

The service is now live at **http://localhost:8000**.
Interactive API docs (Swagger) are auto-generated at **http://localhost:8000/docs**.

```bash
# 4. Load the three "golden example" datasets (see Section 5)
python scripts/load_golden_examples.py

# 5. Launch the visual testing UI in a second terminal
streamlit run streamlit_app.py
```

That is the whole loop: **run the API → seed data → explore in Streamlit.**

---

## 3. "Will it work on my machine?" (prerequisites & fallbacks)

| Requirement | Needed? | What happens if missing |
| --- | --- | --- |
| Python 3.11+ | **Yes** | — (see `.python-version`) |
| `pip install -r requirements.txt` | **Yes** | — |
| `sentence-transformers` (real embeddings) | Recommended | Falls back to a **deterministic hash embedding** stub. API still runs; semantic quality drops. |
| Internet / `HF_TOKEN` (first model download) | First run only | The `all-MiniLM-L6-v2` model downloads once, then is cached locally and runs **offline**. |
| Docker (Qdrant / Neo4j) | Optional | Defaults use **in-memory FlatIndex** + **SQLite graph**, so you need neither. |

**Key idea — graceful degradation is a design rule, not an accident.** The service is built
to start and serve with nothing but Python installed. You opt *into* heavier infrastructure.

---

## 4. How is the code organized? (the map you need)

```text
app/
  main.py              FastAPI app + lifespan: wires every service into app.state
  api/routes.py        ALL endpoints live here; orchestrates the feature-flagged pipeline
  core/
    config.py          Single config dict, read from environment variables
    factory.py         get_vector_store(config): chooses FlatIndex vs QdrantStore
  services/embedding.py   EmbeddingService: MiniLM, with hash-embedding fallback
  vector_store/        VectorStore interface + FlatIndex (in-memory) + QdrantStore
  memory/              MemoryService: per-namespace vector+BM25 indexes, lazy-loaded & persisted
  hybrid/              bm25.py (lexical) + fusion.py (weighted score blending)
  ranking/             Deterministic reranking
  learning/            Feedback store + rule-based & ML learning-to-rank (L2R)
  planner/             Rule-based & adaptive query planning (chooses alpha, k, ...)
  intelligence/        Optional query analysis, rewrite, multi-query (simple or LLM)
  graph/               GraphDB interface + SQLiteGraph + Neo4jGraph
  semantic/            SemanticService: metadata extraction, entity linking, graph queries
  graph_query/         Bounded BFS traversal planner + engine
  linking/             Auto-link engine: scores & connects related graph nodes on ingest
  observability/       Structured JSON logging, request IDs, latency timing, middleware
  auth/                Optional API-key + namespace-scoped auth middleware
  schemas/             Pydantic request/response models

scripts/
  load_sample_data.py      80 generic items across 8 categories (smoke data)
  load_golden_examples.py  The 3 curated "golden example" datasets (Section 5)
  validate_neo4j.py        Validates Neo4j backend behaviour
  train_l2r.py             Trains the optional ML learning-to-rank model

docs/        PRD, architecture, and FAANG-level deep dive
blogs/       Long-form articles about the design (great onboarding reading)
data/        Persisted namespaces (data/<namespace>/) and graph.db
tests/       Smoke + enhancement test suites
streamlit_app.py   Visual tester for every endpoint
```

**Mental model:** `main.py` wires components into `app.state` at startup; `routes.py` reads
feature flags and orchestrates them. Endpoints stay thin; logic lives in services. When you
want to understand a feature, find its flag in `core/config.py`, then follow it into
`routes.py`, then into its service module.

---

## 5. What are the "golden examples" and how do I see them?

The repo ships with **three curated datasets** that demonstrate the platform indexing *real
software-engineering knowledge*. They are seeded into both the **searchable content store**
and the **semantic graph**, so they are visible the instant the app launches.

| Namespace | What it is | Docs |
| --- | --- | --- |
| `semantic_core` | The engineering journey of building **this** platform | 13 |
| `netflix_clone` | System design for a **Netflix-style streaming** platform | 12 |
| `fitness_tracker` | Building an **offline-first Android fitness app** | 12 |

Each document is tagged with a consistent metadata convention that drives the graph:

| Metadata field | Meaning | Becomes graph node | Via edge |
| --- | --- | --- | --- |
| `role` / `people` | the engineering role that owned the work (`MLEngineer`, …) | `person` | `contains` |
| `module` / `location` | the subsystem it belongs to (`hybrid_search`, …) | `location` | `occurs_at` |
| `milestone` / `event` | the epic it shipped under (`E3 Hybrid Retrieval`, …) | `event` | `happens_during` |

### Seed them

```bash
# API must be running first (uvicorn app.main:app --reload)
python scripts/load_golden_examples.py            # seed all three
python scripts/load_golden_examples.py --reset    # remove prior graph nodes, then re-seed
```

After seeding, `GET /admin/health` reports:

```json
{
  "namespaces": ["fitness_tracker", "netflix_clone", "semantic_core"],
  "doc_counts": {"semantic_core": 13, "netflix_clone": 12, "fitness_tracker": 12},
  "graph_node_count": 90
}
```

(90 graph nodes = 37 document nodes + the `person` / `location` / `event` entities they link to.)

### See them — in Streamlit

Launch `streamlit run streamlit_app.py` and try:

- **Search tab** → namespace `semantic_core`, query *"how does hybrid search combine keyword and vector"* → returns ranked `{id, score}` results.
- **Semantic Query tab** → `type = document` → lists every document node; or `event = E3 Hybrid Retrieval` → the docs shipped under that milestone; or `person = MLEngineer` → everything that role owned.
- **Graph Node Lookup tab** → node id `scs-04`, direction `outbound` → the document plus its linked role/module/milestone nodes.
- **System Health tab** → the namespaces, document counts, and feature flags above.

### See them — with curl

```bash
# Semantic search inside a namespace
curl -X POST http://localhost:8000/similar \
  -H "Content-Type: application/json" \
  -d '{"query":"adaptive video streaming quality","k":3,"namespace":"netflix_clone"}'

# Graph: every document owned by the MLEngineer role
curl -X POST http://localhost:8000/semantic/query \
  -H "Content-Type: application/json" \
  -d '{"person":"MLEngineer","k":25}'

# Graph: one node and its neighbours
curl "http://localhost:8000/semantic/node/scs-04?direction=outbound"
```

A real `node/scs-04` response (abridged) — note how the document fans out to entity nodes:

```json
{
  "node": {"id": "scs-04", "type": "document",
           "metadata": {"title": "Hybrid retrieval: fusing BM25 and vector search",
                        "people": ["MLEngineer"], "location": "hybrid_search",
                        "event": "E3 Hybrid Retrieval"}},
  "neighbors": [
    {"id": "person_mlengineer", "type": "person"},
    {"id": "event_e3_hybrid_retrieval", "type": "event"},
    {"id": "location_hybrid_search", "type": "location"}
  ]
}
```

---

## 6. The API surface (what can I call?)

### Retrieval plane

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/content` | Ingest text into a namespace (embed + index). Returns a `content_id`. |
| `POST` | `/similar` | Top-k retrieval. Body: `query`, `k`, `namespace`, optional `filters`. |
| `POST` | `/feedback` | Record a click for learning-to-rank. |
| `PUT` | `/content/{id}` | Update a document (memory mode). |
| `DELETE` | `/content/{id}` | Delete a document. |

### Semantic / graph plane

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/semantic/ingest` | Turn an object into a graph node + auto-linked entities. |
| `POST` | `/semantic/query` | Query nodes by `type` / `person` / `location` / `event`. |
| `GET` | `/semantic/node/{id}` | A node and its neighbours (`direction`, `relation`, `depth`). |
| `POST` | `/graph_query/execute` | Bounded BFS traversal from anchor entities. |

### Operations

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/admin/health` | Feature flags, namespaces, doc counts, graph node count. |
| `GET` | `/docs` | Auto-generated Swagger UI. |

---

## 7. How do I turn features on? (configuration)

Everything is driven by **environment variables** read once in `app/core/config.py`. A `.env`
file in the project root is auto-loaded. The defaults give you a working system; you opt into
more.

| Variable | Default | Turns on |
| --- | --- | --- |
| `VECTOR_STORE_TYPE` | `flat` | `qdrant` for a real vector DB |
| `MEMORY_ENABLED` | `true` | Namespaced, persisted memory path |
| `PERSISTENCE_ENABLED` | `true` | Writes namespace data under `data/<namespace>/` |
| `HYBRID_ENABLED` | `false` | BM25 + vector fusion (`HYBRID_ALPHA` tunes the blend) |
| `RANKING_ENABLED` | `false` | Deterministic reranking |
| `LEARNING_ENABLED` | `true` | Feedback-aware (click) reranking |
| `PLANNER_ENABLED` | `false` | Per-query strategy planning (`PLANNER_MODE=adaptive`) |
| `INTELLIGENCE_ENABLED` | `false` | Query analysis / rewrite / multi-query |
| `INTELLIGENCE_MODE` | `simple` | `llm` to use an Anthropic model (`LLM_API_KEY`) |
| `GRAPH_ENABLED` | `true` | Semantic graph layer |
| `GRAPH_TYPE` | `sqlite` | `neo4j` for a real graph DB |
| `AUTH_ENABLED` | `false` | API-key auth (`X-API-Key`, `API_KEYS`, `NAMESPACE_KEYS`) |
| `RATE_LIMIT_ENABLED` | `false` | Per-namespace token-bucket rate limiting |

Example — enable hybrid search and the adaptive planner for a session:

```bash
# PowerShell
$env:HYBRID_ENABLED="true"; $env:PLANNER_ENABLED="true"; $env:PLANNER_MODE="adaptive"
uvicorn app.main:app --reload
```

```bash
# bash
HYBRID_ENABLED=true PLANNER_ENABLED=true PLANNER_MODE=adaptive uvicorn app.main:app --reload
```

---

## 8. Where is my data stored?

| Data | Location | Backend |
| --- | --- | --- |
| Namespace vectors + text | `data/<namespace>/flat/` (`vectors.npy`, `ids.json`, `texts.json`, `metadata.json`) | FlatIndex on disk |
| Namespace keyword index | `data/<namespace>/bm25/index.json` | BM25 |
| Semantic graph | `data/graph.db` | SQLite |
| Click feedback | `data/feedback.jsonl` | Append-only log |
| Adaptive planner stats | `data/planner_store.json` | JSON |

Because everything persists to `data/`, **a restart does not lose your ingested content** —
namespaces are lazily re-hydrated from disk on first access. To wipe a namespace, delete its
`data/<namespace>/` folder (and remove its nodes from `data/graph.db` if you also want the
graph clean).

---

## 9. Running with real infrastructure (optional)

### Qdrant (production-style vector DB)

```bash
docker compose up -d                      # starts Qdrant on :6333
VECTOR_STORE_TYPE=qdrant uvicorn app.main:app --reload
# Dashboard: http://localhost:6333/dashboard
```

### Neo4j (production-style graph DB)

```bash
docker compose up -d neo4j
GRAPH_TYPE=neo4j NEO4J_PASSWORD=testpassword python scripts/validate_neo4j.py
```

The same code paths and APIs work against both the local and the external backends — that is
the point of the `VectorStore` and `GraphDB` interfaces.

---

## 10. How do I verify it works? (tests & benchmarks)

```bash
pytest -q                                   # smoke + enhancement suites
python scripts/load_sample_data.py          # 80-item smoke load + sample query latencies
python -m app.evaluation.benchmark          # Recall@K / MRR / latency on the bundled dataset
```

`tests/test_smoke.py` is the best first read to see the endpoints exercised end-to-end.

---

## 11. The 30-second request walkthrough (so the code isn't a black box)

**`POST /similar` (the search path):**

1. `routes.py:search_similar` resolves the namespace and, if enabled, runs query intelligence
   (analyze → rewrite → maybe multi-query) and the planner (pick `alpha`, `vector_k`, `bm25_k`).
2. `MemoryService.search` embeds the query and runs the per-namespace vector index.
3. If `HYBRID_ENABLED`, BM25 results are fused with the vector results by normalized score.
4. If `RANKING_ENABLED`, results are reranked deterministically.
5. If `LEARNING_ENABLED`, click history reorders results (rule-based, then optional ML).
6. Structured logs emit per-stage latency; retrieval warnings fire on empty/low-score/high-latency.
7. Top-k `{id, score}` is returned.

Each numbered step is a feature flag you can turn off to isolate behaviour. That is the whole
philosophy of the codebase in one list.

---

## 12. Common gotchas

- **First run is slow / prints HF warnings.** That is the one-time MiniLM download. It is cached
  afterwards and runs offline. No `HF_TOKEN` is required for the public model.
- **Search quality looks random.** You are probably on the hash-embedding fallback — install
  `sentence-transformers`.
- **`/similar` returns nothing for a namespace.** You have not seeded that namespace; run
  `scripts/load_golden_examples.py` (and make sure you pass the right `namespace`).
- **Graph endpoints 503.** `GRAPH_ENABLED` is `false`; the semantic layer is off.
- **Re-running the seed duplicates content docs.** `/content` always mints a new id, so re-seeding
  adds new copies into the namespace (graph nodes, keyed by stable id, are overwritten instead).
  Delete the `data/<namespace>/` folders before a clean re-seed.

---

## 13. Where to go next

- **`docs/PRD.md`** — product requirements, architecture diagram, request flows.
- **`docs/FAANG_PROJECT_GUIDE.md`** — interview-oriented deep dive into every module.
- **`blogs/`** — long-form articles on the design decisions (excellent onboarding reading).
- **`README.md`** — the original setup reference and environment-variable table.

Welcome aboard. Run it, seed it, break it, and read the logs — they were written to be read.
