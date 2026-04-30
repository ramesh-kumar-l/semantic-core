# SemanticCoreService â€” 10-Day Portfolio Sprint Plan

## Context

SemanticCoreService is a FastAPI retrieval platform for a Google Staff/Principal Engineer portfolio.
The goal is to publish 10 targeted enhancements in 10 days that a Google recruiter or interviewer
can see, click, and interrogate in under 10 minutes. Every enhancement must leave the server startable
with zero env vars. Ranked by:

**Score = (Staff Signal Ã— 3) + (Public Visibility Ã— 2) + (Technical Depth Ã— 2) + (Feasibility Ã— 1) âˆ’ (Risk Ã— 1)**

---

## Ranked Enhancements

---

### Rank 1 â€” E3: Complete LLM Integration in Intelligence Module
**Score: 35** (Signal=5Ã—3, Visibility=5Ã—2, Depth=4Ã—2, Feasibility=4Ã—1, Risk=2Ã—1)

**Signal:** AI/ML feature gating, graceful degradation, latency budget, key hygiene. Google's search/ML infra teams probe exactly this: "how do you add an external AI dependency without coupling the critical path?"

**Story hook:** "I wired Anthropic's API as a pluggable query intelligence layer â€” with a three-second timeout, structured fallback to rule-based synonym expansion, and feature-flag gating so the server starts with zero env vars â€” demonstrating how you add LLM capabilities without coupling the critical path to an external dependency."

**Files to change:**
- `app/intelligence/llm.py` â€” implement `AnthropicLLMClient` with `complete(prompt) -> str`, 5s timeout via httpx
- `app/intelligence/rewrite.py` â€” add `LLMQueryRewriter` subclass calling `AnthropicLLMClient`
- `app/intelligence/multi_query.py` â€” add `LLMMultiQueryGenerator` subclass
- `app/api/routes.py` â€” update `_run_intelligence()` factory: if `INTELLIGENCE_MODE=llm`, use LLM variants
- `app/core/config.py` â€” add `LLM_API_KEY`, `LLM_MODEL` (default `claude-haiku-4-5`), `LLM_TIMEOUT_S` under `intelligence`
- `requirements.txt` â€” no new deps (httpx already present)

**Effort:** 1.5 days

**Risk:** LLM call exceeds 200ms observability warning threshold. Mitigation: add `LLM_TIMEOUT_S` config, log timeout separately from the main route timer. Never log `LLM_API_KEY`. Fallback chain: LLM fails â†’ log warning â†’ return original query unchanged.

**Done-when:** `INTELLIGENCE_MODE=llm LLM_API_KEY=sk-... uvicorn app.main:app` starts; POST `/similar` returns LLM-rewritten query in response metadata. With zero env vars, server starts and rule-based rewrite is used. One pytest covering the fallback path (mock `AnthropicLLMClient` to raise, assert original query returned).

---

**Status:** COMPLETE (implemented and tested)
**Score: 29** (Signal=5Ã—3, Visibility=3Ã—2, Depth=5Ã—2, Feasibility=2Ã—1, Risk=4Ã—1)

**Signal:** Atomicity across vector + BM25 + graph. BM25 IDF rebuild on delete. Idempotent upsert pattern. This is a textbook staff system-design answer about distributed write coordination across heterogeneous stores.

**Story hook:** "Deleting a document from a hybrid retrieval system is a distributed systems problem â€” the vector index, BM25 IDF table, and graph all need to stay consistent. BM25 has no incremental delete: every deletion triggers an O(N) corpus rebuild, which is the trade-off between write simplicity and the alternative of a tombstone + lazy rebuild pattern."

**Files to change:**
- `app/vector_store/base.py` â€” add `delete(id: str) -> bool` abstract method
- `app/vector_store/flat.py` â€” pop from `_ids`, `_vectors`, `_texts`, `_metadata` by index
- `app/vector_store/qdrant.py` â€” `client.delete()` with `PointIdsList`; use same hash as `add()`
- `app/hybrid/bm25.py` â€” pop `_doc_tokens` by id; rebuild `_df` by iterating remaining tokens; recompute `_avg_dl`
- `app/memory/service.py` â€” add `delete(namespace, id) -> bool`; add `update(namespace, id, text)` as delete + re-ingest
- `app/graph/base.py` â€” add `delete_node(id: str) -> bool` abstract method
- `app/graph/sqlite_graph.py` â€” `DELETE FROM nodes WHERE id=?` + `DELETE FROM edges WHERE source=? OR target=?`
- `app/graph/neo4j_graph.py` â€” `MATCH (n {id: $id}) DETACH DELETE n`
- `app/persistence/flat_store.py` â€” re-serialize after delete (atomic write already handles this)
- `app/persistence/bm25_store.py` â€” re-serialize after IDF rebuild
- `app/api/routes.py` â€” add `DELETE /content/{content_id}` (204/404), `PUT /content/{content_id}` (200)
- `app/schemas/requests.py` â€” `UpdateRequest`
- `app/schemas/responses.py` â€” `DeleteResponse`

**Effort:** 2 days (Days 9â€“10)

**Risk:** BM25 IDF rebuild is O(N) per delete â€” document this explicitly. Partial delete (vector deleted, BM25 rebuild fails) leaves inconsistency. Mitigation: transaction-style: if BM25 rebuild fails, log error + return 500 without persisting; client must retry.

**Done-when:** POST `/content` â†’ id. DELETE `/content/{id}` â†’ 204. Second DELETE â†’ 404. POST `/similar` with deleted text no longer returns that id. PUT `/content/{id}` with new text â†’ 200; search returns new text. Two pytests: delete-then-search, update-then-search.

---

### Rank 3 â€” E4: API Key Authentication Layer
**Score: 29** (Signal=5Ã—3, Visibility=3Ã—2, Depth=3Ã—2, Feasibility=4Ã—1, Risk=2Ã—1)

**Signal:** Production hygiene: auth before public exposure, per-namespace scoping, 401 vs 403 distinction, constant-time comparison, secret injection via env. Google cares enormously about auth posture.

**Story hook:** "I implemented per-namespace API key scoping so a tenant with namespace 'finance' cannot query namespace 'hr' even with a valid global key â€” using constant-time comparison to prevent timing attacks, and exempting the health endpoint from auth so monitoring systems never get locked out."

**Files to change:**
- `app/core/config.py` â€” add `auth` section: `AUTH_ENABLED`, `API_KEYS` (comma-separated â†’ set), `NAMESPACE_KEYS` (JSON â†’ dict)
- `app/observability/middleware.py` OR new `app/auth/middleware.py` â€” `AuthMiddleware(BaseHTTPMiddleware)` using `hmac.compare_digest()`; exempt `/admin/health` and `/docs`
- `app/main.py` â€” register `AuthMiddleware` before `ObservabilityMiddleware`
- `app/schemas/responses.py` â€” add `ErrorResponse` for 401/403

**Effort:** 1 day

**Risk:** Wrong middleware order logs unauthorized payloads. Mitigation: auth middleware must run before observability. `AUTH_ENABLED` defaults false â€” add startup log: "Auth disabled: all requests accepted" so misconfiguration is visible. Namespace-key parsing: if `NAMESPACE_KEYS` JSON is malformed, fail fast at startup.

**Done-when:** `AUTH_ENABLED=true API_KEYS=test-key-123 uvicorn app.main:app`: POST `/similar` with no header â†’ 401; with `X-API-Key: wrong` â†’ 403; with `X-API-Key: test-key-123` â†’ 200. Zero env vars â†’ all requests accepted. Three pytests: no key, wrong key, correct key.

---

### Rank 4 â€” E5: ML-Based L2R Training Pipeline
**Score: 28** (Signal=4Ã—3, Visibility=3Ã—2, Depth=3Ã—2, Feasibility=5Ã—1, Risk=1Ã—1)

**Signal:** Offline ML pipeline: feedback.jsonl â†’ feature extraction â†’ LogisticRegression â†’ model.pkl â†’ hot-load at startup. Shows ML-systems thinking: the pipeline pattern matters more than the model.

**Story hook:** "I built an offline learning-to-rank training pipeline where user click signals from a JSONL append log become a scikit-learn LogisticRegression model that gets hot-loaded at startup â€” the same pattern used in production recommendation systems."

**Files to change:**
- `app/learning/trainer.py` â€” verify `build_training_data()` min-sample guard exits with clear error; `train()` writes pkl to `./models/l2r.pkl`
- `scripts/train_l2r.py` â€” thin CLI: `python scripts/train_l2r.py --feedback ./data/feedback.jsonl --model ./models/l2r.pkl`
- `app/learning/l2r_model.py` â€” confirm pickle roundtrip stability; `min_samples` parameter guard

**Effort:** 0.5 day (trainer.py is 95% complete)

**Risk:** Training on <10 clicks produces worse-than-baseline models. Mitigation: minimum sample guard already partially in place; print clear error "Insufficient feedback: need â‰¥10 clicks, got N." Sklearn import guard already in `l2r_model.py` â€” verify it is correct.

**Done-when:** `python scripts/train_l2r.py --feedback ./data/feedback.jsonl --model ./models/l2r.pkl` succeeds with â‰¥10 feedback events, fails clearly with <10. `L2RModel.load('./models/l2r.pkl')` succeeds and server uses it on next restart.

---

### Rank 5 — E0: Streamlit Testing Interface
**Status:** COMPLETE (implemented)
**Score: 27** (Signal=3×3, Visibility=5×2, Depth=2×2, Feasibility=5×1, Risk=1×1)

**Signal:** Product sense, demo hygiene. A Google interviewer who can clone and click in 2 minutes forms a much stronger impression than one who has to read curl commands.

**Story hook:** "I built a zero-install demo interface that calls all seven FastAPI endpoints via httpx — not direct imports — so any recruiter or interviewer can clone the repo and interact with the system in under two minutes."

**Files to change:**
- New: `streamlit_app.py` at project root — tabs: Ingest, Search, Feedback, Semantic Ingest, Semantic Query, Graph Node Lookup, Graph Query, System Health; sidebar: base URL + API key; uses `httpx.Client`; System Health tab renders `/admin/health` as `st.metric` grid
- `requirements.txt` — add `streamlit>=1.35.0`

**Effort:** 1 day

**Risk:** FastAPI not running ? Streamlit shows unreadable httpx error. Mitigation: wrap all httpx calls in try/except and display "Service not reachable at {base_url}" with a red `st.error`. Add API key sidebar input now so E4 auth works without UI changes.

**Done-when:** `streamlit run streamlit_app.py` opens browser; all 7 endpoint forms render; ingest + search round-trip works end-to-end; README updated with `streamlit run streamlit_app.py` command.

**Completion note:** Delivered via `streamlit_app.py`, `requirements.txt` (`streamlit>=1.35.0`), and README run instructions.

---
**Status:** COMPLETE (implemented and tested)
**Score: 27** (Signal=4Ã—3, Visibility=3Ã—2, Depth=4Ã—2, Feasibility=3Ã—1, Risk=2Ã—1)

**Signal:** Interface surgery: extending an abstract base class, maintaining backward compatibility, implementing the same contract two ways (linear predicate vs Qdrant payload conditions). Classic staff-level API design.

**Story hook:** "I extended the VectorStore abstract interface with an optional filters parameter â€” backward-compatible by design â€” and implemented it two ways: as a post-hoc Python predicate in FlatIndex and as native Qdrant payload conditions, so the behavior is consistent regardless of which backend is active."

**Files to change:**
- `app/vector_store/base.py` â€” `search(vector, k, filters=None)` â€” `filters=None` preserves all existing callers
- `app/vector_store/flat.py` â€” store `self._metadata: List[dict]` parallel to `_ids`; in `add()` accept `metadata: dict = {}`; in `search()`, post-hoc predicate: `all(meta.get(k)==v for k,v in filters.items())`
- `app/vector_store/qdrant.py` â€” pass `query_filter=Filter(must=[FieldCondition(key=k, match=MatchValue(value=v))])` to `client.search()`; store `metadata` in upsert payload
- `app/schemas/requests.py` â€” `filters: Optional[Dict[str, str]] = None` on `SearchRequest`
- `app/memory/service.py` â€” thread `filters` through to `store.search()`
- `app/api/routes.py` â€” pass `body.filters`

**Effort:** 1.5 days

**Risk:** FlatIndex `add()` signature change must not break existing callers that omit `metadata`. Mitigation: `metadata: dict = {}` default. Qdrant: payload fields must have been stored at upsert time; update `QdrantStore.add()` to store `metadata` in payload alongside `text`.

**Done-when:** POST `/similar` with `{"query": "q", "filters": {"category": "news"}}` returns only docs ingested with that metadata. No filters â†’ identical behavior. Two pytests: matching filter, non-matching filter returns empty.

---

**Status:** COMPLETE (implemented and tested)
**Score: 26** (Signal=4Ã—3, Visibility=3Ã—2, Depth=2Ã—2, Feasibility=5Ã—1, Risk=1Ã—1)

**Signal:** Operational excellence: health checks showing system state â€” feature flags, per-tenant doc counts, graph stats â€” so an on-call engineer can diagnose retrieval quality issues without touching the database.

**Story hook:** "Any production service needs an operator-facing health endpoint that shows not just liveness, but system state â€” feature flags, per-tenant document counts, graph size, and feedback corpus depth â€” so an on-call engineer can diagnose retrieval quality issues without touching the database directly."

**Files to change:**
- `app/api/routes.py` â€” `GET /admin/health` route
- `app/schemas/responses.py` â€” `AdminHealthResponse` with `feature_flags`, `namespaces`, `doc_counts`, `graph_node_count`, `feedback_count`
- `app/memory/service.py` â€” add `namespace_stats() -> Dict[str, int]` returning `{ns: len(store._ids)}`

**Effort:** 1 day

**Risk:** `feedback_store.load_all()` re-reads file on every poll â€” maintain a counter in `FeedbackStore` instead. Graph `query_nodes({})` is a full scan â€” document the O(N) caveat and suggest caching.

**Done-when:** `curl http://localhost:8000/admin/health` with zero env vars returns JSON with all five fields. One pytest asserting response schema. Streamlit System Health tab renders this endpoint as a metric grid.

---

### Rank 8 â€” E6: Neo4j End-to-End Validation
**Score: 25** (Signal=4Ã—3, Visibility=3Ã—2, Depth=3Ã—2, Feasibility=4Ã—1, Risk=3Ã—1)

**Signal:** Distributed graph DB integration, connection resilience, docker-compose reproducibility. Converts "I wrote a Neo4j adapter" into "here is a reproducible test that proves it works."

**Story hook:** "I wrote an end-to-end validation script that spins up Neo4j via docker-compose and asserts node creation, edge traversal, and relation normalization â€” turning 'I wrote a Neo4j adapter' into 'here is a reproducible test that proves it works.'"

**Files to change:**
- `docker-compose.yml` â€” add `neo4j:5` service block; ports 7474/7687; `NEO4J_AUTH=neo4j/testpassword`; named volume `neo4j_data`
- New: `scripts/validate_neo4j.py` â€” connect, add 3 nodes, 2 edges, traverse, assert counts; retry connection up to 10Ã—2s; exit 1 on any failure
- `app/graph/neo4j_graph.py` â€” fix any bugs found during manual validation run

**Effort:** 1 day

**Risk:** Neo4j startup takes ~15 seconds. Validation script must retry with backoff. Relation normalization (`rel.upper()`) already in `add_edge()` â€” verify it is applied consistently in query methods too.

**Done-when:** `docker compose up -d neo4j` â†’ `GRAPH_TYPE=neo4j NEO4J_PASSWORD=testpassword python scripts/validate_neo4j.py` exits 0, all assertions printed as PASS. Zero env vars: server starts with SQLite default.

---

### Rank 9 â€” E7: Per-Namespace Rate Limiting
**Score: 24** (Signal=4Ã—3, Visibility=2Ã—2, Depth=3Ã—2, Feasibility=4Ã—1, Risk=2Ã—1)

**Signal:** Operational excellence: token bucket per tenant, preventing resource saturation, trade-off discussion (in-memory vs Redis for multi-worker) â€” classic SRE conversation at Google.

**Story hook:** "I added per-namespace token-bucket rate limiting in the middleware layer with a Retry-After header, and explicitly documented that in-memory buckets are per-process â€” so in a multi-worker deployment you'd need Redis, which is the exact conversation about horizontal scaling the interviewer wants."

**Files to change:**
- `app/observability/middleware.py` â€” add token-bucket dict `{namespace: {tokens, last_refill}}`; `threading.Lock` for safety; use `time.monotonic()`; return 429 with `Retry-After` header; exempt `/admin/health` and `/docs`
- `app/core/config.py` â€” `RATE_LIMIT_ENABLED` (default false), `RATE_LIMIT_RPM` (default 60)

**Effort:** 0.5 day

**Risk:** Token bucket with `time.monotonic()` is correct for single-process. Multi-worker uvicorn has isolated buckets. Document this limitation explicitly in README â€” it becomes an interview talking point.

**Done-when:** `RATE_LIMIT_ENABLED=true RATE_LIMIT_RPM=5` â€” 6 requests/second â†’ 6th returns 429 with `Retry-After`. Zero env vars â†’ no rate limiting. One pytest: RPM=1, fire 2 requests, assert second is 429.

---

### Rank 10 â€” E9: Schema Versioning for Persisted Data
**Score: 18** (Signal=3Ã—3, Visibility=1Ã—2, Depth=2Ã—2, Feasibility=5Ã—1, Risk=2Ã—1)

**Signal:** Data correctness: forward/backward compat detection on load. Important discipline but lowest portfolio ROI in a 10-day sprint â€” completely invisible without triggering a schema mismatch.

**Story hook:** "I added schema version fields to every persisted index so that a format change after a deployment produces a clear SchemaVersionError instead of silent data corruption â€” the minimum viable migration safety net."

**Files to change:**
- `app/persistence/flat_store.py` â€” wrap ids.json in `{"schema_version": 1, "data": ...}`; raise `SchemaVersionError` on mismatch
- `app/persistence/bm25_store.py` â€” same pattern on index.json

**Effort:** 0.5 day

**Risk:** Existing data files have no version field â€” loading them after this change will raise `SchemaVersionError`. Mitigation: treat missing version as version 0, emit a deprecation warning, continue loading. Document migration path.

**Done-when:** Loading a stale (version 0) index prints a deprecation warning but succeeds. Loading a mismatched version (e.g., version 2 when expecting 1) raises `SchemaVersionError` with clear message. One pytest per store.

---

## 10-Day Sprint Plan

Dependencies and ordering rationale:
- **E5 is nearly free** (trainer.py 95% done) â€” do it Day 1 alongside the test harness
- **E8 before E0** â€” Streamlit System Health tab needs the `/admin/health` endpoint
- **E0 before E3** â€” LLM integration is most impressive when the Streamlit UI can demo it with before/after comparison
- **E4 (auth) after E0/E3** â€” auth before "public exposure" framing in README; Streamlit sidebar already has API key input ready
- **E1 after E4** â€” filter queries are more interesting with a secured multi-tenant narrative
- **E2 last** â€” highest effort, most touching of core interfaces; test harness from Day 1 provides safety net

| Day | Enhancement | Deliverable |
|-----|-------------|-------------|
| 1   | Test harness + E5 | `tests/conftest.py`, `tests/test_smoke.py` (5 smoke tests); `scripts/train_l2r.py`; models/l2r.pkl roundtrip confirmed |
| 2   | E8 Admin Health | `GET /admin/health` live; `AdminHealthResponse` schema; `namespace_stats()` in MemoryService |
| 3   | E0 Streamlit UI | `streamlit_app.py`; all 7 endpoint tabs working; System Health tab rendering `/admin/health` |
| 4   | E3 LLM Integration (Part 1) | `AnthropicLLMClient`; `LLMQueryRewriter`; config vars wired |
| 5   | E3 LLM Integration (Part 2) + E4 Auth | `LLMMultiQueryGenerator`; factory dispatch; fallback tested. Auth middleware live. |
| 6   | E1 Metadata Filtering | `VectorStore.base` interface extended; FlatIndex predicate; Qdrant payload conditions |
| 7   | E7 Rate Limiting | Token bucket in middleware; 429 responses; Retry-After header |
| 8   | E6 Neo4j Validation | `docker-compose.yml` Neo4j block; `scripts/validate_neo4j.py` passes |
| 9   | E2 Delete (FlatIndex + BM25) | `DELETE /content/{id}` â†’ 204/404; FlatIndex.delete(); BM25 IDF rebuild |
| 10  | E2 Delete (Qdrant + Graph + Update) | `PUT /content/{id}` upsert; QdrantStore.delete(); GraphDB.delete_node(); full persistence |

---

## Verification â€” End-to-End Test Suite

After all enhancements, this sequence must work with zero env vars:

```bash
# 1. Server starts
uvicorn app.main:app --reload

# 2. Ingest
curl -s -X POST http://localhost:8000/content \
  -H "Content-Type: application/json" \
  -d '{"text": "FastAPI observability patterns", "namespace": "test"}'

# 3. Search
curl -s -X POST http://localhost:8000/similar \
  -H "Content-Type: application/json" \
  -d '{"query": "API monitoring", "k": 3, "namespace": "test"}'

# 4. Health check
curl -s http://localhost:8000/admin/health

# 5. Delete
curl -s -X DELETE http://localhost:8000/content/{id_from_step_2}

# 6. Streamlit
streamlit run streamlit_app.py

# 7. L2R training (after collecting feedback events)
python scripts/train_l2r.py --feedback ./data/feedback.jsonl --model ./models/l2r.pkl

# 8. Neo4j validation
docker compose up -d neo4j
GRAPH_TYPE=neo4j NEO4J_PASSWORD=testpassword python scripts/validate_neo4j.py

# 9. Auth enabled
AUTH_ENABLED=true API_KEYS=demo-key uvicorn app.main:app --reload

# 10. LLM mode
INTELLIGENCE_MODE=llm LLM_API_KEY=sk-... uvicorn app.main:app --reload
```

All ten enhancements leave `uvicorn app.main:app --reload` startable with zero env vars.



