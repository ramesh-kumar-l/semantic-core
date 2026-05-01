# Semantic Core Service

Semantic Core Service is a FastAPI-based retrieval platform with semantic search, hybrid retrieval, namespaced memory, feedback-aware ranking, and semantic graph capabilities.

- `FlatIndex` for simple in-memory search during local development
- `QdrantStore` for a production-style vector database workflow

Detailed product and architecture documentation:

- [docs/PRD.md](/e:/ClaudeProjects/SemanticCoreService/docs/PRD.md:1)

## Project Structure

```text
app/
  main.py                  FastAPI app + lifespan + logging middleware
  api/routes.py            POST /content, POST /similar
  core/config.py           Config dict (env-var driven)
  core/factory.py          get_vector_store(config) factory
  services/embedding.py    EmbeddingService (MiniLM / hash stub)
  vector_store/base.py     VectorStore interface
  vector_store/flat.py     FlatIndex - in-memory cosine search
  vector_store/qdrant.py   QdrantStore - qdrant-client wrapper
  schemas/requests.py      IngestRequest, SearchRequest
  schemas/responses.py     IngestResponse, SearchResponse
scripts/
  load_sample_data.py      Loads 80 samples and runs example queries
requirements.txt
docker-compose.yml
```

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

`sentence-transformers` is optional but recommended. If it is not available, the service falls back to a deterministic hash-based embedding stub. The API will still run, but semantic quality will be limited.

If you want authenticated Hugging Face Hub downloads, create a `.env` file in the project root with:

```env
HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

The project will automatically load `.env` values when it starts.

## Run Locally

Start the API with the default in-memory backend:

```bash
uvicorn app.main:app --reload
```

The service will be available at `http://localhost:8000`.

## Streamlit Testing Interface

Run the API first, then launch the demo UI:

```bash
streamlit run streamlit_app.py
```

The UI provides tabs for:
- Ingest (`POST /content`)
- Search (`POST /similar`)
- Feedback (`POST /feedback`)
- Semantic Ingest (`POST /semantic/ingest`)
- Semantic Query (`POST /semantic/query`)
- Graph Node Lookup (`GET /semantic/node/{node_id}`)
- Graph Query (`POST /graph_query/execute`)
- System Health (`GET /admin/health`)

If the API is not reachable, the UI shows a clear error instead of crashing.

## Run with Qdrant

Start Qdrant:

```bash
docker compose up -d
```

Qdrant dashboard: `http://localhost:6333/dashboard`

Set the backend and run the app:

```powershell
$env:VECTOR_STORE_TYPE = "qdrant"
uvicorn app.main:app --reload
```

```bash
VECTOR_STORE_TYPE=qdrant uvicorn app.main:app --reload
```

To switch back to the in-memory backend:

```powershell
$env:VECTOR_STORE_TYPE = "flat"
```

```bash
unset VECTOR_STORE_TYPE
```

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `VECTOR_STORE_TYPE` | `flat` | Backend selection: `flat` or `qdrant` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_COLLECTION` | `content` | Collection name |
| `VECTOR_DIM` | `384` | Embedding dimension |
| `HF_TOKEN` | none | Optional Hugging Face access token for authenticated model downloads |
| `AUTH_ENABLED` | `false` | Enable API key auth middleware |
| `API_KEYS` | empty | Comma-separated global API keys |
| `NAMESPACE_KEYS` | empty | JSON map of API key to allowed namespaces |
| `RATE_LIMIT_ENABLED` | `false` | Enable per-namespace in-memory token bucket |
| `RATE_LIMIT_RPM` | `60` | Requests per minute per namespace |
| `INTELLIGENCE_MODE` | `simple` | `simple` or `llm` query intelligence |
| `LLM_API_KEY` | empty | Anthropic API key for LLM intelligence mode |
| `LLM_MODEL` | `claude-haiku-4-5` | Anthropic model name |
| `LLM_TIMEOUT_S` | `5` | Timeout for LLM calls in seconds |
| `GRAPH_TYPE` | `sqlite` | `sqlite` or `neo4j` |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | empty | Neo4j password |

## Auth and Rate Limiting

- Auth is disabled by default so zero-env startup still works.
- When enabled, set `X-API-Key` on requests.
- Namespace-scoped keys can be configured with `NAMESPACE_KEYS`, example:

```bash
NAMESPACE_KEYS='{"tenant-key-1":["default","travel"]}'
```

- Rate limiting is in-memory and per-process. In multi-worker deployments, each worker has independent buckets.

## Neo4j Validation

```bash
docker compose up -d neo4j
GRAPH_TYPE=neo4j NEO4J_PASSWORD=testpassword python scripts/validate_neo4j.py
```

The script verifies node creation, edge traversal, bounded traversal, and deletion behavior.

## LLM Intelligence Mode

```bash
INTELLIGENCE_ENABLED=true INTELLIGENCE_MODE=llm LLM_API_KEY=sk-... uvicorn app.main:app --reload
```

If an LLM call fails or times out, the service falls back to the original query.

## API Examples

Add content:

```bash
curl -X POST http://localhost:8000/content \
  -H "Content-Type: application/json" \
  -d '{"text": "Hiking through mountain trails with breathtaking views"}'
```

Example response:

```json
{ "content_id": "3f8a1b2c-..." }
```

Search for similar content:

```bash
curl -X POST http://localhost:8000/similar \
  -H "Content-Type: application/json" \
  -d '{"query": "mountain adventure", "k": 5}'
```

Example response:

```json
{
  "results": [
    { "id": "3f8a1b2c-...", "score": 0.8721 },
    { "id": "7a2c4d5e-...", "score": 0.8103 }
  ]
}
```

## Sample Data Script

```bash
python scripts/load_sample_data.py
```

Or point it at another server:

```bash
python scripts/load_sample_data.py --url http://localhost:8000
```

The script loads 80 items across 8 categories and prints example query scores and latency.

## Performance Targets

| Backend | Search latency | Notes |
| --- | --- | --- |
| FlatIndex | < 5 ms | Linear scan; appropriate for smaller local datasets |
| Qdrant | < 200 ms local | HNSW index with cosine distance |

Example log output:

```text
2024-01-01 12:00:00 INFO app.api.routes search latency=3.2ms query='beach' k=5 hits=5
```

## Stable Interfaces

These interfaces are intended to remain stable:

```python
class VectorStore:
    def add(self, id: str, vector: List[float]) -> None: ...
    def search(self, vector: List[float], k: int) -> List[Tuple[str, float]]: ...

class EmbeddingService:
    def embed(self, input_data: Any) -> List[float]: ...
```

Business logic depends on `VectorStore`, and the API layer does not call Qdrant directly.
