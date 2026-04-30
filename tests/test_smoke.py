"""
Smoke tests — one round-trip per endpoint, verifying status codes and
response shapes. Uses a session-scoped TestClient so the app lifespan
runs once for the whole suite.
"""
import pytest


# ── /content ──────────────────────────────────────────────────────────────────

class TestIngest:
    def test_returns_content_id(self, client):
        r = client.post("/content", json={"text": "FastAPI observability patterns"})
        assert r.status_code == 200
        data = r.json()
        assert "content_id" in data
        assert isinstance(data["content_id"], str) and data["content_id"]

    def test_namespace_param_accepted(self, client):
        r = client.post("/content", json={"text": "namespaced document", "namespace": "smoke"})
        assert r.status_code == 200
        assert "content_id" in r.json()


# ── /similar ──────────────────────────────────────────────────────────────────

class TestSearch:
    def test_returns_results_list(self, client):
        client.post("/content", json={"text": "machine learning ranking systems", "namespace": "smoke"})
        r = client.post("/similar", json={"query": "ranking", "k": 3, "namespace": "smoke"})
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_results_have_id_and_score(self, client):
        client.post("/content", json={"text": "vector similarity search", "namespace": "smoke"})
        r = client.post("/similar", json={"query": "vector search", "k": 1, "namespace": "smoke"})
        assert r.status_code == 200
        results = r.json()["results"]
        if results:
            assert "id" in results[0]
            assert "score" in results[0]


# ── /feedback ─────────────────────────────────────────────────────────────────

class TestFeedback:
    def test_records_or_disabled(self, client):
        r = client.post("/feedback", json={
            "query": "test query",
            "results": ["fake-id-1", "fake-id-2"],
            "clicked": "fake-id-1",
            "position": 0,
        })
        assert r.status_code == 200
        assert r.json()["status"] in ("ok", "disabled")


# ── /semantic/ingest ──────────────────────────────────────────────────────────

class TestSemanticIngest:
    def test_returns_id_and_type(self, client):
        r = client.post("/semantic/ingest", json={
            "text": "Alice attended the conference in London",
            "type": "event",
            "metadata": {"title": "AI Summit"},
        })
        assert r.status_code == 200
        data = r.json()
        assert "id" in data
        assert "type" in data
        assert isinstance(data["metadata"], dict)


# ── /semantic/query ───────────────────────────────────────────────────────────

class TestSemanticQuery:
    def test_returns_nodes_list(self, client):
        client.post("/semantic/ingest", json={"type": "person", "metadata": {"name": "Bob"}})
        r = client.post("/semantic/query", json={"type": "person", "k": 5})
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data
        assert "total" in data
        assert isinstance(data["nodes"], list)


# ── /graph_query/execute ──────────────────────────────────────────────────────

class TestGraphQuery:
    def test_returns_graph_response_shape(self, client):
        r = client.post("/graph_query/execute", json={"type": "event", "k": 5})
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data
        assert "traversal_steps" in data
        assert "truncated" in data
