import json
import uuid
from pathlib import Path

import pytest
import numpy as np

from app.core.config import config
from app.persistence import bm25_store, flat_store
from app.persistence.errors import SchemaVersionError


def test_e3_llm_fallback_returns_original_query(monkeypatch):
    from app.intelligence.llm import AnthropicLLMClient
    from app.intelligence.rewrite import LLMQueryRewriter

    def _boom(self, prompt: str):
        raise RuntimeError("timeout")

    monkeypatch.setattr(AnthropicLLMClient, "complete", _boom)
    rewriter = LLMQueryRewriter(AnthropicLLMClient(api_key="x"))
    query = "fast retrieval strategy"
    assert rewriter.rewrite(query) == query


def test_e4_auth_no_key_wrong_key_correct_key(client):
    auth_cfg = config["auth"]
    auth_cfg["enabled"] = True
    auth_cfg["api_keys"] = {"test-key-123"}
    auth_cfg["namespace_keys"] = {}
    try:
        no_key = client.post("/similar", json={"query": "x", "k": 1})
        assert no_key.status_code == 401

        wrong = client.post("/similar", json={"query": "x", "k": 1}, headers={"X-API-Key": "wrong"})
        assert wrong.status_code == 403

        ok = client.post("/similar", json={"query": "x", "k": 1}, headers={"X-API-Key": "test-key-123"})
        assert ok.status_code == 200
    finally:
        auth_cfg["enabled"] = False
        auth_cfg["api_keys"] = set()


def test_e7_rate_limit_returns_429(client):
    rate_cfg = config["rate_limit"]
    rate_cfg["enabled"] = True
    rate_cfg["rpm"] = 1
    try:
        first = client.post("/similar", json={"query": "x", "k": 1, "namespace": "rate-limit-test"})
        assert first.status_code == 200
        second = client.post("/similar", json={"query": "x", "k": 1, "namespace": "rate-limit-test"})
        assert second.status_code == 429
        assert "Retry-After" in second.headers
    finally:
        rate_cfg["enabled"] = False
        rate_cfg["rpm"] = 60


def test_e9_schema_versioning_flat_and_bm25():
    tmp_path = Path.cwd() / "data" / "test_runtime" / f"schema_case_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = str(tmp_path)

    ids_path = tmp_path / "ns" / "flat" / "ids.json"
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text(json.dumps({"schema_version": 99, "data": []}), encoding="utf-8")
    np.save(tmp_path / "ns" / "flat" / "vectors.npy", np.empty((0,), dtype=np.float32))
    with pytest.raises(SchemaVersionError):
        flat_store.load(base, "ns")

    bm25_dir = tmp_path / "bmns" / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    (bm25_dir / "index.json").write_text(
        json.dumps({"schema_version": 99, "data": {"ids": [], "doc_tokens": [], "df": {}, "avg_dl": 0.0}}),
        encoding="utf-8",
    )
    with pytest.raises(SchemaVersionError):
        bm25_store.load(base, "bmns")
