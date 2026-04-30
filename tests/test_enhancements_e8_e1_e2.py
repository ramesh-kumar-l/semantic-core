def _ingest(client, text: str, namespace: str, metadata: dict | None = None) -> str:
    payload = {"text": text, "namespace": namespace}
    if metadata is not None:
        payload["metadata"] = metadata
    resp = client.post("/content", json=payload)
    assert resp.status_code == 200
    return resp.json()["content_id"]


def _search_ids(client, query: str, namespace: str, filters: dict | None = None) -> list[str]:
    payload = {"query": query, "k": 10, "namespace": namespace}
    if filters is not None:
        payload["filters"] = filters
    resp = client.post("/similar", json=payload)
    assert resp.status_code == 200
    return [item["id"] for item in resp.json()["results"]]


def test_e8_admin_health_schema(client):
    resp = client.get("/admin/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "feature_flags" in body
    assert "namespaces" in body
    assert "doc_counts" in body
    assert "graph_node_count" in body
    assert "feedback_count" in body


def test_e1_metadata_filter_matching(client):
    namespace = "e1_filter_match"
    matching_id = _ingest(
        client,
        "A retrieval article about ranking and relevance",
        namespace,
        metadata={"category": "news"},
    )
    _ingest(
        client,
        "A travel note unrelated to ranking systems",
        namespace,
        metadata={"category": "travel"},
    )

    ids = _search_ids(client, "ranking relevance", namespace, filters={"category": "news"})
    assert matching_id in ids


def test_e1_metadata_filter_non_matching_returns_empty(client):
    namespace = "e1_filter_miss"
    _ingest(
        client,
        "A document tagged as pets",
        namespace,
        metadata={"category": "pets"},
    )
    ids = _search_ids(client, "document tagged pets", namespace, filters={"category": "finance"})
    assert ids == []


def test_e2_delete_then_search(client):
    namespace = "e2_delete"
    content_id = _ingest(client, "delete target semantic text", namespace, metadata={"category": "ops"})
    before_ids = _search_ids(client, "delete target semantic text", namespace)
    assert content_id in before_ids

    delete_resp = client.delete(f"/content/{content_id}?namespace={namespace}")
    assert delete_resp.status_code == 204

    after_ids = _search_ids(client, "delete target semantic text", namespace)
    assert content_id not in after_ids


def test_e2_update_then_search(client):
    namespace = "e2_update"
    content_id = _ingest(client, "old phrase unique for update", namespace, metadata={"version": "v1"})

    update_resp = client.put(
        f"/content/{content_id}",
        json={"text": "new phrase unique after update", "namespace": namespace, "metadata": {"version": "v2"}},
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["content_id"] == content_id

    old_version_ids = _search_ids(
        client,
        "new phrase unique after update",
        namespace,
        filters={"version": "v1"},
    )
    new_version_ids = _search_ids(
        client,
        "new phrase unique after update",
        namespace,
        filters={"version": "v2"},
    )
    assert content_id not in old_version_ids
    assert content_id in new_version_ids
