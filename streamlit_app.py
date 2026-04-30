import json
from typing import Any, Dict, Optional

import httpx
import streamlit as st


def _headers(api_key: str) -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["X-API-Key"] = api_key.strip()
    return headers


def _request(
    method: str,
    base_url: str,
    path: str,
    api_key: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        with httpx.Client(timeout=8.0) as client:
            response = client.request(
                method=method,
                url=url,
                headers=_headers(api_key),
                params=params,
                json=body,
            )
    except httpx.HTTPError:
        st.error(f"Service not reachable at {base_url}")
        return None

    content_type = response.headers.get("content-type", "")
    payload: Dict[str, Any] = {
        "status_code": response.status_code,
        "ok": response.is_success,
        "url": url,
        "data": None,
        "text": response.text,
    }
    if "application/json" in content_type.lower():
        try:
            payload["data"] = response.json()
        except ValueError:
            payload["data"] = None
    return payload


def _show_response(resp: Optional[Dict[str, Any]]) -> None:
    if resp is None:
        return
    if resp["ok"]:
        st.success(f"{resp['status_code']} {resp['url']}")
    else:
        st.error(f"{resp['status_code']} {resp['url']}")

    if resp["data"] is not None:
        st.json(resp["data"])
    elif resp["text"]:
        st.code(resp["text"], language="text")


st.set_page_config(page_title="Semantic Core Service Tester", layout="wide")
st.title("Semantic Core Service Testing Interface")

with st.sidebar:
    st.subheader("Connection")
    base_url = st.text_input("Base URL", value="http://localhost:8000")
    api_key = st.text_input("API Key", value="", type="password")
    st.caption("All calls go through HTTP endpoints via httpx.")

tabs = st.tabs(
    [
        "Ingest",
        "Search",
        "Feedback",
        "Semantic Ingest",
        "Semantic Query",
        "Graph Node Lookup",
        "Graph Query",
        "System Health",
    ]
)

with tabs[0]:
    st.subheader("POST /content")
    text = st.text_area("Text", value="Semantic retrieval systems are easier to debug with good observability.")
    namespace = st.text_input("Namespace", value="default", key="ingest_namespace")
    if st.button("Ingest Content", use_container_width=True):
        payload = {"text": text, "namespace": namespace or None}
        _show_response(_request("POST", base_url, "/content", api_key, body=payload))

with tabs[1]:
    st.subheader("POST /similar")
    query = st.text_input("Query", value="observability for retrieval systems")
    k = st.slider("k", min_value=1, max_value=20, value=5)
    namespace = st.text_input("Namespace", value="default", key="search_namespace")
    if st.button("Search Similar", use_container_width=True):
        payload = {"query": query, "k": k, "namespace": namespace or None}
        _show_response(_request("POST", base_url, "/similar", api_key, body=payload))

with tabs[2]:
    st.subheader("POST /feedback")
    query = st.text_input("Query", value="observability for retrieval systems", key="feedback_query")
    namespace = st.text_input("Namespace", value="default", key="feedback_namespace")
    results_raw = st.text_input("Results IDs (comma-separated)", value="")
    clicked = st.text_input("Clicked ID", value="")
    position = st.number_input("Position", min_value=0, value=0, step=1)
    timestamp = st.text_input("Timestamp (optional epoch seconds)", value="")

    if st.button("Send Feedback", use_container_width=True):
        results = [item.strip() for item in results_raw.split(",") if item.strip()]
        payload: Dict[str, Any] = {
            "query": query,
            "namespace": namespace or None,
            "results": results,
            "clicked": clicked,
            "position": int(position),
        }
        if timestamp.strip():
            payload["timestamp"] = int(timestamp.strip())
        _show_response(_request("POST", base_url, "/feedback", api_key, body=payload))

with tabs[3]:
    st.subheader("POST /semantic/ingest")
    node_id = st.text_input("ID (optional)", value="")
    node_type = st.text_input("Type (optional)", value="")
    text = st.text_area("Text (optional)", value="", key="semantic_ingest_text")
    metadata_raw = st.text_area("Metadata JSON", value='{"person": "Ada"}')
    namespace = st.text_input("Namespace (optional)", value="", key="semantic_ingest_namespace")

    if st.button("Semantic Ingest", use_container_width=True):
        try:
            metadata = json.loads(metadata_raw) if metadata_raw.strip() else {}
        except json.JSONDecodeError:
            st.error("Metadata must be valid JSON")
            metadata = None
        if metadata is not None:
            payload = {
                "id": node_id or None,
                "type": node_type or None,
                "text": text or None,
                "metadata": metadata,
                "namespace": namespace or None,
            }
            _show_response(_request("POST", base_url, "/semantic/ingest", api_key, body=payload))

with tabs[4]:
    st.subheader("POST /semantic/query")
    query = st.text_input("Query (optional)", value="", key="semantic_query_query")
    node_type = st.text_input("Type (optional)", value="", key="semantic_query_type")
    person = st.text_input("Person (optional)", value="")
    location = st.text_input("Location (optional)", value="")
    event = st.text_input("Event (optional)", value="")
    depth = st.slider("Depth", min_value=1, max_value=4, value=1)
    k = st.slider("k", min_value=1, max_value=25, value=10, key="semantic_query_k")
    namespace = st.text_input("Namespace (optional)", value="", key="semantic_query_namespace")
    if st.button("Semantic Query", use_container_width=True):
        payload = {
            "query": query or None,
            "type": node_type or None,
            "person": person or None,
            "location": location or None,
            "event": event or None,
            "depth": depth,
            "k": k,
            "namespace": namespace or None,
        }
        _show_response(_request("POST", base_url, "/semantic/query", api_key, body=payload))

with tabs[5]:
    st.subheader("GET /semantic/node/{node_id}")
    node_id = st.text_input("Node ID", value="")
    relation = st.text_input("Relation (optional)", value="")
    direction = st.selectbox("Direction", options=["outbound", "inbound", "both"], index=0)
    depth = st.slider("Depth", min_value=1, max_value=4, value=1, key="node_lookup_depth")
    if st.button("Lookup Node", use_container_width=True):
        if not node_id.strip():
            st.error("Node ID is required")
        else:
            params = {
                "relation": relation or None,
                "direction": direction,
                "depth": depth,
            }
            _show_response(_request("GET", base_url, f"/semantic/node/{node_id.strip()}", api_key, params=params))

with tabs[6]:
    st.subheader("POST /graph_query/execute")
    person = st.text_input("Person (optional)", value="", key="graph_query_person")
    location = st.text_input("Location (optional)", value="", key="graph_query_location")
    event = st.text_input("Event (optional)", value="", key="graph_query_event")
    time_val = st.text_input("Time (optional)", value="")
    node_type = st.text_input("Type (optional)", value="", key="graph_query_type")
    traversal_raw = st.text_input("Traversal steps (comma-separated, optional)", value="")
    k = st.slider("k", min_value=1, max_value=25, value=10, key="graph_query_k")
    fallback = st.checkbox("fallback_to_retrieval", value=True)
    if st.button("Execute Graph Query", use_container_width=True):
        traversal = [item.strip() for item in traversal_raw.split(",") if item.strip()] or None
        payload = {
            "person": person or None,
            "location": location or None,
            "event": event or None,
            "time": time_val or None,
            "type": node_type or None,
            "traversal": traversal,
            "k": k,
            "fallback_to_retrieval": fallback,
        }
        _show_response(_request("POST", base_url, "/graph_query/execute", api_key, body=payload))

with tabs[7]:
    st.subheader("GET /admin/health")
    if st.button("Refresh Health", use_container_width=True):
        resp = _request("GET", base_url, "/admin/health", api_key)
        if resp is None:
            pass
        elif not resp["ok"]:
            _show_response(resp)
            st.info("Health endpoint is unavailable in this server build.")
        elif not isinstance(resp["data"], dict):
            _show_response(resp)
        else:
            data = resp["data"]
            st.success("Health endpoint reachable")

            col1, col2, col3 = st.columns(3)
            col1.metric("Status", str(data.get("status", "unknown")))
            col2.metric("Uptime", str(data.get("uptime_s", "n/a")))
            col3.metric("Version", str(data.get("version", "n/a")))

            flags = data.get("feature_flags")
            if isinstance(flags, dict):
                st.write("Feature Flags")
                flag_cols = st.columns(max(1, min(4, len(flags))))
                for i, (k, v) in enumerate(flags.items()):
                    flag_cols[i % len(flag_cols)].metric(k, str(v))

            st.write("Raw Health Payload")
            st.json(data)
