import os
from typing import Any, Dict

# Config is read from environment variables with sensible defaults.
# Override VECTOR_STORE_TYPE=qdrant to switch to Qdrant.

config: Dict[str, Any] = {
    "vector_store": {
        "type": os.getenv("VECTOR_STORE_TYPE", "flat"),
        "url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "collection": os.getenv("QDRANT_COLLECTION", "content"),
        "dim": int(os.getenv("VECTOR_DIM", "384")),
    },
    "ranking": {
        "enabled": os.getenv("RANKING_ENABLED", "false").lower() == "true",
    },
    "hybrid": {
        "enabled": os.getenv("HYBRID_ENABLED", "false").lower() == "true",
        "alpha": float(os.getenv("HYBRID_ALPHA", "0.7")),
        "bm25_k": int(os.getenv("HYBRID_BM25_K", "20")),
        "vector_k": int(os.getenv("HYBRID_VECTOR_K", "20")),
    },
    "memory": {
        "enabled": os.getenv("MEMORY_ENABLED", "true").lower() == "true",
        "default_namespace": os.getenv("MEMORY_DEFAULT_NAMESPACE", "default"),
        "lazy_load": os.getenv("MEMORY_LAZY_LOAD", "true").lower() == "true",
    },
    "persistence": {
        "enabled": os.getenv("PERSISTENCE_ENABLED", "true").lower() == "true",
        "base_path": os.getenv("PERSISTENCE_BASE_PATH", "./data"),
    },
    "intelligence": {
        "enabled": os.getenv("INTELLIGENCE_ENABLED", "false").lower() == "true",
        "mode": os.getenv("INTELLIGENCE_MODE", "simple"),  # "simple" or "llm"
        "rewrite": os.getenv("INTELLIGENCE_REWRITE", "true").lower() == "true",
        "multi_query": os.getenv("INTELLIGENCE_MULTI_QUERY", "false").lower() == "true",
        "max_queries": int(os.getenv("INTELLIGENCE_MAX_QUERIES", "3")),
    },
}
