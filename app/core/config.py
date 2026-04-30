import os
from pathlib import Path
from typing import Any, Dict


def _load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return

    try:
        with env_path.open("r", encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key or key in os.environ:
                    continue
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                os.environ[key] = value
    except OSError:
        # Ignore unreadable .env files and continue with existing environment vars.
        return


# Load .env values before reading environment settings.
_load_dotenv()

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
    "planner": {
        "enabled": os.getenv("PLANNER_ENABLED", "false").lower() == "true",
        "mode": os.getenv("PLANNER_MODE", "rule_based"),  # "rule_based" or "adaptive"
        "store_path": os.getenv("PLANNER_STORE_PATH", "./data/planner_store.json"),
    },
    "learning": {
        "enabled": os.getenv("LEARNING_ENABLED", "true").lower() == "true",
        "rule_based": os.getenv("LEARNING_RULE_BASED", "true").lower() == "true",
        "ml_model": os.getenv("LEARNING_ML_MODEL", "false").lower() == "true",
        "model_path": os.getenv("LEARNING_MODEL_PATH", "./models/l2r.pkl"),
        "feedback_path": os.getenv("LEARNING_FEEDBACK_PATH", "./data/feedback.jsonl"),
    },
    "graph": {
        "enabled": os.getenv("GRAPH_ENABLED", "true").lower() == "true",
        "type": os.getenv("GRAPH_TYPE", "sqlite"),          # "sqlite" | "neo4j"
        "path": os.getenv("GRAPH_PATH", "./data/graph.db"),
        "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
        "neo4j_password": os.getenv("NEO4J_PASSWORD", ""),
    },
    "linking": {
        "enabled": os.getenv("LINKING_ENABLED", "true").lower() == "true",
        "threshold": float(os.getenv("LINKING_THRESHOLD", "0.5")),
        "time_window_hours": int(os.getenv("LINKING_TIME_WINDOW_HOURS", "2")),
    },
}
