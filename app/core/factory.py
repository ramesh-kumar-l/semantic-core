from typing import Any, Dict
from app.vector_store.base import VectorStore


def get_vector_store(config: Dict[str, Any]) -> VectorStore:
    vs_cfg = config["vector_store"]
    store_type = vs_cfg["type"]

    if store_type == "flat":
        from app.vector_store.flat import FlatIndex
        return FlatIndex()

    if store_type == "qdrant":
        from app.vector_store.qdrant import QdrantStore
        return QdrantStore(
            url=vs_cfg["url"],
            collection=vs_cfg["collection"],
            dim=vs_cfg["dim"],
        )

    raise ValueError(f"Unknown vector_store type: {store_type!r}. Use 'flat' or 'qdrant'.")
