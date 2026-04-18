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
    }
}
