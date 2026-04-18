from .base import VectorStore
from .flat import FlatIndex

__all__ = ["VectorStore", "FlatIndex", "QdrantStore"]


def __getattr__(name: str):
    if name == "QdrantStore":
        from .qdrant import QdrantStore
        return QdrantStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
