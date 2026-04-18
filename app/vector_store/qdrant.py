from typing import List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    ScoredPoint,
)
from .base import VectorStore


class QdrantStore(VectorStore):
    def __init__(self, url: str, collection: str, dim: int) -> None:
        self._client = QdrantClient(url=url)
        self._collection = collection
        self._dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )

    def add(self, id: str, vector: List[float]) -> None:
        # Qdrant requires integer or UUID point ids; hash string id to int
        point_id = abs(hash(id)) % (2**63)
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=point_id, vector=vector, payload={"str_id": id})],
        )

    def search(self, vector: List[float], k: int) -> List[Tuple[str, float]]:
        hits: List[ScoredPoint] = self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=k,
            with_payload=True,
        )
        return [(hit.payload["str_id"], hit.score) for hit in hits]
