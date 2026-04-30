from typing import Any, Dict, List, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
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

    def add(
        self,
        id: str,
        vector: List[float],
        text: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Qdrant requires integer or UUID point ids; hash string id to int
        point_id = abs(hash(id)) % (2**63)
        payload: Dict[str, Any] = {"str_id": id, "text": text, "metadata": metadata or {}}
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    def get_texts(self) -> dict:
        results, _ = self._client.scroll(
            collection_name=self._collection,
            with_payload=True,
            limit=10000,
        )
        return {p.payload["str_id"]: p.payload.get("text", "") for p in results}

    def search(
        self,
        vector: List[float],
        k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        query_filter = None
        if filters:
            query_filter = Filter(
                must=[
                    FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                    for key, value in filters.items()
                ]
            )
        hits: List[ScoredPoint] = self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=k,
            with_payload=True,
            query_filter=query_filter,
        )
        return [(hit.payload["str_id"], hit.score) for hit in hits]

    def delete(self, id: str) -> bool:
        point_id = abs(hash(id)) % (2**63)
        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=[point_id]),
            wait=True,
        )
        return True
