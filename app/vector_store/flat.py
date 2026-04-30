from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .base import VectorStore


class FlatIndex(VectorStore):
    def __init__(self) -> None:
        self._ids: List[str] = []
        self._vectors: List[np.ndarray] = []
        self._texts: List[str] = []
        self._metadata: List[Dict[str, Any]] = []

    def add(
        self,
        id: str,
        vector: List[float],
        text: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        idx = self._index_of(id)
        if idx is not None:
            self._vectors[idx] = v
            self._texts[idx] = text
            self._metadata[idx] = dict(metadata or {})
            return
        self._ids.append(id)
        self._vectors.append(v)
        self._texts.append(text)
        self._metadata.append(dict(metadata or {}))

    def get_texts(self) -> dict:
        return dict(zip(self._ids, self._texts))

    def search(
        self,
        vector: List[float],
        k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        if not self._ids:
            return []

        q = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        matrix = np.stack(self._vectors)
        scores = matrix @ q
        top_k = min(k, len(self._ids))
        indices = np.argpartition(scores, -top_k)[-top_k:]
        indices = indices[np.argsort(scores[indices])[::-1]]

        results: List[Tuple[str, float]] = []
        for i in indices:
            if filters and not self._match_filters(self._metadata[i], filters):
                continue
            results.append((self._ids[i], float(scores[i])))
        return results

    def delete(self, id: str) -> bool:
        idx = self._index_of(id)
        if idx is None:
            return False
        self._ids.pop(idx)
        self._vectors.pop(idx)
        self._texts.pop(idx)
        self._metadata.pop(idx)
        return True

    def _index_of(self, id: str) -> Optional[int]:
        try:
            return self._ids.index(id)
        except ValueError:
            return None

    @staticmethod
    def _match_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        return all(metadata.get(k) == v for k, v in filters.items())
