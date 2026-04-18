from typing import List, Tuple
import numpy as np
from .base import VectorStore


class FlatIndex(VectorStore):
    def __init__(self) -> None:
        self._ids: List[str] = []
        self._vectors: List[np.ndarray] = []
        self._texts: List[str] = []

    def add(self, id: str, vector: List[float], text: str = "") -> None:
        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        self._ids.append(id)
        self._vectors.append(v)
        self._texts.append(text)

    def get_texts(self) -> dict:
        return dict(zip(self._ids, self._texts))

    def search(self, vector: List[float], k: int) -> List[Tuple[str, float]]:
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

        return [(self._ids[i], float(scores[i])) for i in indices]
