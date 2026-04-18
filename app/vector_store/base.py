from typing import List, Tuple


class VectorStore:
    def add(self, id: str, vector: List[float]) -> None:
        raise NotImplementedError

    def search(self, vector: List[float], k: int) -> List[Tuple[str, float]]:
        raise NotImplementedError
