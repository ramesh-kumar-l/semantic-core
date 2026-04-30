from typing import Any, Dict, List, Optional, Tuple


class VectorStore:
    def add(self, id: str, vector: List[float], text: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    def search(
        self,
        vector: List[float],
        k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError

    def delete(self, id: str) -> bool:
        raise NotImplementedError
