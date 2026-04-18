from typing import Dict, List, Tuple


class RankingService:
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        texts: Dict[str, str],
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError
