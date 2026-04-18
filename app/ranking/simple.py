from typing import Dict, List, Tuple
from .base import RankingService

_KEYWORD_BOOST = 0.15


class SimpleRankingService(RankingService):
    """
    Normalizes vector scores then applies a keyword boost when query terms
    appear in the stored text.  Deterministic and dependency-free.
    """

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        texts: Dict[str, str],
    ) -> List[Tuple[str, float]]:
        if not candidates:
            return []

        query_terms = set(query.lower().split())

        scores = [score for _, score in candidates]
        min_s, max_s = min(scores), max(scores)
        span = max_s - min_s if max_s > min_s else 1.0

        results: List[Tuple[str, float]] = []
        for id_, score in candidates:
            normalized = (score - min_s) / span
            text = texts.get(id_, "")
            text_terms = set(text.lower().split())
            boost = _KEYWORD_BOOST if query_terms & text_terms else 0.0
            results.append((id_, normalized + boost))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
