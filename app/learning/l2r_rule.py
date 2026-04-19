import threading
from collections import defaultdict
from typing import Dict, List, Tuple

from app.observability.logger import get_json_logger

logger = get_json_logger("learning.l2r_rule")

_LAMBDA = 0.2  # click boost weight


class RuleBasedL2R:
    """
    Boosts documents that were previously clicked for similar queries.
    Namespace-aware, in-memory, zero external dependencies.
    """

    def __init__(self) -> None:
        # {namespace: {query: {doc_id: click_count}}}
        self._clicks: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self._lock = threading.Lock()

    def record_click(self, namespace: str, query: str, doc_id: str) -> None:
        with self._lock:
            self._clicks[namespace][query][doc_id] += 1

    def load_from_feedback(self, events: list) -> None:
        """Bootstrap click statistics from historical feedback events."""
        with self._lock:
            for ev in events:
                ns = ev.get("namespace", "default")
                q = ev.get("query", "")
                clicked = ev.get("clicked", "")
                if q and clicked:
                    self._clicks[ns][q][clicked] += 1

    def rerank(
        self,
        query: str,
        namespace: str,
        candidates: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        if not candidates:
            return candidates

        with self._lock:
            ns_clicks = self._clicks.get(namespace, {})
            query_clicks = ns_clicks.get(query, {})

        if not query_clicks:
            return candidates

        total = sum(query_clicks.values()) or 1

        results: List[Tuple[str, float]] = []
        for doc_id, base_score in candidates:
            click_count = query_clicks.get(doc_id, 0)
            click_boost = _LAMBDA * (click_count / total)
            final_score = base_score + click_boost

            if click_count > 0:
                logger.info(
                    "l2r_applied",
                    extra={
                        "event": "l2r_applied",
                        "query": query,
                        "doc_id": doc_id,
                        "base_score": round(base_score, 4),
                        "rule_boost": round(click_boost, 4),
                        "model_score": 0.0,
                        "final_score": round(final_score, 4),
                    },
                )

            results.append((doc_id, final_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
