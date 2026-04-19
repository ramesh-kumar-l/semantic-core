from app.planner.base import QueryPlanner
from app.intelligence.analyzer import QueryAnalyzer
from app.core.config import config


class RuleBasedPlanner(QueryPlanner):
    def __init__(self) -> None:
        self._analyzer = QueryAnalyzer()

    def plan(self, query: str, namespace: str, context: dict) -> dict:
        analysis = context.get("analysis") or self._analyzer.analyze(query)
        hybrid_cfg = config.get("hybrid", {})
        query_type = analysis.get("type", "hybrid")

        if query_type == "keyword":
            alpha = 0.3
            vector_k = 10
            bm25_k = 30
        elif query_type == "semantic":
            alpha = 0.8
            vector_k = 30
            bm25_k = 10
        else:
            alpha = hybrid_cfg.get("alpha", 0.6)
            vector_k = hybrid_cfg.get("vector_k", 20)
            bm25_k = hybrid_cfg.get("bm25_k", 20)

        return {
            "alpha": alpha,
            "vector_k": vector_k,
            "bm25_k": bm25_k,
            "use_hybrid": hybrid_cfg.get("enabled", False),
            "rerank": config.get("ranking", {}).get("enabled", False),
        }
