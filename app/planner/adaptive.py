from app.planner.base import QueryPlanner
from app.planner.rule_based import RuleBasedPlanner
from app.planner.store import PlannerStore


class AdaptivePlanner(QueryPlanner):
    _ALPHA_MAX = 1.0
    _BM25_K_MAX = 50

    def __init__(self, store: PlannerStore) -> None:
        self._rule = RuleBasedPlanner()
        self._store = store

    def plan(self, query: str, namespace: str, context: dict) -> dict:
        plan = self._rule.plan(query, namespace, context)
        query_hash = PlannerStore.hash_query(query)
        history = self._store.get(query_hash)

        if history:
            metrics = history.get("metrics", {})
            ctr = metrics.get("ctr", 1.0)
            mrr = metrics.get("mrr", 1.0)
            if ctr < 0.3:
                plan["bm25_k"] = min(plan["bm25_k"] + 10, self._BM25_K_MAX)
            if mrr < 0.4:
                plan["alpha"] = min(plan["alpha"] + 0.1, self._ALPHA_MAX)

        return plan
