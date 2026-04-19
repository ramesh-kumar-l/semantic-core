import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.observability.logger import get_json_logger

logger = get_json_logger("learning.l2r_model")

try:
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class L2RModel:
    """
    Optional ML-based re-ranker using Logistic Regression.
    Falls back gracefully when scikit-learn is unavailable or model not trained.
    """

    def __init__(self) -> None:
        self._model = None
        self._available = _SKLEARN_AVAILABLE

    def train(self, data: List[Dict]) -> bool:
        """Train on list of feature dicts with 'is_clicked' label."""
        if not self._available:
            logger.warning("l2r_model_unavailable", extra={"reason": "scikit-learn not installed"})
            return False
        if len(data) < 10:
            logger.warning("l2r_model_sparse_data", extra={"samples": len(data)})
            return False

        feature_keys = ["vector_score", "bm25_score", "combined_score", "rank_position", "inv_rank"]
        X = np.array([[row.get(k, 0.0) for k in feature_keys] for row in data])
        y = np.array([int(row.get("is_clicked", 0)) for row in data])

        if len(set(y)) < 2:
            logger.warning("l2r_model_single_class", extra={"samples": len(data)})
            return False

        self._model = LogisticRegression(max_iter=500, C=1.0)
        self._model.fit(X, y)
        logger.info("l2r_model_trained", extra={"samples": len(data), "features": len(feature_keys)})
        return True

    def predict(self, features: Dict) -> float:
        """Return click probability in [0, 1]. Returns 0.0 if model not available."""
        if not self._available or self._model is None:
            return 0.0
        try:
            import numpy as np
            feature_keys = ["vector_score", "bm25_score", "combined_score", "rank_position", "inv_rank"]
            X = np.array([[features.get(k, 0.0) for k in feature_keys]])
            return float(self._model.predict_proba(X)[0][1])
        except Exception as exc:
            logger.warning("l2r_model_predict_error", extra={"error": str(exc)})
            return 0.0

    def save(self, path: str) -> bool:
        if self._model is None:
            return False
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("wb") as f:
                pickle.dump(self._model, f)
            logger.info("l2r_model_saved", extra={"path": path})
            return True
        except Exception as exc:
            logger.warning("l2r_model_save_error", extra={"error": str(exc)})
            return False

    def load(self, path: str) -> bool:
        if not self._available:
            return False
        try:
            p = Path(path)
            if not p.exists():
                logger.info("l2r_model_not_found", extra={"path": path})
                return False
            with p.open("rb") as f:
                self._model = pickle.load(f)
            logger.info("l2r_model_loaded", extra={"path": path})
            return True
        except Exception as exc:
            logger.warning("l2r_model_load_error", extra={"error": str(exc)})
            self._model = None
            return False

    @property
    def is_ready(self) -> bool:
        return self._available and self._model is not None

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        rule_boosts: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """Apply ML scores on top of existing candidate scores."""
        if not self.is_ready:
            return candidates

        rule_boosts = rule_boosts or {}
        results: List[Tuple[str, float]] = []

        for position, (doc_id, base_score) in enumerate(candidates):
            features = {
                "vector_score": base_score,
                "bm25_score": 0.0,
                "combined_score": base_score,
                "rank_position": float(position),
                "inv_rank": 1.0 / (position + 1),
            }
            model_score = self.predict(features)
            rule_boost = rule_boosts.get(doc_id, 0.0)
            final_score = base_score + rule_boost + model_score * 0.1

            logger.info(
                "l2r_applied",
                extra={
                    "event": "l2r_applied",
                    "query": query,
                    "doc_id": doc_id,
                    "base_score": round(base_score, 4),
                    "rule_boost": round(rule_boost, 4),
                    "model_score": round(model_score, 4),
                    "final_score": round(final_score, 4),
                },
            )
            results.append((doc_id, final_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
