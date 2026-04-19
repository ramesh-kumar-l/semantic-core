from app.core.config import config


class RetrievalStrategy:
    def plan(self, analysis: dict) -> dict:
        hybrid_cfg = config.get("hybrid", {})
        default_alpha = hybrid_cfg.get("alpha", 0.7)
        default_vector_k = hybrid_cfg.get("vector_k", 20)
        default_bm25_k = hybrid_cfg.get("bm25_k", 20)

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
            alpha = default_alpha
            vector_k = default_vector_k
            bm25_k = default_bm25_k

        use_hybrid = hybrid_cfg.get("enabled", False)

        return {
            "alpha": alpha,
            "vector_k": vector_k,
            "bm25_k": bm25_k,
            "use_hybrid": use_hybrid,
        }
