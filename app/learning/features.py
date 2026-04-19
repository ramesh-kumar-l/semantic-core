from typing import Dict


def extract_features(
    query: str,
    doc_id: str,
    scores: Dict[str, float],
    position: int,
    is_clicked: int = 0,
) -> Dict[str, float]:
    """Extract deterministic features for a (query, doc) pair."""
    return {
        "vector_score": float(scores.get("vector", 0.0)),
        "bm25_score": float(scores.get("bm25", 0.0)),
        "combined_score": float(scores.get("combined", scores.get("vector", 0.0))),
        "rank_position": float(position),
        "inv_rank": 1.0 / (position + 1),
        "is_clicked": float(is_clicked),
    }
