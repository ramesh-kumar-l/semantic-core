from typing import Dict, List, Tuple


def _normalize(results: List[Tuple[str, float]]) -> Dict[str, float]:
    if not results:
        return {}
    scores = [s for _, s in results]
    min_s, max_s = min(scores), max(scores)
    span = max_s - min_s if max_s > min_s else 1.0
    return {id_: (s - min_s) / span for id_, s in results}


def fuse_results(
    vector_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]],
    alpha: float = 0.7,
) -> List[Tuple[str, float]]:
    """
    Weighted sum fusion: final = alpha * vector_score + (1-alpha) * bm25_score.
    Both score lists are normalized to [0, 1] before fusion.
    """
    vec_norm = _normalize(vector_results)
    bm25_norm = _normalize(bm25_results)

    all_ids = set(vec_norm) | set(bm25_norm)
    fused: List[Tuple[str, float]] = []
    for id_ in all_ids:
        score = alpha * vec_norm.get(id_, 0.0) + (1 - alpha) * bm25_norm.get(id_, 0.0)
        fused.append((id_, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused
