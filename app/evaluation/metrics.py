from typing import List, Tuple


def recall_at_k(results: List[Tuple[str, float]], expected_ids: List[str], k: int) -> float:
    if not expected_ids:
        return 1.0
    retrieved = {id_ for id_, _ in results[:k]}
    hits = sum(1 for eid in expected_ids if eid in retrieved)
    return hits / len(expected_ids)


def average_latency(latencies: List[float]) -> float:
    if not latencies:
        return 0.0
    return sum(latencies) / len(latencies)
