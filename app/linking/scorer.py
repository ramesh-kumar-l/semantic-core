from typing import Any, Dict

TIME_WINDOW_SECS = 7200  # 2 hours


def score(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Return a [0, 1] similarity score between two graph nodes."""
    s = 0.0
    if _time_overlap(a, b):
        s += 0.4
    if _same_location(a, b):
        s += 0.3
    if _shared_people(a, b):
        s += 0.5
    return min(s, 1.0)


def _time_overlap(a: Dict, b: Dict) -> bool:
    ts_a = a.get("metadata", {}).get("timestamp")
    ts_b = b.get("metadata", {}).get("timestamp")
    if ts_a is None or ts_b is None:
        return False
    try:
        return abs(float(ts_a) - float(ts_b)) <= TIME_WINDOW_SECS
    except (TypeError, ValueError):
        return False


def _same_location(a: Dict, b: Dict) -> bool:
    loc_a = a.get("metadata", {}).get("location")
    loc_b = b.get("metadata", {}).get("location")
    return bool(loc_a and loc_b and str(loc_a).lower() == str(loc_b).lower())


def _shared_people(a: Dict, b: Dict) -> bool:
    def _people_set(node: Dict) -> set:
        p = node.get("metadata", {}).get("people", [])
        if isinstance(p, str):
            p = [p]
        return {str(x).lower() for x in p}

    return bool(_people_set(a) & _people_set(b))
