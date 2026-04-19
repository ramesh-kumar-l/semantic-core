from typing import Any, Dict, List

from ..graph.base import GraphDB

MAX_TIME_CANDIDATES = 100
TIME_WINDOW_SECS = 7200  # 2 hours


def find_candidates(node: Dict[str, Any], graph: GraphDB) -> Dict[str, List[Dict]]:
    """
    Return candidate nodes grouped by signal without scanning the full graph.

    Strategies:
    - location_near: anchor traversal via occurs_at edge (O(predecessors))
    - same_person:   anchor traversal via contains edge for each person (O(P * predecessors))
    - time_near:     bounded type scan filtered in Python (O(type_count) ≤ MAX_TIME_CANDIDATES)
    """
    meta = node.get("metadata", {})
    node_id = node["id"]

    result: Dict[str, List[Dict]] = {"time_near": [], "location_near": [], "same_person": []}

    # location_near — inbound traversal from location anchor
    location = meta.get("location")
    if location:
        lid = f"location_{location.lower()}"
        preds = graph.get_predecessors(lid, "occurs_at")
        result["location_near"] = [n for n in preds if n["id"] != node_id]

    # same_person — inbound traversal from each person anchor
    people = meta.get("people", [])
    if isinstance(people, str):
        people = [people]
    person_ids: set = set()
    for person in people:
        pid = f"person_{person.lower()}"
        preds = graph.get_predecessors(pid, "contains")
        for n in preds:
            if n["id"] != node_id:
                person_ids.add(n["id"])
    if person_ids:
        result["same_person"] = [
            n for nid in person_ids if (n := graph.get_node(nid)) is not None
        ]

    # time_near — bounded type scan with Python time filter
    timestamp = meta.get("timestamp")
    if timestamp is not None:
        node_type = node.get("type", "resource")
        typed_nodes = graph.query_nodes({"type": node_type})[:MAX_TIME_CANDIDATES]
        result["time_near"] = [
            n
            for n in typed_nodes
            if n["id"] != node_id
            and _within_window(timestamp, n.get("metadata", {}).get("timestamp"))
        ]

    return result


def _within_window(ts_a: Any, ts_b: Any) -> bool:
    if ts_a is None or ts_b is None:
        return False
    try:
        return abs(float(ts_a) - float(ts_b)) <= TIME_WINDOW_SECS
    except (TypeError, ValueError):
        return False
