from typing import Any, Dict, List, Optional

from ..graph.base import GraphDB


def build_filters(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a query-filters dict from an intent mapping."""
    allowed = {"type", "person", "location", "event", "activity", "source"}
    return {k: v for k, v in mapping.items() if k in allowed and v}


def graph_candidates(
    graph: GraphDB,
    entity_type: Optional[str] = None,
    person: Optional[str] = None,
    location: Optional[str] = None,
    event: Optional[str] = None,
) -> List[str]:
    """
    Resolve filter criteria via graph traversal and return candidate node IDs.

    Strategy:
    1. Find anchor nodes (person / location / event).
    2. Walk inbound edges to find media/objects that reference those anchors.
    3. Intersect across multiple filters.
    4. Fall back to type-scan when no anchors are given.
    """
    sets: List[set] = []

    if person:
        pid = f"person_{person.lower()}"
        if graph.get_node(pid):
            preds = graph.get_predecessors(pid, "contains")
            sets.append({n["id"] for n in preds})

    if location:
        lid = f"location_{location.lower()}"
        if graph.get_node(lid):
            preds = graph.get_predecessors(lid, "occurs_at")
            sets.append({n["id"] for n in preds})

    if event:
        eid = f"event_{event.lower().replace(' ', '_')}"
        if graph.get_node(eid):
            preds = graph.get_predecessors(eid, "happens_during")
            sets.append({n["id"] for n in preds})

    if sets:
        # Intersection: node must satisfy ALL supplied filters
        candidate_ids: set = sets[0].intersection(*sets[1:])
        # Optionally narrow by entity_type
        if entity_type:
            typed = {n["id"] for n in graph.query_nodes({"type": entity_type})}
            candidate_ids &= typed
        return list(candidate_ids)

    # No anchor filters — fall back to type-scan
    if entity_type:
        return [n["id"] for n in graph.query_nodes({"type": entity_type})]

    return []
