import logging
from typing import Any, Dict, List

from ..graph.base import GraphDB

logger = logging.getLogger(__name__)

MAX_NODES = 200
MAX_DEPTH = 2

# Media subtypes recognised during type-matching
_MEDIA_TYPES = frozenset({"media", "image", "video", "screenshot", "document"})
_COMM_TYPES = frozenset({"communication", "call", "message", "email"})
_EVENT_TYPES = frozenset({"event", "meeting", "birthday", "holiday"})

_TYPE_GROUPS: Dict[str, frozenset] = {
    "media": _MEDIA_TYPES,
    "communication": _COMM_TYPES,
    "event": _EVENT_TYPES,
}


class GraphQueryEngine:
    """
    Executes bounded graph traversal plans produced by GraphPlanner.

    Traversal is BFS, checking both outbound (get_neighbors) and inbound
    (get_predecessors) edges so that cross-domain related_to links are found
    regardless of edge direction.
    """

    def execute(self, plan: Dict[str, Any], graph: GraphDB) -> Dict[str, Any]:
        """
        Run the plan against the graph.

        Returns::
            {
                "node_ids":       list of result node IDs,
                "traversal_steps": [{"step": type, "count": N}, ...],
                "truncated":       bool — True if MAX_NODES was hit,
            }
        """
        start_nodes: List[str] = plan.get("start", [])
        traversal_steps: List[str] = plan.get("traversal", [])[:MAX_DEPTH]
        filters: Dict[str, Any] = plan.get("filters", {})

        if not start_nodes:
            return {"node_ids": [], "traversal_steps": [], "truncated": False}

        # Seed frontier with existing start nodes only
        visited: set = set()
        frontier: set = set()
        for nid in start_nodes:
            if graph.get_node(nid) is not None:
                frontier.add(nid)
                visited.add(nid)

        if not frontier:
            return {"node_ids": [], "traversal_steps": [], "truncated": False}

        steps_log: List[Dict] = []
        truncated = False

        for step_type in traversal_steps:
            if len(visited) >= MAX_NODES:
                truncated = True
                break

            next_frontier: set = set()

            for nid in frontier:
                neighbors = graph.get_neighbors(nid) + graph.get_predecessors(nid)
                for n in neighbors:
                    cid = n["id"]
                    if cid not in visited and _matches_type(n.get("type", "resource"), step_type):
                        next_frontier.add(cid)
                        visited.add(cid)
                        if len(visited) >= MAX_NODES:
                            truncated = True
                            break
                if truncated:
                    break

            steps_log.append({"step": step_type, "count": len(next_frontier)})
            frontier = next_frontier

            if not frontier:
                break

        result_ids = list(frontier)

        if filters:
            result_ids = _apply_filters(result_ids, filters, graph)

        logger.info(
            "graph_traversal",
            extra={
                "event": "graph_traversal",
                "start": start_nodes,
                "steps": steps_log,
                "result_count": len(result_ids),
                "truncated": truncated,
            },
        )

        return {
            "node_ids": result_ids,
            "traversal_steps": steps_log,
            "truncated": truncated,
        }


def _matches_type(node_type: str, step: str) -> bool:
    group = _TYPE_GROUPS.get(step)
    if group:
        return node_type in group
    return node_type == step


def _apply_filters(node_ids: List[str], filters: Dict[str, Any], graph: GraphDB) -> List[str]:
    """Post-traversal filter pass. Extensible for future filter types."""
    required_type = filters.get("type")
    if required_type:
        node_ids = [
            nid for nid in node_ids
            if _node_type(nid, graph) == required_type
        ]
    # Future: time range, location match, etc.
    return node_ids


def _node_type(node_id: str, graph: GraphDB) -> str:
    n = graph.get_node(node_id)
    return n.get("type", "resource") if n else "resource"
