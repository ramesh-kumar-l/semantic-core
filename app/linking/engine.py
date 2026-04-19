import logging
from typing import Any, Dict

from ..graph.base import GraphDB
from .candidate import find_candidates
from .scorer import score

logger = logging.getLogger(__name__)

THRESHOLD = 0.5


class LinkingEngine:
    """
    Auto-creates graph edges between semantically related nodes during ingestion.

    Called once per ingest — finds candidates without full-graph scan, scores each,
    and writes edges for pairs that exceed THRESHOLD.
    """

    def __init__(self, threshold: float = THRESHOLD) -> None:
        self._threshold = threshold

    def link(self, node: Dict[str, Any], graph: GraphDB) -> int:
        """
        Link `node` to related existing nodes in the graph.
        Returns number of edges created.
        """
        candidates = find_candidates(node, graph)

        # Deduplicate across candidate groups (same node may appear in multiple groups)
        seen: Dict[str, Dict] = {}
        for group in candidates.values():
            for cand in group:
                if cand and "id" in cand:
                    seen[cand["id"]] = cand

        links_created = 0
        for cand_id, cand in seen.items():
            s = score(node, cand)
            if s >= self._threshold:
                relation = _determine_relation(node, cand)
                graph.add_edge(node["id"], relation, cand_id, weight=s)
                links_created += 1

        if links_created:
            logger.debug(
                "linking.engine node=%s links=%d", node.get("id"), links_created
            )
        return links_created


def _determine_relation(a: Dict, b: Dict) -> str:
    """Pick a semantically meaningful relation type based on node types."""
    a_type = a.get("type", "resource")
    b_type = b.get("type", "resource")

    if a_type in ("media", "image", "video", "screenshot") and b_type in (
        "call",
        "communication",
        "message",
        "email",
    ):
        return "related_to_call"

    if a_type in ("media", "image", "video", "screenshot") and b_type in (
        "event",
        "meeting",
        "birthday",
        "holiday",
    ):
        return "belongs_to_event"

    return "related_to"
