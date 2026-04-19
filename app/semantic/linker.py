import hashlib
import logging
from typing import Any, Dict, List

from ..graph.base import GraphDB
from .models import SemanticObject

logger = logging.getLogger(__name__)


class RelationLinker:
    """Builds graph edges from SemanticObject metadata."""

    def link(self, obj: SemanticObject, graph: GraphDB) -> None:
        self._link_people(obj.id, obj.metadata.get("people", []), graph)
        self._link_location(obj.id, obj.metadata.get("location"), graph)
        self._link_timestamp(obj.id, obj.metadata.get("timestamp"), graph)
        self._link_event(obj.id, obj.metadata.get("event"), graph)

        for rel in obj.relations:
            graph.add_edge(obj.id, rel.relation, rel.target_id)

    def link_raw(self, obj_id: str, obj_type: str, metadata: Dict[str, Any], graph: GraphDB) -> None:
        """Convenience wrapper — no SemanticObject required."""
        from .models import SemanticObject as SO
        self.link(SO(id=obj_id, type=obj_type, metadata=metadata), graph)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _link_people(obj_id: str, people: List[str], graph: GraphDB) -> None:
        for person in people:
            pid = f"person_{person.lower()}"
            graph.add_node(pid, "person", {"name": person})
            graph.add_edge(obj_id, "contains", pid)

    @staticmethod
    def _link_location(obj_id: str, location: Any, graph: GraphDB) -> None:
        if not location:
            return
        loc_id = f"location_{str(location).lower()}"
        graph.add_node(loc_id, "location", {"name": str(location)})
        graph.add_edge(obj_id, "occurs_at", loc_id)

    @staticmethod
    def _link_timestamp(obj_id: str, timestamp: Any, graph: GraphDB) -> None:
        if not timestamp:
            return
        time_id = f"time_{_short_hash(str(timestamp))}"
        graph.add_node(time_id, "time", {"value": str(timestamp)})
        graph.add_edge(obj_id, "captured_at", time_id)

    @staticmethod
    def _link_event(obj_id: str, event: Any, graph: GraphDB) -> None:
        if not event:
            return
        event_id = f"event_{str(event).lower().replace(' ', '_')}"
        graph.add_node(event_id, "event", {"name": str(event)})
        graph.add_edge(obj_id, "happens_during", event_id)


def _short_hash(val: str) -> str:
    return hashlib.md5(val.encode()).hexdigest()[:8]
