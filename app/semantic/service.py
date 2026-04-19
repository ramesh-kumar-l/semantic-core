import logging
from typing import Any, Dict, List, Optional

from ..graph.base import GraphDB
from .extractor import MetadataExtractor
from .filters import graph_candidates
from .linker import RelationLinker
from .models import SemanticObject, SemanticRelation
from .schema import resolve_entity_type

logger = logging.getLogger(__name__)


class SemanticService:
    def __init__(self, graph: GraphDB) -> None:
        self._graph = graph
        self._extractor = MetadataExtractor()
        self._linker = RelationLinker()

    # ------------------------------------------------------------------ ingest

    def ingest(
        self,
        id: str,
        input_data: Any,
        type_hint: Optional[str] = None,
        extra_relations: Optional[List[SemanticRelation]] = None,
    ) -> SemanticObject:
        """
        Extract metadata → create SemanticObject → persist node + edges in graph.

        Returns the created SemanticObject.
        """
        metadata = self._extractor.extract(input_data)

        if type_hint:
            metadata["type"] = resolve_entity_type(type_hint)

        obj_type = metadata.pop("type", "resource")
        obj = SemanticObject(
            id=id,
            type=obj_type,
            metadata=metadata,
            relations=extra_relations or [],
        )

        self._graph.add_node(id, obj_type, metadata)
        self._linker.link(obj, self._graph)

        logger.debug("semantic.ingest id=%s type=%s", id, obj_type)
        return obj

    # ------------------------------------------------------------------ query

    def query(
        self,
        intent: Dict[str, Any],
        pre_filtered_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Resolve an intent dict against the graph and return matching nodes.

        Intent keys (all optional):
            type, person, location, event
        """
        entity_type = intent.get("type")
        person = intent.get("person")
        location = intent.get("location")
        event = intent.get("event")

        ids = graph_candidates(
            self._graph,
            entity_type=entity_type,
            person=person,
            location=location,
            event=event,
        )

        if pre_filtered_ids is not None:
            allowed = set(pre_filtered_ids)
            ids = [i for i in ids if i in allowed] or ids

        results = []
        for node_id in ids:
            node = self._graph.get_node(node_id)
            if node:
                results.append(node)

        return results

    # ------------------------------------------------------------------ graph helpers

    def get_node(self, id: str) -> Optional[Dict]:
        return self._graph.get_node(id)

    def get_related(
        self,
        id: str,
        relation: Optional[str] = None,
        depth: int = 1,
        direction: str = "outbound",
    ) -> List[Dict]:
        """
        Retrieve related nodes.

        direction: "outbound" (default) | "inbound" | "both"
        """
        if direction == "inbound":
            return self._graph.get_predecessors(id, relation)
        if direction == "both":
            out = self._graph.get_neighbors(id, relation)
            inc = self._graph.get_predecessors(id, relation)
            seen: set = set()
            merged = []
            for n in out + inc:
                if n["id"] not in seen:
                    seen.add(n["id"])
                    merged.append(n)
            return merged
        if depth > 1:
            return self._graph.traverse([id], relation or "", depth)
        return self._graph.get_neighbors(id, relation)

    def link_nodes(self, source: str, relation: str, target: str) -> None:
        self._graph.add_edge(source, relation, target)
