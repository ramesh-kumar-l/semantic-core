import logging
from typing import Dict, List, Optional

from .base import GraphDB

logger = logging.getLogger(__name__)


class Neo4jGraph(GraphDB):
    def __init__(self, uri: str, user: str, password: str):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None
        self._try_connect()

    def _try_connect(self) -> None:
        try:
            from neo4j import GraphDatabase  # noqa: PLC0415
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            self._driver.verify_connectivity()
            logger.info("Neo4j connected: %s", self._uri)
        except ImportError:
            logger.warning("neo4j package not installed — pip install neo4j")
        except Exception as exc:
            logger.warning("Neo4j unavailable (%s). Use SQLiteGraph as fallback.", exc)
            self._driver = None

    @property
    def _ready(self) -> bool:
        return self._driver is not None

    def _session(self):
        if not self._ready:
            raise RuntimeError("Neo4j driver not available")
        return self._driver.session()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _row_to_node(node) -> Dict:
        props = dict(node.items())
        node_id = props.pop("id", "")
        label = list(node.labels)[0].lower() if node.labels else "resource"
        return {"id": node_id, "type": label, "metadata": props}

    # ------------------------------------------------------------------ writes

    def add_node(self, id: str, type: str, metadata: dict) -> None:
        if not self._ready:
            return
        try:
            label = type.capitalize()
            props = {"id": id, **metadata}
            with self._session() as s:
                s.run(f"MERGE (n:{label} {{id: $id}}) SET n += $props", id=id, props=props)
        except Exception as exc:
            logger.error("Neo4j add_node: %s", exc)

    def add_edge(self, source: str, relation: str, target: str) -> None:
        if not self._ready:
            return
        try:
            rel = relation.upper()
            with self._session() as s:
                s.run(
                    f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) MERGE (a)-[:{rel}]->(b)",
                    src=source,
                    tgt=target,
                )
        except Exception as exc:
            logger.error("Neo4j add_edge: %s", exc)

    # ------------------------------------------------------------------ reads

    def get_node(self, id: str) -> Optional[Dict]:
        if not self._ready:
            return None
        try:
            with self._session() as s:
                rec = s.run("MATCH (n {id: $id}) RETURN n LIMIT 1", id=id).single()
                if rec:
                    return self._row_to_node(rec["n"])
        except Exception as exc:
            logger.error("Neo4j get_node: %s", exc)
        return None

    def get_neighbors(self, id: str, relation: str = None) -> List[Dict]:
        if not self._ready:
            return []
        try:
            with self._session() as s:
                if relation:
                    query = f"MATCH (a {{id: $id}})-[:{relation.upper()}]->(b) RETURN b"
                else:
                    query = "MATCH (a {id: $id})-->(b) RETURN b"
                return [self._row_to_node(r["b"]) for r in s.run(query, id=id)]
        except Exception as exc:
            logger.error("Neo4j get_neighbors: %s", exc)
        return []

    def get_predecessors(self, id: str, relation: str = None) -> List[Dict]:
        if not self._ready:
            return []
        try:
            with self._session() as s:
                if relation:
                    query = f"MATCH (a)-[:{relation.upper()}]->(b {{id: $id}}) RETURN a"
                else:
                    query = "MATCH (a)-->(b {id: $id}) RETURN a"
                return [self._row_to_node(r["a"]) for r in s.run(query, id=id)]
        except Exception as exc:
            logger.error("Neo4j get_predecessors: %s", exc)
        return []

    def query_nodes(self, filters: dict) -> List[Dict]:
        if not self._ready:
            return []
        try:
            label = filters.get("type", "").capitalize()
            with self._session() as s:
                query = f"MATCH (n:{label}) RETURN n" if label else "MATCH (n) RETURN n"
                results = [self._row_to_node(r["n"]) for r in s.run(query)]
            extra = {k: v for k, v in filters.items() if k != "type"}
            if extra:
                results = [r for r in results if all(r["metadata"].get(k) == v for k, v in extra.items())]
            return results
        except Exception as exc:
            logger.error("Neo4j query_nodes: %s", exc)
        return []

    def traverse(self, start_ids: List[str], relation: str, depth: int = 1) -> List[Dict]:
        if not self._ready:
            return []
        rel = relation.upper()
        seen: set = set(start_ids)
        results: List[Dict] = []
        try:
            with self._session() as s:
                for start_id in start_ids:
                    query = (
                        f"MATCH (a {{id: $id}})-[:{rel}*1..{depth}]->(b) "
                        "RETURN DISTINCT b"
                    )
                    for r in s.run(query, id=start_id):
                        node = self._row_to_node(r["b"])
                        if node["id"] not in seen:
                            seen.add(node["id"])
                            results.append(node)
        except Exception as exc:
            logger.error("Neo4j traverse: %s", exc)
        return results

    def close(self) -> None:
        if self._driver:
            self._driver.close()
