import logging

from .base import GraphDB
from .sqlite_graph import SQLiteGraph

logger = logging.getLogger(__name__)


def get_graph(config: dict) -> GraphDB:
    graph_cfg = config.get("graph", {})
    backend = graph_cfg.get("type", "sqlite")

    if backend == "neo4j":
        try:
            from .neo4j_graph import Neo4jGraph  # noqa: PLC0415
            graph = Neo4jGraph(
                uri=graph_cfg.get("neo4j_uri", "bolt://localhost:7687"),
                user=graph_cfg.get("neo4j_user", "neo4j"),
                password=graph_cfg.get("neo4j_password", ""),
            )
            logger.info("Graph backend: neo4j uri=%s", graph_cfg.get("neo4j_uri"))
            return graph
        except Exception as exc:
            logger.warning("Neo4j init failed (%s) — falling back to SQLite", exc)

    path = graph_cfg.get("path", "./data/graph.db")
    logger.info("Graph backend: sqlite path=%s", path)
    return SQLiteGraph(path=path)
