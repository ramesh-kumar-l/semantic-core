#!/usr/bin/env python3
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.graph.neo4j_graph import Neo4jGraph


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main() -> int:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "testpassword")

    graph = None
    for attempt in range(10):
        graph = Neo4jGraph(uri=uri, user=user, password=password)
        if graph._ready:
            break
        time.sleep(2)
    if graph is None or not graph._ready:
        print("FAIL: Neo4j did not become ready after retries")
        return 1

    graph.add_node("n1", "person", {"name": "Alice"})
    graph.add_node("n2", "event", {"title": "Summit"})
    graph.add_node("n3", "location", {"city": "London"})
    graph.add_edge("n1", "attended", "n2")
    graph.add_edge("n2", "in_city", "n3")

    _require(graph.get_node("n1") is not None, "node retrieval works")
    _require(len(graph.get_neighbors("n1", relation="attended")) == 1, "relation traversal outbound works")
    _require(len(graph.get_predecessors("n3", relation="in_city")) == 1, "relation traversal inbound works")
    _require(len(graph.traverse(["n1"], relation="attended", depth=2)) >= 1, "bounded traversal works")
    _require(graph.delete_node("n2"), "delete node works")
    _require(graph.get_node("n2") is None, "deleted node no longer exists")
    print("PASS: Neo4j validation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
