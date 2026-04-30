import json
import os
import sqlite3
import threading
from typing import Dict, List, Optional

from .base import GraphDB


class SQLiteGraph(GraphDB):
    def __init__(self, path: str = "./data/graph.db"):
        self._path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS nodes (
                        id       TEXT PRIMARY KEY,
                        type     TEXT NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '{}'
                    );
                    CREATE TABLE IF NOT EXISTS edges (
                        source   TEXT NOT NULL,
                        relation TEXT NOT NULL,
                        target   TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_nodes_type     ON nodes(type);
                    CREATE INDEX IF NOT EXISTS idx_edges_source   ON edges(source);
                    CREATE INDEX IF NOT EXISTS idx_edges_target   ON edges(target);
                    CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique
                        ON edges(source, relation, target);
                """)
                # Migrate: add weight column if missing (safe on existing DBs)
                try:
                    conn.execute("ALTER TABLE edges ADD COLUMN weight REAL DEFAULT 1.0")
                except sqlite3.OperationalError:
                    pass  # column already exists

    # ------------------------------------------------------------------ writes

    def add_node(self, id: str, type: str, metadata: dict) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO nodes (id, type, metadata) VALUES (?, ?, ?)",
                    (id, type, json.dumps(metadata, default=str)),
                )

    def add_edge(self, source: str, relation: str, target: str, weight: float = 1.0) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO edges (source, relation, target, weight) VALUES (?, ?, ?, ?)",
                    (source, relation, target, weight),
                )

    # ------------------------------------------------------------------ reads

    def get_node(self, id: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE id = ?", (id,)).fetchone()
        if row:
            return {"id": row["id"], "type": row["type"], "metadata": json.loads(row["metadata"])}
        return None

    def get_neighbors(self, id: str, relation: str = None) -> List[Dict]:
        with self._conn() as conn:
            if relation:
                rows = conn.execute(
                    "SELECT target FROM edges WHERE source = ? AND relation = ?",
                    (id, relation),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT target FROM edges WHERE source = ?", (id,)
                ).fetchall()
        return [n for r in rows if (n := self.get_node(r["target"])) is not None]

    def get_predecessors(self, id: str, relation: str = None) -> List[Dict]:
        with self._conn() as conn:
            if relation:
                rows = conn.execute(
                    "SELECT source FROM edges WHERE target = ? AND relation = ?",
                    (id, relation),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT source FROM edges WHERE target = ?", (id,)
                ).fetchall()
        return [n for r in rows if (n := self.get_node(r["source"])) is not None]

    def query_nodes(self, filters: dict) -> List[Dict]:
        with self._conn() as conn:
            if "type" in filters:
                rows = conn.execute(
                    "SELECT * FROM nodes WHERE type = ?", (filters["type"],)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM nodes").fetchall()

        extra = {k: v for k, v in filters.items() if k != "type"}
        results = []
        for row in rows:
            meta = json.loads(row["metadata"])
            if all(meta.get(k) == v for k, v in extra.items()):
                results.append({"id": row["id"], "type": row["type"], "metadata": meta})
        return results

    def delete_node(self, id: str) -> bool:
        with self._lock:
            with self._conn() as conn:
                conn.execute("DELETE FROM edges WHERE source = ? OR target = ?", (id, id))
                cur = conn.execute("DELETE FROM nodes WHERE id = ?", (id,))
                return cur.rowcount > 0

    def traverse(self, start_ids: List[str], relation: str, depth: int = 1) -> List[Dict]:
        visited: set = set(start_ids)
        frontier = list(start_ids)
        collected: List[Dict] = []

        for _ in range(depth):
            next_frontier = []
            for node_id in frontier:
                for neighbor in self.get_neighbors(node_id, relation or None):
                    if neighbor["id"] not in visited:
                        visited.add(neighbor["id"])
                        next_frontier.append(neighbor["id"])
                        collected.append(neighbor)
            frontier = next_frontier

        return collected
