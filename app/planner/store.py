import json
import hashlib
import threading
from pathlib import Path
from typing import Optional


class PlannerStore:
    def __init__(self, path: str = "./data/planner_store.json") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._data, f)

    @staticmethod
    def hash_query(query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def get(self, query_hash: str) -> Optional[dict]:
        return self._data.get(query_hash)

    def update(self, query_hash: str, strategy: dict, metrics: dict) -> None:
        with self._lock:
            existing = self._data.get(query_hash)
            if existing is None:
                entry = {
                    "query_hash": query_hash,
                    "strategy": strategy,
                    "metrics": {"ctr": metrics.get("ctr", 0.0), "mrr": metrics.get("mrr", 0.0), "count": 1},
                }
            else:
                old = existing["metrics"]
                count = old["count"] + 1
                entry = {
                    "query_hash": query_hash,
                    "strategy": strategy,
                    "metrics": {
                        "ctr": (old["ctr"] * old["count"] + metrics.get("ctr", 0.0)) / count,
                        "mrr": (old["mrr"] * old["count"] + metrics.get("mrr", 0.0)) / count,
                        "count": count,
                    },
                }
            self._data[query_hash] = entry
            self._save()
