import json
import time
import threading
from pathlib import Path
from typing import List


class FeedbackStore:
    """Append-only JSONL store for user interaction signals."""

    def __init__(self, path: str = "./data/feedback.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._count: int = 0
        self._count_loaded: bool = False

    def add(self, event: dict) -> None:
        record = {
            "query": event.get("query", ""),
            "namespace": event.get("namespace", "default"),
            "results": event.get("results", []),
            "clicked": event.get("clicked", ""),
            "position": event.get("position", 0),
            "timestamp": event.get("timestamp", int(time.time())),
        }
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            if self._count_loaded:
                self._count += 1

    def load_all(self) -> List[dict]:
        if not self._path.exists():
            self._count = 0
            self._count_loaded = True
            return []
        records = []
        with self._lock:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        self._count = len(records)
        self._count_loaded = True
        return records

    def count(self) -> int:
        if self._count_loaded:
            return self._count
        _ = self.load_all()
        return self._count
