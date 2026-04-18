"""
Benchmark runner: compares FlatIndex vs QdrantStore on Recall@K and latency.

Usage:
    python -m app.evaluation.benchmark [dataset.json]

The dataset JSON format:
    {
      "k": 5,
      "corpus": [{"id": "...", "text": "..."}],
      "queries": [{"query": "...", "expected_ids": ["id1", ...]}]
    }
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.config import config
from app.core.factory import get_vector_store
from app.evaluation.metrics import average_latency, recall_at_k
from app.services.embedding import EmbeddingService

DEFAULT_DATASET = Path(__file__).parent / "benchmark_dataset.json"
LOW_SCORE_THRESHOLD = 0.3
HIGH_LATENCY_MS = 200


def _run_backend(
    backend_type: str,
    corpus: List[Dict[str, str]],
    queries: List[Dict[str, Any]],
    k: int,
) -> None:
    cfg = {"vector_store": {**config["vector_store"], "type": backend_type}}
    try:
        store = get_vector_store(cfg)
    except Exception as exc:
        print(f"  [skip] {backend_type}: {exc}")
        return

    embedder = EmbeddingService()

    for item in corpus:
        store.add(item["id"], embedder.embed(item["text"]))

    recalls: List[float] = []
    latencies: List[float] = []

    for case in queries:
        vec = embedder.embed(case["query"])
        t0 = time.perf_counter()
        hits = store.search(vec, k)
        latency_ms = (time.perf_counter() - t0) * 1000

        latencies.append(latency_ms)
        recalls.append(recall_at_k(hits, case["expected_ids"], k))

        if not hits:
            print(f"  [warn] empty_results  query={case['query']!r}")
        elif hits[0][1] < LOW_SCORE_THRESHOLD:
            print(f"  [warn] low_score={hits[0][1]:.3f}  query={case['query']!r}")
        if latency_ms > HIGH_LATENCY_MS:
            print(f"  [warn] high_latency={latency_ms:.1f}ms  query={case['query']!r}")

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_lat = average_latency(latencies)

    print(f"\nBackend: {backend_type}")
    print(f"Recall@{k}: {avg_recall:.2f}")
    print(f"Avg Latency: {avg_lat:.1f}ms")


def main() -> None:
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_path) as f:
        data = json.load(f)

    corpus = data["corpus"]
    queries = data["queries"]
    k = data.get("k", 5)

    print(f"Dataset: {len(corpus)} docs, {len(queries)} queries, k={k}")
    print("=" * 40)

    for backend in ["flat", "qdrant"]:
        _run_backend(backend, corpus, queries, k)

    print()


if __name__ == "__main__":
    main()
