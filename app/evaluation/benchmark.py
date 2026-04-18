"""
Benchmark runner: compares retrieval pipeline modes on Recall@K, MRR, and latency.

Modes compared:
  1. Vector only
  2. Vector + Ranking
  3. Hybrid (BM25 + Vector fusion) + Ranking

Usage:
    python -m app.evaluation.benchmark [dataset.json]

Dataset JSON format:
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
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.config import config
from app.core.factory import get_vector_store
from app.evaluation.metrics import average_latency, recall_at_k
from app.hybrid.bm25 import BM25Index
from app.hybrid.fusion import fuse_results
from app.ranking.simple import SimpleRankingService
from app.services.embedding import EmbeddingService

DEFAULT_DATASET = Path(__file__).parent / "benchmark_dataset.json"
LOW_SCORE_THRESHOLD = 0.3
HIGH_LATENCY_MS = 200


def _mrr(hits: List[Tuple[str, float]], expected_ids: List[str]) -> float:
    for rank, (id_, _) in enumerate(hits, start=1):
        if id_ in expected_ids:
            return 1.0 / rank
    return 0.0


def _run_mode(
    label: str,
    corpus: List[Dict[str, str]],
    queries: List[Dict[str, Any]],
    k: int,
    use_hybrid: bool,
    use_ranking: bool,
    backend_type: str = "flat",
    alpha: float = 0.7,
    vector_k: int = 20,
    bm25_k: int = 20,
) -> None:
    cfg = {"vector_store": {**config["vector_store"], "type": backend_type}}
    try:
        store = get_vector_store(cfg)
    except Exception as exc:
        print(f"  [skip] {label}: {exc}")
        return

    embedder = EmbeddingService()
    bm25 = BM25Index()
    ranker = SimpleRankingService()

    for item in corpus:
        store.add(item["id"], embedder.embed(item["text"]), item["text"])
        bm25.add(item["id"], item["text"])

    recalls: List[float] = []
    mrrs: List[float] = []
    latencies: List[float] = []

    for case in queries:
        vec = embedder.embed(case["query"])
        t0 = time.perf_counter()

        if use_hybrid:
            vector_hits = store.search(vec, vector_k)
            bm25_hits = bm25.search(case["query"], bm25_k)
            hits = fuse_results(vector_hits, bm25_hits, alpha=alpha)
        else:
            hits = store.search(vec, vector_k)

        if use_ranking and hits:
            texts = store.get_texts()
            hits = ranker.rerank(case["query"], hits, texts)

        hits = hits[:k]
        latency_ms = (time.perf_counter() - t0) * 1000

        latencies.append(latency_ms)
        recalls.append(recall_at_k(hits, case["expected_ids"], k))
        mrrs.append(_mrr(hits, case["expected_ids"]))

        if not hits:
            print(f"  [warn] empty_results  query={case['query']!r}")
        elif hits[0][1] < LOW_SCORE_THRESHOLD:
            print(f"  [warn] low_score={hits[0][1]:.3f}  query={case['query']!r}")
        if latency_ms > HIGH_LATENCY_MS:
            print(f"  [warn] high_latency={latency_ms:.1f}ms  query={case['query']!r}")

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0
    avg_lat = average_latency(latencies)

    print(f"\nMode: {label}")
    print(f"  Recall@{k}:    {avg_recall:.3f}")
    print(f"  MRR:         {avg_mrr:.3f}")
    print(f"  Avg Latency: {avg_lat:.1f}ms")


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
    print("=" * 50)

    _run_mode("Vector only",          corpus, queries, k, use_hybrid=False, use_ranking=False)
    _run_mode("Vector + Ranking",     corpus, queries, k, use_hybrid=False, use_ranking=True)
    _run_mode("Hybrid + Ranking",     corpus, queries, k, use_hybrid=True,  use_ranking=True)

    print()


if __name__ == "__main__":
    main()
