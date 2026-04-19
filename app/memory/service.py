import logging
from typing import Any, Dict, List, Tuple

from app.core.config import config
from app.core.factory import get_vector_store
from app.hybrid.bm25 import BM25Index
from app.hybrid.fusion import fuse_results
from app.ranking.simple import SimpleRankingService
from app.services.embedding import EmbeddingService
from app.vector_store.base import VectorStore
from app.vector_store.flat import FlatIndex

from . import namespace as ns_mod
from .namespace import NamespaceManager

logger = logging.getLogger(__name__)


class MemoryService:
    def __init__(self) -> None:
        self._ns_mgr = NamespaceManager()
        self._stores: Dict[str, FlatIndex] = {}
        self._bm25s: Dict[str, BM25Index] = {}
        self._embedder = EmbeddingService()
        self._ranker = SimpleRankingService()
        self._persistence_cfg: Dict[str, Any] = config.get("persistence", {})
        self._hybrid_cfg: Dict[str, Any] = config.get("hybrid", {})
        self._ranking_cfg: Dict[str, Any] = config.get("ranking", {})

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add(self, namespace: str, id: str, text: str) -> None:
        namespace = ns_mod.resolve(namespace)
        ns_mod.validate(namespace)
        self._ensure_loaded(namespace)

        vector = self._embedder.embed(text)
        self._stores[namespace].add(id, vector, text)
        self._bm25s[namespace].add(id, text)

        self._persist(namespace)

    def search(self, namespace: str, query: str, k: int) -> List[Tuple[str, float]]:
        namespace = ns_mod.resolve(namespace)
        ns_mod.validate(namespace)
        self._ensure_loaded(namespace)

        vector = self._embedder.embed(query)
        store = self._stores[namespace]
        bm25 = self._bm25s[namespace]

        hybrid_cfg = self._hybrid_cfg
        ranking_cfg = self._ranking_cfg

        if hybrid_cfg.get("enabled"):
            vector_k = hybrid_cfg.get("vector_k", 20)
            bm25_k = hybrid_cfg.get("bm25_k", 20)
            hits = fuse_results(
                store.search(vector, vector_k),
                bm25.search(query, bm25_k),
                alpha=hybrid_cfg.get("alpha", 0.7),
            )
        else:
            hits = store.search(vector, k)

        if ranking_cfg.get("enabled") and hits:
            hits = self._ranker.rerank(query, hits, store.get_texts())

        return hits[:k]

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self, namespace: str) -> None:
        if self._ns_mgr.is_loaded(namespace):
            return

        if self._persistence_cfg.get("enabled", True):
            self._load_from_disk(namespace)
        else:
            self._init_empty(namespace)

        self._ns_mgr.mark_loaded(namespace)

    def _load_from_disk(self, namespace: str) -> None:
        from app.persistence import flat_store, bm25_store

        base = self._persistence_cfg.get("base_path", "./data")

        vs_cfg = config.get("vector_store", {})
        if vs_cfg.get("type") == "qdrant":
            self._stores[namespace] = self._make_store(namespace)
        else:
            flat_data = flat_store.load(base, namespace)
            if flat_data:
                vectors, ids, texts = flat_data
                store = FlatIndex()
                for id_, vec, text in zip(ids, vectors, texts):
                    store._ids.append(id_)
                    store._vectors.append(vec)
                    store._texts.append(text)
                self._stores[namespace] = store
                logger.info("memory.load namespace=%s docs=%d", namespace, len(ids))
            else:
                self._stores[namespace] = FlatIndex()

        bm25 = bm25_store.load(base, namespace)
        self._bm25s[namespace] = bm25 if bm25 is not None else BM25Index()

    def _make_store(self, namespace: str) -> VectorStore:
        vs_cfg = config.get("vector_store", {})
        if vs_cfg.get("type") == "qdrant":
            base_collection = vs_cfg.get("collection", "content")
            ns_cfg = {
                "vector_store": {
                    **vs_cfg,
                    "collection": f"{base_collection}_{namespace}",
                },
            }
            return get_vector_store(ns_cfg)
        return FlatIndex()

    def _init_empty(self, namespace: str) -> None:
        self._stores[namespace] = self._make_store(namespace)
        self._bm25s[namespace] = BM25Index()

    def _persist(self, namespace: str) -> None:
        if not self._persistence_cfg.get("enabled", True):
            return

        from app.persistence import flat_store, bm25_store

        base = self._persistence_cfg.get("base_path", "./data")
        vs_cfg = config.get("vector_store", {})
        store = self._stores[namespace]
        try:
            if vs_cfg.get("type") != "qdrant":
                flat_store.save(base, namespace, store._vectors, store._ids, store._texts)
            bm25_store.save(base, namespace, self._bm25s[namespace])
        except Exception as exc:
            logger.error("memory.persist failed namespace=%s: %s", namespace, exc)
