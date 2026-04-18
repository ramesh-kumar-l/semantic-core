import uuid
from fastapi import APIRouter, Request
from app.schemas.requests import IngestRequest, SearchRequest
from app.schemas.responses import IngestResponse, SearchResponse, SearchResult
from app.services.embedding import EmbeddingService
from app.observability.logger import get_json_logger
from app.observability.timing import Timer
from app.core.config import config
from app.hybrid.fusion import fuse_results

logger = get_json_logger("api.routes")
router = APIRouter()

_embedder = EmbeddingService()

_LOW_SCORE = 0.3
_HIGH_LATENCY_MS = 200


@router.post("/content", response_model=IngestResponse)
def ingest_content(body: IngestRequest, request: Request) -> IngestResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    backend = type(request.app.state.store).__name__

    embed_timer = Timer().start()
    vector = _embedder.embed(body.text)
    embedding_ms = embed_timer.stop()

    content_id = str(uuid.uuid4())

    store_timer = Timer().start()
    request.app.state.store.add(content_id, vector, body.text)
    if config["hybrid"]["enabled"]:
        request.app.state.bm25.add(content_id, body.text)
    store_ms = store_timer.stop()

    total_ms = embedding_ms + store_ms

    logger.info(
        "ingest",
        extra={
            "request_id": request_id,
            "endpoint": "/content",
            "backend": backend,
            "embedding_ms": round(embedding_ms, 2),
            "store_ms": round(store_ms, 2),
            "total_ms": round(total_ms, 2),
            "content_id": content_id,
        },
    )

    if total_ms > _HIGH_LATENCY_MS:
        logger.warning(
            "retrieval_warning",
            extra={"event": "retrieval_warning", "type": "high_latency",
                   "latency_ms": round(total_ms, 2), "endpoint": "/content"},
        )

    return IngestResponse(content_id=content_id)


@router.post("/similar", response_model=SearchResponse)
def search_similar(body: SearchRequest, request: Request) -> SearchResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    backend = type(request.app.state.store).__name__
    hybrid_cfg = config["hybrid"]
    ranking_cfg = config["ranking"]

    embed_timer = Timer().start()
    vector = _embedder.embed(body.query)
    embedding_ms = embed_timer.stop()

    search_timer = Timer().start()

    if hybrid_cfg["enabled"]:
        vector_k = hybrid_cfg["vector_k"]
        bm25_k = hybrid_cfg["bm25_k"]
        vector_hits = request.app.state.store.search(vector, vector_k)
        bm25_hits = request.app.state.bm25.search(body.query, bm25_k)

        if not vector_hits and not bm25_hits:
            logger.warning(
                "retrieval_warning",
                extra={"event": "retrieval_warning", "type": "empty_fusion",
                       "query": body.query},
            )

        # Log disagreement: IDs in one result set but not the other
        vec_ids = {id_ for id_, _ in vector_hits}
        bm25_ids = {id_ for id_, _ in bm25_hits}
        if vec_ids and bm25_ids and not (vec_ids & bm25_ids):
            logger.warning(
                "retrieval_warning",
                extra={"event": "retrieval_warning", "type": "result_disagreement",
                       "query": body.query},
            )

        hits = fuse_results(vector_hits, bm25_hits, alpha=hybrid_cfg["alpha"])
    else:
        hits = request.app.state.store.search(vector, body.k)

    search_ms = search_timer.stop()

    if ranking_cfg["enabled"] and hits:
        texts = request.app.state.store.get_texts()

        # Log large score divergence before reranking
        scores = [s for _, s in hits]
        if scores and (max(scores) - min(scores)) > 0.8:
            logger.warning(
                "retrieval_warning",
                extra={"event": "retrieval_warning", "type": "large_score_divergence",
                       "query": body.query},
            )

        hits = request.app.state.ranker.rerank(body.query, hits, texts)

    hits = hits[: body.k]

    total_ms = embedding_ms + search_ms

    logger.info(
        "search",
        extra={
            "request_id": request_id,
            "endpoint": "/similar",
            "backend": backend,
            "hybrid": hybrid_cfg["enabled"],
            "ranking": ranking_cfg["enabled"],
            "embedding_ms": round(embedding_ms, 2),
            "search_ms": round(search_ms, 2),
            "total_ms": round(total_ms, 2),
            "k": body.k,
            "hits": len(hits),
        },
    )

    if not hits:
        logger.warning(
            "retrieval_warning",
            extra={"event": "retrieval_warning", "type": "empty_results",
                   "query": body.query},
        )
    elif hits[0][1] < _LOW_SCORE:
        logger.warning(
            "retrieval_warning",
            extra={"event": "retrieval_warning", "type": "low_score",
                   "score": round(hits[0][1], 4), "query": body.query},
        )

    if total_ms > _HIGH_LATENCY_MS:
        logger.warning(
            "retrieval_warning",
            extra={"event": "retrieval_warning", "type": "high_latency",
                   "latency_ms": round(total_ms, 2), "query": body.query},
        )

    return SearchResponse(results=[SearchResult(id=id_, score=score) for id_, score in hits])
