import uuid
from fastapi import APIRouter, Request
from app.schemas.requests import IngestRequest, SearchRequest
from app.schemas.responses import IngestResponse, SearchResponse, SearchResult
from app.services.embedding import EmbeddingService
from app.observability.logger import get_json_logger
from app.observability.timing import Timer

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
    request.app.state.store.add(content_id, vector)
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

    embed_timer = Timer().start()
    vector = _embedder.embed(body.query)
    embedding_ms = embed_timer.stop()

    search_timer = Timer().start()
    hits = request.app.state.store.search(vector, body.k)
    search_ms = search_timer.stop()

    total_ms = embedding_ms + search_ms

    logger.info(
        "search",
        extra={
            "request_id": request_id,
            "endpoint": "/similar",
            "backend": backend,
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
