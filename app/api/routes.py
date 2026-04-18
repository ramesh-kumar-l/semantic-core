import time
import uuid
import logging
from fastapi import APIRouter, Request
from app.schemas.requests import IngestRequest, SearchRequest
from app.schemas.responses import IngestResponse, SearchResponse, SearchResult
from app.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)
router = APIRouter()

# EmbeddingService is stateless; one instance is fine.
_embedder = EmbeddingService()


@router.post("/content", response_model=IngestResponse)
def ingest_content(body: IngestRequest, request: Request) -> IngestResponse:
    t0 = time.perf_counter()
    content_id = str(uuid.uuid4())
    vector = _embedder.embed(body.text)
    request.app.state.store.add(content_id, vector)
    ms = (time.perf_counter() - t0) * 1000
    logger.info("ingest latency=%.1fms id=%s", ms, content_id)
    return IngestResponse(content_id=content_id)


@router.post("/similar", response_model=SearchResponse)
def search_similar(body: SearchRequest, request: Request) -> SearchResponse:
    t0 = time.perf_counter()
    vector = _embedder.embed(body.query)
    hits = request.app.state.store.search(vector, body.k)
    ms = (time.perf_counter() - t0) * 1000
    logger.info("search latency=%.1fms query=%r k=%d hits=%d", ms, body.query, body.k, len(hits))
    return SearchResponse(results=[SearchResult(id=id_, score=score) for id_, score in hits])
