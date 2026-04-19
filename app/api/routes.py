import uuid
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from app.schemas.requests import (
    IngestRequest, SearchRequest, FeedbackRequest,
    SemanticIngestRequest, SemanticQueryRequest, GraphQueryRequest,
)
from app.schemas.responses import (
    IngestResponse, SearchResponse, SearchResult, FeedbackResponse,
    SemanticIngestResponse, SemanticQueryResponse, GraphNode, GraphNeighborsResponse,
    GraphQueryResponse, TraversalStep,
)
from app.services.embedding import EmbeddingService
from app.observability.logger import get_json_logger
from app.observability.timing import Timer
from app.core.config import config
from app.hybrid.fusion import fuse_results
from app.memory.namespace import resolve as resolve_namespace
from app.intelligence.analyzer import QueryAnalyzer
from app.intelligence.strategy import RetrievalStrategy
from app.intelligence.rewrite import QueryRewriter
from app.intelligence.multi_query import MultiQueryGenerator
from app.planner.store import PlannerStore

logger = get_json_logger("api.routes")
router = APIRouter()

_embedder = EmbeddingService()
_analyzer = QueryAnalyzer()
_strategist = RetrievalStrategy()
_rewriter = QueryRewriter()
_multi_query_gen = MultiQueryGenerator()

_LOW_SCORE = 0.3
_HIGH_LATENCY_MS = 200


@router.post("/content", response_model=IngestResponse)
def ingest_content(body: IngestRequest, request: Request) -> IngestResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    memory_cfg = config.get("memory", {})

    if memory_cfg.get("enabled", False):
        namespace = resolve_namespace(body.namespace)
        ingest_timer = Timer().start()
        content_id = str(uuid.uuid4())
        request.app.state.memory.add(namespace, content_id, body.text)
        total_ms = ingest_timer.stop()

        logger.info(
            "ingest",
            extra={
                "request_id": request_id,
                "endpoint": "/content",
                "namespace": namespace,
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

    # Legacy path (memory disabled)
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


def _run_intelligence(query: str) -> tuple[str, dict, dict]:
    """Returns (effective_query, analysis, strategy_plan). No-op if intelligence disabled."""
    intel_cfg = config.get("intelligence", {})
    if not intel_cfg.get("enabled", False):
        return query, {}, {}

    analysis = _analyzer.analyze(query)
    plan = _strategist.plan(analysis)

    effective_query = query
    if intel_cfg.get("rewrite", True):
        effective_query = _rewriter.rewrite(query)
        if not effective_query:
            effective_query = query

    logger.info(
        "query_analysis",
        extra={
            "event": "query_analysis",
            "original_query": query,
            "rewritten_query": effective_query,
            "type": analysis.get("type"),
            "alpha": plan.get("alpha"),
            "is_ambiguous": analysis.get("is_ambiguous"),
        },
    )
    return effective_query, analysis, plan


def _fuse_multi_query(
    queries: list[str],
    store,
    bm25,
    plan: dict,
    embedder: EmbeddingService,
) -> list[tuple[str, float]]:
    """Run retrieval for each query and merge by max score."""
    merged: dict[str, float] = {}
    for q in queries:
        vec = embedder.embed(q)
        if plan.get("use_hybrid") and bm25 is not None:
            vec_hits = store.search(vec, plan.get("vector_k", 20))
            bm25_hits = bm25.search(q, plan.get("bm25_k", 20))
            hits = fuse_results(vec_hits, bm25_hits, alpha=plan.get("alpha", 0.7))
        else:
            hits = store.search(vec, 20)
        for id_, score in hits:
            if score > merged.get(id_, -1):
                merged[id_] = score
    return sorted(merged.items(), key=lambda x: x[1], reverse=True)


@router.post("/similar", response_model=SearchResponse)
def search_similar(body: SearchRequest, request: Request) -> SearchResponse:
    intel_cfg = config.get("intelligence", {})
    effective_query, analysis, plan = _run_intelligence(body.query)

    planner_cfg = config.get("planner", {})
    if planner_cfg.get("enabled", False):
        namespace_for_plan = resolve_namespace(body.namespace)
        plan = request.app.state.planner.plan(
            body.query, namespace_for_plan, {"analysis": analysis}
        )
        logger.info(
            "query_plan",
            extra={
                "event": "query_plan",
                "mode": planner_cfg.get("mode", "rule_based"),
                "alpha": plan.get("alpha"),
                "vector_k": plan.get("vector_k"),
                "bm25_k": plan.get("bm25_k"),
                "use_hybrid": plan.get("use_hybrid"),
            },
        )

    memory_cfg = config.get("memory", {})
    if memory_cfg.get("enabled", False):
        namespace = resolve_namespace(body.namespace)
        search_query = effective_query if intel_cfg.get("enabled", False) else body.query
        hits = request.app.state.memory.search(namespace, search_query, body.k)

        learning_cfg = config.get("learning", {})
        if learning_cfg.get("enabled", False) and hits:
            if learning_cfg.get("rule_based", True):
                hits = request.app.state.l2r_rule.rerank(search_query, namespace, hits)
            if learning_cfg.get("ml_model", False) and request.app.state.l2r_model.is_ready:
                hits = request.app.state.l2r_model.rerank(search_query, hits)
            hits = hits[: body.k]

        return SearchResponse(results=[SearchResult(id=id_, score=score) for id_, score in hits])

    # Legacy path (memory disabled)
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    backend = type(request.app.state.store).__name__
    hybrid_cfg = config["hybrid"]
    ranking_cfg = config["ranking"]

    intel_enabled = intel_cfg.get("enabled", False)
    active_query = effective_query if intel_enabled else body.query

    embed_timer = Timer().start()
    vector = _embedder.embed(active_query)
    embedding_ms = embed_timer.stop()

    search_timer = Timer().start()

    use_multi = intel_enabled and intel_cfg.get("multi_query", False)
    if use_multi:
        queries = _multi_query_gen.generate(active_query)
        logger.info(
            "query_analysis",
            extra={"event": "query_analysis", "queries_generated": len(queries)},
        )
        hits = _fuse_multi_query(
            queries,
            request.app.state.store,
            request.app.state.bm25 if hybrid_cfg["enabled"] else None,
            plan if plan else {"use_hybrid": hybrid_cfg["enabled"], "alpha": hybrid_cfg["alpha"],
                               "vector_k": hybrid_cfg["vector_k"], "bm25_k": hybrid_cfg["bm25_k"]},
            _embedder,
        )
    elif hybrid_cfg["enabled"]:
        vector_k = plan.get("vector_k", hybrid_cfg["vector_k"]) if plan else hybrid_cfg["vector_k"]
        bm25_k = plan.get("bm25_k", hybrid_cfg["bm25_k"]) if plan else hybrid_cfg["bm25_k"]
        alpha = plan.get("alpha", hybrid_cfg["alpha"]) if plan else hybrid_cfg["alpha"]

        vector_hits = request.app.state.store.search(vector, vector_k)
        bm25_hits = request.app.state.bm25.search(active_query, bm25_k)

        if not vector_hits and not bm25_hits:
            logger.warning(
                "retrieval_warning",
                extra={"event": "retrieval_warning", "type": "empty_fusion",
                       "query": body.query},
            )

        vec_ids = {id_ for id_, _ in vector_hits}
        bm25_ids = {id_ for id_, _ in bm25_hits}
        if vec_ids and bm25_ids and not (vec_ids & bm25_ids):
            logger.warning(
                "retrieval_warning",
                extra={"event": "retrieval_warning", "type": "result_disagreement",
                       "query": body.query},
            )

        hits = fuse_results(vector_hits, bm25_hits, alpha=alpha)
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

    # L2R phase
    learning_cfg = config.get("learning", {})
    if learning_cfg.get("enabled", False) and hits:
        namespace = resolve_namespace(body.namespace)
        if learning_cfg.get("rule_based", True):
            hits = request.app.state.l2r_rule.rerank(body.query, namespace, hits)
        if learning_cfg.get("ml_model", False) and request.app.state.l2r_model.is_ready:
            hits = request.app.state.l2r_model.rerank(body.query, hits)

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


@router.post("/feedback", response_model=FeedbackResponse)
def record_feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
    learning_cfg = config.get("learning", {})
    if not learning_cfg.get("enabled", False):
        return FeedbackResponse(status="disabled")

    namespace = resolve_namespace(body.namespace)
    event = {
        "query": body.query,
        "namespace": namespace,
        "results": body.results,
        "clicked": body.clicked,
        "position": body.position,
        "timestamp": body.timestamp,
    }

    request.app.state.feedback_store.add(event)
    if learning_cfg.get("rule_based", True):
        request.app.state.l2r_rule.record_click(namespace, body.query, body.clicked)

    planner_cfg = config.get("planner", {})
    if planner_cfg.get("enabled", False):
        ctr = 1.0 if body.clicked else 0.0
        mrr = 1.0 / (body.position + 1) if body.clicked else 0.0
        query_hash = PlannerStore.hash_query(body.query)
        current_plan = request.app.state.planner.plan(body.query, namespace, {})
        request.app.state.planner_store.update(query_hash, current_plan, {"ctr": ctr, "mrr": mrr})

    logger.info(
        "feedback_recorded",
        extra={
            "event": "feedback_recorded",
            "query": body.query,
            "namespace": namespace,
            "clicked": body.clicked,
            "position": body.position,
        },
    )
    return FeedbackResponse(status="ok")


# ── Semantic / Graph endpoints ─────────────────────────────────────────────────

def _require_semantic(request: Request):
    svc = getattr(request.app.state, "semantic", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Semantic layer not enabled (GRAPH_ENABLED=false)")
    return svc


@router.post("/semantic/ingest", response_model=SemanticIngestResponse)
def semantic_ingest(body: SemanticIngestRequest, request: Request) -> SemanticIngestResponse:
    svc = _require_semantic(request)

    obj_id = body.id or str(uuid.uuid4())
    input_data: dict = dict(body.metadata)
    if body.text:
        input_data["text"] = body.text

    obj = svc.ingest(id=obj_id, input_data=input_data, type_hint=body.type)

    # Auto-link to related existing nodes
    linking_engine = getattr(request.app.state, "linking_engine", None)
    links_created = 0
    if linking_engine is not None:
        node = svc.get_node(obj.id)
        if node:
            links_created = linking_engine.link(node, svc._graph)

    logger.info(
        "semantic_ingest",
        extra={
            "event": "semantic_ingest",
            "id": obj_id,
            "type": obj.type,
            "links_created": links_created,
        },
    )
    return SemanticIngestResponse(id=obj.id, type=obj.type, metadata=obj.metadata)


@router.post("/semantic/query", response_model=SemanticQueryResponse)
def semantic_query(body: SemanticQueryRequest, request: Request) -> SemanticQueryResponse:
    svc = _require_semantic(request)

    intent = {
        "type": body.type,
        "person": body.person,
        "location": body.location,
        "event": body.event,
    }
    intent = {k: v for k, v in intent.items() if v}

    nodes = svc.query(intent)
    nodes = nodes[: body.k]

    logger.info(
        "semantic_query",
        extra={"event": "semantic_query", "intent": intent, "hits": len(nodes)},
    )
    return SemanticQueryResponse(
        nodes=[GraphNode(**n) for n in nodes],
        total=len(nodes),
    )


@router.get("/semantic/node/{node_id}", response_model=GraphNeighborsResponse)
def get_graph_node(
    node_id: str,
    request: Request,
    relation: Optional[str] = None,
    direction: str = "outbound",
    depth: int = 1,
) -> GraphNeighborsResponse:
    svc = _require_semantic(request)

    node = svc.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    neighbors = svc.get_related(node_id, relation=relation, depth=depth, direction=direction)
    return GraphNeighborsResponse(
        node=GraphNode(**node),
        neighbors=[GraphNode(**n) for n in neighbors],
    )


# ── Graph traversal query endpoint ────────────────────────────────────────────

@router.post("/graph_query/execute", response_model=GraphQueryResponse)
def graph_query_execute(body: GraphQueryRequest, request: Request) -> GraphQueryResponse:
    """
    Graph-traversal-based retrieval.

    Builds an anchor-based traversal plan from person/location/event filters,
    runs bounded BFS, and returns matching nodes.  Falls back to an empty result
    set (not an error) when no anchors resolve — callers may set
    fallback_to_retrieval=true to indicate they want the caller-side retrieval
    fallback applied.
    """
    svc = _require_semantic(request)

    planner = getattr(request.app.state, "graph_planner", None)
    engine = getattr(request.app.state, "graph_query_engine", None)

    if planner is None or engine is None:
        raise HTTPException(status_code=503, detail="Graph query engine not initialised")

    mapping: dict = {
        k: v for k, v in {
            "person": body.person,
            "location": body.location,
            "event": body.event,
            "time": body.time,
            "type": body.type,
            "traversal": body.traversal,
        }.items() if v is not None
    }

    plan = planner.plan(mapping)
    result = engine.execute(plan, svc._graph)

    node_ids: list = result["node_ids"][: body.k]
    fallback_used = False

    if not node_ids and body.fallback_to_retrieval:
        fallback_used = True

    nodes = []
    for nid in node_ids:
        n = svc.get_node(nid)
        if n:
            nodes.append(GraphNode(**n))

    return GraphQueryResponse(
        nodes=nodes,
        total=len(nodes),
        traversal_steps=[
            TraversalStep(**s) for s in result.get("traversal_steps", [])
        ],
        truncated=result.get("truncated", False),
        fallback_used=fallback_used,
    )
