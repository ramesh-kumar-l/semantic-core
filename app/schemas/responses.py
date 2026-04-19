from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class IngestResponse(BaseModel):
    content_id: str


class SearchResult(BaseModel):
    id: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


class FeedbackResponse(BaseModel):
    status: str


# ── Semantic / Graph responses ─────────────────────────────────────────────────

class SemanticIngestResponse(BaseModel):
    id: str
    type: str
    metadata: Dict[str, Any]


class GraphNode(BaseModel):
    id: str
    type: str
    metadata: Dict[str, Any]


class SemanticQueryResponse(BaseModel):
    nodes: List[GraphNode]
    total: int


class GraphNeighborsResponse(BaseModel):
    node: Optional[GraphNode]
    neighbors: List[GraphNode]


class TraversalStep(BaseModel):
    step: str
    count: int


class GraphQueryResponse(BaseModel):
    nodes: List[GraphNode]
    total: int
    traversal_steps: List[TraversalStep]
    truncated: bool
    fallback_used: bool
