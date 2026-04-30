from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str
    namespace: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=100)
    namespace: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None


class UpdateRequest(BaseModel):
    text: str
    namespace: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    query: str
    namespace: Optional[str] = None
    results: List[str] = Field(default_factory=list)
    clicked: str
    position: int = Field(default=0, ge=0)
    timestamp: Optional[int] = None


# ── Semantic / Graph requests ──────────────────────────────────────────────────

class SemanticIngestRequest(BaseModel):
    id: Optional[str] = None                        # auto-generated if omitted
    text: Optional[str] = None
    type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    namespace: Optional[str] = None


class SemanticQueryRequest(BaseModel):
    query: Optional[str] = None
    type: Optional[str] = None
    person: Optional[str] = None
    location: Optional[str] = None
    event: Optional[str] = None
    depth: int = Field(default=1, ge=1, le=4)
    k: int = Field(default=10, ge=1, le=100)
    namespace: Optional[str] = None


class GraphQueryRequest(BaseModel):
    person: Optional[str] = None
    location: Optional[str] = None
    event: Optional[str] = None
    time: Optional[str] = None
    type: Optional[str] = None
    traversal: Optional[List[str]] = None   # override traversal steps
    k: int = Field(default=10, ge=1, le=100)
    fallback_to_retrieval: bool = True
