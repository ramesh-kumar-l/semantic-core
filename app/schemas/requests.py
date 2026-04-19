from typing import List, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str
    namespace: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=100)
    namespace: Optional[str] = None


class FeedbackRequest(BaseModel):
    query: str
    namespace: Optional[str] = None
    results: List[str] = Field(default_factory=list)
    clicked: str
    position: int = Field(default=0, ge=0)
    timestamp: Optional[int] = None
