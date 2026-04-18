from typing import List
from pydantic import BaseModel


class IngestResponse(BaseModel):
    content_id: str


class SearchResult(BaseModel):
    id: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
