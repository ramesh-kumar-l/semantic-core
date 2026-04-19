from typing import Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str
    namespace: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=100)
    namespace: Optional[str] = None
