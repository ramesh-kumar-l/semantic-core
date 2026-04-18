from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=100)
