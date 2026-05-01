import re
from typing import List
from app.intelligence.llm import AnthropicLLMClient

_STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "with", "this", "that", "are", "was",
    "be", "been", "by", "from", "as", "do", "does", "did", "not",
}

_SYNONYMS: dict = {
    "cpu": ["processor", "compute"],
    "speed": ["performance", "benchmark", "throughput"],
    "fast": ["quick", "rapid", "performance"],
    "slow": ["latency", "bottleneck", "delay"],
    "big": ["large", "scale", "size"],
    "small": ["tiny", "compact", "lightweight"],
    "error": ["exception", "failure", "bug"],
    "fix": ["resolve", "patch", "repair"],
    "search": ["retrieval", "query", "lookup"],
    "store": ["storage", "persist", "save"],
    "memory": ["ram", "cache", "buffer"],
    "ml": ["machine learning", "model", "ai"],
    "ai": ["machine learning", "neural", "model"],
    "db": ["database", "storage", "persistence"],
    "api": ["endpoint", "service", "interface"],
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class QueryRewriter:
    def __init__(self, remove_stopwords: bool = False, expand_synonyms: bool = True) -> None:
        self._remove_stopwords = remove_stopwords
        self._expand_synonyms = expand_synonyms

    def rewrite(self, query: str) -> str:
        tokens = _tokenize(query)

        if self._remove_stopwords:
            tokens = [t for t in tokens if t not in _STOPWORDS]

        if not tokens:
            return query

        expanded: List[str] = list(tokens)
        if self._expand_synonyms:
            for token in tokens:
                syns = _SYNONYMS.get(token, [])
                for syn in syns:
                    if syn not in expanded:
                        expanded.append(syn)

        result = " ".join(expanded)
        return result if result.strip() else query


class LLMQueryRewriter:
    def __init__(self, client: AnthropicLLMClient) -> None:
        self._client = client

    def rewrite(self, query: str) -> str:
        prompt = (
            "Rewrite the query for semantic search. Keep intent identical, expand with concise synonyms, "
            "and return one line only.\n"
            f"Query: {query}"
        )
        try:
            rewritten = self._client.complete(prompt).strip()
            return rewritten or query
        except Exception:
            return query
