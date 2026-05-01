import re
from typing import List
from app.intelligence.llm import AnthropicLLMClient

_SYNONYMS: dict = {
    "cpu": ["processor"],
    "speed": ["performance"],
    "fast": ["quick"],
    "slow": ["latency"],
    "error": ["exception"],
    "fix": ["resolve"],
    "search": ["retrieval"],
    "store": ["storage"],
    "memory": ["cache"],
    "ml": ["machine learning"],
    "ai": ["neural network"],
    "db": ["database"],
    "api": ["endpoint"],
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class MultiQueryGenerator:
    def __init__(self, max_queries: int = 3) -> None:
        self._max_queries = max_queries

    def generate(self, query: str) -> List[str]:
        queries = [query]
        tokens = _tokenize(query)

        for i, token in enumerate(tokens):
            if len(queries) >= self._max_queries:
                break
            syns = _SYNONYMS.get(token)
            if syns:
                variant_tokens = tokens[:i] + [syns[0]] + tokens[i + 1:]
                variant = " ".join(variant_tokens)
                if variant != query and variant not in queries:
                    queries.append(variant)

        return queries[: self._max_queries]


class LLMMultiQueryGenerator:
    def __init__(self, client: AnthropicLLMClient, max_queries: int = 3) -> None:
        self._client = client
        self._max_queries = max_queries

    def generate(self, query: str) -> List[str]:
        prompt = (
            "Generate concise paraphrases for retrieval. Return one query per line, no numbering. "
            f"Return at most {self._max_queries} lines.\n"
            f"Query: {query}"
        )
        try:
            output = self._client.complete(prompt)
            queries = [line.strip("- ").strip() for line in output.splitlines() if line.strip()]
            deduped: List[str] = []
            for candidate in [query, *queries]:
                if candidate and candidate not in deduped:
                    deduped.append(candidate)
            return deduped[: self._max_queries]
        except Exception:
            return [query]
