import re
from typing import List

_STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "with", "this", "that", "are", "was",
    "be", "been", "by", "from", "as", "do", "does", "did", "not",
}

_NATURAL_LANG_PATTERNS = re.compile(
    r"\b(how|what|why|when|where|who|which|can|could|should|would|explain|describe|tell me)\b",
    re.IGNORECASE,
)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class QueryAnalyzer:
    def analyze(self, query: str) -> dict:
        tokens = _tokenize(query)
        keywords = [t for t in tokens if t not in _STOPWORDS]
        length = len(tokens)
        is_natural = bool(_NATURAL_LANG_PATTERNS.search(query))
        is_ambiguous = length <= 2

        if length <= 2 or (not is_natural and len(keywords) == len(tokens)):
            query_type = "keyword"
        elif is_natural or length >= 8:
            query_type = "semantic"
        else:
            query_type = "hybrid"

        return {
            "length": length,
            "type": query_type,
            "keywords": keywords,
            "is_ambiguous": is_ambiguous,
        }
