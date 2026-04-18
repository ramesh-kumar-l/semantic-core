import math
from collections import defaultdict
from typing import Dict, List, Tuple


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Index:
    """
    Lightweight BM25 implementation — no external dependencies.
    Parameters follow standard BM25 defaults (k1=1.5, b=0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._ids: List[str] = []
        self._doc_tokens: List[List[str]] = []
        self._df: Dict[str, int] = defaultdict(int)
        self._avg_dl: float = 0.0

    def add(self, id: str, text: str) -> None:
        tokens = _tokenize(text)
        self._ids.append(id)
        self._doc_tokens.append(tokens)
        for term in set(tokens):
            self._df[term] += 1
        total = sum(len(d) for d in self._doc_tokens)
        self._avg_dl = total / len(self._doc_tokens)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        if not self._ids:
            return []

        query_terms = _tokenize(query)
        n = len(self._ids)
        scores: List[float] = []

        for tokens in self._doc_tokens:
            tf_map: Dict[str, int] = defaultdict(int)
            for t in tokens:
                tf_map[t] += 1
            dl = len(tokens)
            score = 0.0
            for term in query_terms:
                tf = tf_map.get(term, 0)
                df = self._df.get(term, 0)
                if df == 0:
                    continue
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (self._k1 + 1)) / (
                    tf + self._k1 * (1 - self._b + self._b * dl / self._avg_dl)
                )
                score += idf * tf_norm
            scores.append(score)

        top_k = min(k, n)
        indexed = sorted(range(n), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._ids[i], scores[i]) for i in indexed if scores[i] > 0]
