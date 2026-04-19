import logging

logger = logging.getLogger(__name__)


class LLMRewriter:
    """
    Pluggable LLM-based query rewriter. Override _call_llm to connect a real LLM.
    Falls back to original query on any failure or timeout.
    """

    def __init__(self, timeout_s: float = 2.0) -> None:
        self._timeout_s = timeout_s

    def rewrite(self, query: str) -> str:
        try:
            result = self._call_llm(query)
            return result.strip() if result and result.strip() else query
        except Exception as exc:
            logger.warning("llm_rewriter.fallback query=%r reason=%s", query, exc)
            return query

    def _call_llm(self, query: str) -> str:
        # Subclass and override to plug in a real LLM (OpenAI, Anthropic, etc.)
        raise NotImplementedError("LLMRewriter._call_llm not implemented")
