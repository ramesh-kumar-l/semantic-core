import logging

import httpx

logger = logging.getLogger(__name__)


class AnthropicLLMClient:
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5", timeout_s: float = 5.0) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s

    def complete(self, prompt: str) -> str:
        if not self._api_key:
            raise ValueError("Missing LLM_API_KEY")

        payload = {
            "model": self._model,
            "max_tokens": 180,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        with httpx.Client(timeout=self._timeout_s) as client:
            response = client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        content = data.get("content", [])
        if not content:
            return ""
        text_chunks = [item.get("text", "") for item in content if item.get("type") == "text"]
        return " ".join(chunk.strip() for chunk in text_chunks if chunk).strip()
