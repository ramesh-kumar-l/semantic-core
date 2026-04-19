import re
from typing import Any, Dict, List

from .schema import resolve_entity_type


class MetadataExtractor:
    """Derive structured metadata from arbitrary input."""

    def extract(self, input_data: Any) -> Dict[str, Any]:
        if isinstance(input_data, dict):
            return self._from_dict(input_data)
        if isinstance(input_data, str):
            return self._from_text(input_data)
        return {"type": "resource", "raw": str(input_data)}

    # ---------------------------------------------------------------- internals

    def _from_dict(self, data: dict) -> Dict[str, Any]:
        meta = {k: v for k, v in data.items()}
        meta["type"] = resolve_entity_type(meta.get("type", "resource"))
        return meta

    def _from_text(self, text: str) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "type": "resource",
            "text": text,
            "people": self._extract_people(text),
            "location": self._extract_location(text),
            "timestamp": None,
            "source": "text",
        }
        return meta

    @staticmethod
    def _extract_people(text: str) -> List[str]:
        # @mention style
        return re.findall(r"@(\w+)", text)

    @staticmethod
    def _extract_location(text: str) -> str | None:
        # #tag style
        tags = re.findall(r"#(\w+)", text)
        return tags[0] if tags else None
