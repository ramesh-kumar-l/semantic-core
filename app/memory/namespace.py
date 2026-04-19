import re
from typing import Set

_VALID = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")
DEFAULT_NAMESPACE = "default"


def resolve(namespace: str | None) -> str:
    if not namespace:
        return DEFAULT_NAMESPACE
    return namespace.strip()


def validate(namespace: str) -> None:
    if not _VALID.match(namespace):
        raise ValueError(
            f"Invalid namespace {namespace!r}: use alphanumeric, hyphens, underscores, max 64 chars."
        )


class NamespaceManager:
    def __init__(self) -> None:
        self._loaded: Set[str] = set()

    def mark_loaded(self, namespace: str) -> None:
        self._loaded.add(namespace)

    def is_loaded(self, namespace: str) -> bool:
        return namespace in self._loaded

    def loaded_namespaces(self) -> Set[str]:
        return set(self._loaded)
