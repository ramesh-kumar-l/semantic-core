from typing import Dict, List

BASE_ENTITIES: List[str] = [
    "media",
    "person",
    "time",
    "location",
    "event",
    "activity",
    "communication",
    "calendar_event",
    "resource",
]

SUBTYPES: Dict[str, List[str]] = {
    "media": ["image", "video", "screenshot", "document"],
    "communication": ["call", "message", "email"],
    "event": ["meeting", "birthday", "holiday"],
    "activity": ["exercise", "work", "leisure"],
}

RELATIONS: List[str] = [
    "captured_at",
    "contains",
    "involves",
    "occurs_at",
    "belongs_to",
    "related_to",
    "produced_by",
    "tagged_with",
    "happened_during",
]

# Subtype → canonical base entity
SUBTYPE_TO_BASE: Dict[str, str] = {
    sub: base
    for base, subs in SUBTYPES.items()
    for sub in subs
}

# All known types (base + subtypes)
ALL_TYPES: set = set(BASE_ENTITIES) | {sub for subs in SUBTYPES.values() for sub in subs}


def resolve_entity_type(type_str: str) -> str:
    """Return the canonical entity type, falling back to 'resource'."""
    t = type_str.lower().strip()
    if t in ALL_TYPES:
        return t
    return "resource"
