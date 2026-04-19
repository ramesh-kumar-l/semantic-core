from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SemanticRelation:
    relation: str
    target_id: str
    target_type: str = "resource"


@dataclass
class SemanticObject:
    id: str
    type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    relations: List[SemanticRelation] = field(default_factory=list)
