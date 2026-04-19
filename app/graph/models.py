from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Node:
    id: str
    type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    relation: str
    target: str
