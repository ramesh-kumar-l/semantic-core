from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class GraphDB(ABC):
    @abstractmethod
    def add_node(self, id: str, type: str, metadata: dict) -> None: ...

    @abstractmethod
    def add_edge(self, source: str, relation: str, target: str, weight: float = 1.0) -> None: ...

    @abstractmethod
    def get_node(self, id: str) -> Optional[Dict]: ...

    @abstractmethod
    def get_neighbors(self, id: str, relation: str = None) -> List[Dict]:
        """Nodes reachable FROM id via relation (outbound)."""
        ...

    @abstractmethod
    def get_predecessors(self, id: str, relation: str = None) -> List[Dict]:
        """Nodes that point TO id via relation (inbound)."""
        ...

    @abstractmethod
    def query_nodes(self, filters: dict) -> List[Dict]: ...

    @abstractmethod
    def traverse(self, start_ids: List[str], relation: str, depth: int = 1) -> List[Dict]:
        """Multi-hop traversal (outbound) up to `depth` hops."""
        ...
