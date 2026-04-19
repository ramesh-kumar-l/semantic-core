from typing import Any, Dict, List


class GraphPlanner:
    """
    Converts an intent mapping into a bounded traversal plan.

    Keeps planning deterministic and simple — no ML, no heuristics.
    Each filter type maps to a known anchor prefix and traversal target.
    """

    def plan(self, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a traversal plan from an intent mapping.

        Supported mapping keys: person, location, event, time, traversal (override).

        Returns::
            {
                "start": [anchor_node_ids],
                "traversal": [type_steps],   # types to follow at each hop
                "filters": {...},
            }
        """
        plan: Dict[str, Any] = {"start": [], "traversal": [], "filters": {}}

        traversal_override: List[str] = mapping.get("traversal") or []

        if mapping.get("person"):
            pid = f"person_{mapping['person'].lower().strip()}"
            plan["start"].append(pid)

        if mapping.get("location"):
            lid = f"location_{mapping['location'].lower().strip()}"
            plan["start"].append(lid)

        if mapping.get("event"):
            eid = f"event_{mapping['event'].lower().strip().replace(' ', '_')}"
            plan["start"].append(eid)

        if traversal_override:
            plan["traversal"] = traversal_override
        elif plan["start"]:
            plan["traversal"] = ["media"]

        if mapping.get("time"):
            plan["filters"]["time"] = mapping["time"]

        if mapping.get("type"):
            plan["filters"]["type"] = mapping["type"]

        return plan
