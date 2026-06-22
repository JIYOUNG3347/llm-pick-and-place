from __future__ import annotations

from typing import Protocol, runtime_checkable

from llm_manip.contracts import Plan, WorldState

# Fixed skill vocabulary — all planners and executors must use these names.
SKILL_SCHEMA: dict[str, list[str]] = {
    "pick":     ["label"],          # pick up object with given label
    "place":    ["label", "target"],# place held object at/on target
    "move_to":  ["label"],          # move EE to object position (no grasp)
}


@runtime_checkable
class Planner(Protocol):
    def plan(self, world: WorldState, instruction: str) -> Plan: ...
