from __future__ import annotations

import re

from llm_manip.contracts import Plan, SkillCall, WorldState

# Matches: "put/move/place the <X> on/onto the <Y>"
_PATTERN = re.compile(
    r"(?:put|move|place|stack)\s+the\s+(\w[\w\s]*?)\s+(?:on|onto)\s+the\s+(\w[\w\s]*)",
    re.IGNORECASE,
)


class RuleBasedPlanner:
    """Keyword-based planner for simple pick-and-place instructions."""

    def plan(self, world: WorldState, instruction: str) -> Plan:
        m = _PATTERN.search(instruction)
        if m:
            obj_label = m.group(1).strip().replace(" ", "_")
            target_label = m.group(2).strip().replace(" ", "_")
            return Plan(steps=[
                SkillCall(skill="pick",  args={"label": obj_label}),
                SkillCall(skill="place", args={"label": obj_label, "target": target_label}),
            ])
        raise ValueError(
            f"RuleBasedPlanner cannot parse instruction: {instruction!r}. "
            "Expected: 'put/move/place the <X> on/onto the <Y>'"
        )
