from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from llm_manip.contracts import (
    Action,
    ActionResult,
    SkillCall,
    SkillStatus,
    WorldState,
)
from llm_manip.env.base import Env
from llm_manip.perception.base import Perception


@runtime_checkable
class Skill(Protocol):
    def reset(self, args: dict, world: WorldState) -> None: ...
    def act(self, world: WorldState) -> Action: ...
    def status(self, world: WorldState) -> SkillStatus: ...


class SkillExecutor:
    """Closed-loop executor: act → env.step → perceive → repeat."""

    def __init__(
        self,
        env: Env,
        perception: Perception,
        skills: dict[str, type],  # name → Skill class (not instance)
    ) -> None:
        self._env = env
        self._perception = perception
        self._skills = skills

    def run(self, call: SkillCall, world: WorldState) -> ActionResult:
        skill_cls = self._skills.get(call.skill)
        if skill_cls is None:
            return ActionResult(
                status=SkillStatus.FAILURE,
                message=f"Unknown skill: {call.skill!r}",
                world_after=world,
            )

        skill: Skill = skill_cls()
        skill.reset(call.args, world)

        current_world = world
        while True:
            st = skill.status(current_world)
            if st != SkillStatus.RUNNING:
                return ActionResult(
                    status=st,
                    message=f"{call.skill} → {st.value}",
                    world_after=current_world,
                )
            action = skill.act(current_world)
            raw = self._env.step(action)
            current_world = self._perception.perceive(raw)
