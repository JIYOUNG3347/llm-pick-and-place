from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from llm_manip.contracts import ActionResult, SkillStatus, WorldState
from llm_manip.env.base import Env
from llm_manip.executor.base import SkillExecutor
from llm_manip.perception.base import Perception
from llm_manip.planner.base import Planner


@dataclass
class RunResult:
    success: bool
    n_replans: int
    trace: list[str] = field(default_factory=list)
    final_world: Optional[WorldState] = None


class Orchestrator:
    def __init__(
        self,
        env: Env,
        perception: Perception,
        planner: Planner,
        executor: SkillExecutor,
        max_replans: int = 3,
    ) -> None:
        self._env = env
        self._perception = perception
        self._planner = planner
        self._executor = executor
        self._max_replans = max_replans

    def run(self, instruction: str, verbose: bool = True) -> RunResult:
        """Execute instruction with perceive → plan → execute → re-plan loop.

        Parameters
        ----------
        verbose:
            When True (default) each trace line is also printed to stdout.
            Pass verbose=False from the Gradio launcher to suppress console output.
        """
        trace: list[str] = []
        n_replans = 0
        world: Optional[WorldState] = None

        def log(msg: str) -> None:
            if verbose:
                print(msg)
            trace.append(msg)

        log("[RESET]")
        raw = self._env.reset()
        world = self._perception.perceive(raw)
        log(f"[PERCEIVE] t={world.t} objects={[o.label for o in world.objects]}")

        plan = self._planner.plan(world, instruction)
        log(f"[PLAN] steps={[s.skill+'('+str(s.args)+')' for s in plan.steps]}")

        while True:
            failed_step = None
            for call in plan.steps:
                log(f"[EXEC] {call.skill}({call.args})")
                result: ActionResult = self._executor.run(call, world)
                world = result.world_after
                log(f"  → {result.status.value}: {result.message}")
                if result.status == SkillStatus.FAILURE:
                    failed_step = call
                    break

            if failed_step is None:
                log("[DONE] success")
                return RunResult(
                    success=True,
                    n_replans=n_replans,
                    trace=trace,
                    final_world=world,
                )

            if n_replans >= self._max_replans:
                log(f"[FAIL] replan budget exhausted ({n_replans}/{self._max_replans})")
                return RunResult(
                    success=False,
                    n_replans=n_replans,
                    trace=trace,
                    final_world=world,
                )

            n_replans += 1
            log(f"[RE-PLAN #{n_replans}] re-perceiving after failure of {failed_step.skill!r}")
            raw = self._env.observe()
            world = self._perception.perceive(raw)
            log(f"[PERCEIVE] t={world.t} objects={[o.label for o in world.objects]}")
            plan = self._planner.plan(world, instruction)
            log(f"[PLAN] steps={[s.skill+'('+str(s.args)+')' for s in plan.steps]}")
