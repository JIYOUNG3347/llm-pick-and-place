import pytest

from llm_manip.env.mock_env import MockEnv
from llm_manip.executor.base import SkillExecutor
from llm_manip.executor.mock_skills import MOCK_SKILLS
from llm_manip.orchestrator import Orchestrator
from llm_manip.perception.oracle import OraclePerception
from llm_manip.planner.rule_based import RuleBasedPlanner

INSTRUCTION = "put the red cube on the blue plate"


def _make_orchestrator(slip_once: bool) -> Orchestrator:
    env        = MockEnv(slip_once=slip_once)
    perception = OraclePerception()
    planner    = RuleBasedPlanner()
    executor   = SkillExecutor(env, perception, MOCK_SKILLS)
    return Orchestrator(env, perception, planner, executor, max_replans=3)


def test_slip_once_causes_one_replan():
    """First pick times out → re-plan → second attempt succeeds."""
    result = _make_orchestrator(slip_once=True).run(INSTRUCTION)
    assert result.success is True
    assert result.n_replans == 1


def test_no_slip_no_replan():
    """Without the slip, pick succeeds on first try — no re-plan needed."""
    result = _make_orchestrator(slip_once=False).run(INSTRUCTION)
    assert result.success is True
    assert result.n_replans == 0
