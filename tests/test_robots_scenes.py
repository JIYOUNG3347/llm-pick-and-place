"""Tests for robot/scene registries and their integration with MockEnv."""
import math

import numpy as np
import pytest

from llm_manip.robots import ROBOTS, RobotConfig, get_robot
from llm_manip.scenes import SCENES, SceneConfig, get_scene


# ── Registry ────────────────────────────────────────────────────────────────

def test_only_panda_present():
    assert set(ROBOTS) == {"panda"}


def test_get_robot_returns_config():
    for name in ROBOTS:
        r = get_robot(name)
        assert isinstance(r, RobotConfig)
        assert r.n_joints >= 6
        assert r.reach > 0
        assert len(r.joint_limits) == r.n_joints
        assert r.home_q.shape == (r.n_joints,)


def test_get_robot_unknown_raises():
    with pytest.raises(KeyError, match="unknown_bot"):
        get_robot("unknown_bot")


def test_tabletop_rb_scene_present():
    assert "tabletop_rb" in SCENES


def test_get_scene_returns_config():
    for name in SCENES:
        s = get_scene(name)
        assert isinstance(s, SceneConfig)
        assert len(s.objects) >= 2
        for oid, label, xyz in s.objects:
            assert isinstance(oid, str)
            assert isinstance(label, str)
            assert len(xyz) == 3


def test_tabletop_rb_has_red_and_blue_cube():
    s = get_scene("tabletop_rb")
    labels = [label for _, label, _ in s.objects]
    assert "red_cube"  in labels
    assert "blue_cube" in labels
    assert len(labels) == 2


def test_tabletop_rb_random_spawn_fields():
    s = get_scene("tabletop_rb")
    assert s.use_table is True
    assert s.random_spawn is True
    assert len(s.spawn_bounds) == 4
    assert s.min_object_dist > 0


def test_get_scene_unknown_raises():
    with pytest.raises(KeyError, match="nonexistent"):
        get_scene("nonexistent")


# ── MockEnv integration ─────────────────────────────────────────────────────

def test_mock_env_with_panda_robot():
    from llm_manip.env.mock_env import MockEnv
    robot = get_robot("panda")
    env = MockEnv(slip_once=False, robot=robot)
    raw = env.reset()
    assert env.n_joints == 7
    assert raw.joint_positions.shape == (7,)


def test_mock_env_with_tabletop_rb_scene():
    from llm_manip.env.mock_env import MockEnv
    scene = get_scene("tabletop_rb")
    env = MockEnv(slip_once=False, scene=scene)
    raw = env.reset()
    labels = [label for _, label, _ in raw.gt_objects]
    assert "red_cube"  in labels
    assert "blue_cube" in labels


def test_mock_env_robot_and_scene_combined():
    from llm_manip.env.mock_env import MockEnv
    robot = get_robot("panda")
    scene = get_scene("tabletop_rb")
    env = MockEnv(slip_once=False, robot=robot, scene=scene)
    raw = env.reset()
    assert env.n_joints == 7
    assert len(raw.gt_objects) == 2


def test_mock_env_default_unchanged():
    """Backward-compat: no robot/scene → two default objects, 6 joints."""
    from llm_manip.env.mock_env import MockEnv
    env = MockEnv(slip_once=False)
    raw = env.reset()
    assert env.n_joints == 6
    labels = {label for _, label, _ in raw.gt_objects}
    assert "red_cube" in labels
    assert len(labels) == 2


# ── End-to-end with tabletop_rb ──────────────────────────────────────────────

def test_tabletop_rb_scenario_no_crash():
    """panda + tabletop_rb + literal instruction completes without error."""
    from llm_manip.env.mock_env import MockEnv
    from llm_manip.executor.base import SkillExecutor
    from llm_manip.executor.mock_skills import MOCK_SKILLS
    from llm_manip.orchestrator import Orchestrator
    from llm_manip.perception.oracle import OraclePerception
    from llm_manip.planner.rule_based import RuleBasedPlanner

    robot = get_robot("panda")
    scene = get_scene("tabletop_rb")
    env        = MockEnv(slip_once=True, robot=robot, scene=scene)
    perception = OraclePerception()
    planner    = RuleBasedPlanner()
    executor   = SkillExecutor(env, perception, MOCK_SKILLS)
    orch       = Orchestrator(env, perception, planner, executor, max_replans=3)

    result = orch.run("put the red cube on the blue cube", verbose=False)
    assert isinstance(result.success, bool)
    assert result.n_replans >= 0


def test_tabletop_rb_no_slip_succeeds():
    """Without slip, tabletop_rb pick-place should succeed on first try."""
    from llm_manip.env.mock_env import MockEnv
    from llm_manip.executor.base import SkillExecutor
    from llm_manip.executor.mock_skills import MOCK_SKILLS
    from llm_manip.orchestrator import Orchestrator
    from llm_manip.perception.oracle import OraclePerception
    from llm_manip.planner.rule_based import RuleBasedPlanner

    robot = get_robot("panda")
    scene = get_scene("tabletop_rb")
    env        = MockEnv(slip_once=False, robot=robot, scene=scene)
    perception = OraclePerception()
    planner    = RuleBasedPlanner()
    executor   = SkillExecutor(env, perception, MOCK_SKILLS)
    orch       = Orchestrator(env, perception, planner, executor, max_replans=3)

    result = orch.run("put the red cube on the blue cube", verbose=False)
    assert result.success is True
    assert result.n_replans == 0
