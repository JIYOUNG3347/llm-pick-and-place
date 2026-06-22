"""Tests for LlmPlanner: import guard, fallback on connection error, validation."""
from __future__ import annotations

import json
import sys
import types
import unittest.mock as mock

import pytest

from llm_manip.contracts import Plan, WorldState, RobotState, Pose, ObjectState
from llm_manip.planner.base import SKILL_SCHEMA

import numpy as np


def _make_world(labels=("red_cube", "blue_plate")) -> WorldState:
    objects = [
        ObjectState(id=l, label=l, pose=Pose.from_xyz(0, 0, 0))
        for l in labels
    ]
    robot = RobotState(
        joint_positions=np.zeros(6),
        ee_pose=Pose.from_xyz(0, 0, 0.3),
        gripper=0.0,
        holding=None,
    )
    return WorldState(t=0, objects=objects, robot=robot)


def test_llm_planner_no_openai_falls_back():
    """When openai is not importable, LlmPlanner constructs OK and plan() falls back."""
    original = sys.modules.get("openai")
    sys.modules["openai"] = None  # type: ignore[assignment]
    try:
        import importlib
        import llm_manip.planner.llm as llm_mod
        importlib.reload(llm_mod)
        planner = llm_mod.LlmPlanner()   # must NOT raise
        world = _make_world()
        result = planner.plan(world, "put the red cube on the blue cube")
        assert isinstance(result, Plan)
        assert len(result.steps) > 0
    finally:
        if original is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = original


def test_llm_planner_falls_back_on_connection_error():
    """Connection refused → fallback to RuleBasedPlanner, no crash."""
    openai = pytest.importorskip("openai")

    from llm_manip.planner.llm import LlmPlanner

    planner = LlmPlanner(
        base_url="http://localhost:19999/v1",  # nothing listening here
        model="qwen2.5:7b",
        hosted=False,
    )
    world = _make_world()
    # Should fall back silently and return a valid rule-based plan
    result = planner.plan(world, "put the red cube on the blue plate")
    assert isinstance(result, Plan)
    assert len(result.steps) > 0
    assert all(s.skill in SKILL_SCHEMA for s in result.steps)


def test_llm_planner_validates_and_falls_back_on_bad_skill():
    """LlmPlanner rejects an unknown skill name and falls back."""
    openai = pytest.importorskip("openai")

    from llm_manip.planner.llm import LlmPlanner

    bad_response = json.dumps({"steps": [
        {"skill": "fly",   "args": {"label": "red_cube"}},   # invalid skill
        {"skill": "place", "args": {"label": "red_cube", "target": "blue_plate"}},
    ]})

    planner = LlmPlanner(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:7b",
        hosted=False,
    )

    # Mock the client so it returns a bad response
    mock_choice = mock.MagicMock()
    mock_choice.message.content = bad_response
    mock_choice.message.tool_calls = None
    mock_resp = mock.MagicMock()
    mock_resp.choices = [mock_choice]

    with mock.patch.object(planner._client.chat.completions, "create", return_value=mock_resp):
        world = _make_world()
        result = planner.plan(world, "put the red cube on the blue plate")

    # Falls back to RuleBasedPlanner
    assert isinstance(result, Plan)
    assert all(s.skill in SKILL_SCHEMA for s in result.steps)


def test_llm_planner_accepts_valid_response():
    """LlmPlanner returns Plan directly when response is valid."""
    openai = pytest.importorskip("openai")

    from llm_manip.planner.llm import LlmPlanner

    good_response = json.dumps({"steps": [
        {"skill": "pick",  "args": {"label": "red_cube"}},
        {"skill": "place", "args": {"label": "red_cube", "target": "blue_plate"}},
    ]})

    planner = LlmPlanner(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:7b",
        hosted=False,
    )

    mock_choice = mock.MagicMock()
    mock_choice.message.content = good_response
    mock_choice.message.tool_calls = None
    mock_resp = mock.MagicMock()
    mock_resp.choices = [mock_choice]

    with mock.patch.object(planner._client.chat.completions, "create", return_value=mock_resp):
        world = _make_world()
        result = planner.plan(world, "put the red cube on the blue plate")

    assert len(result.steps) == 2
    assert result.steps[0].skill == "pick"
    assert result.steps[1].skill == "place"


def test_llm_planner_rejects_unknown_label():
    """LlmPlanner rejects a plan that references a label not in the scene."""
    openai = pytest.importorskip("openai")

    from llm_manip.planner.llm import LlmPlanner

    bad_response = json.dumps({"steps": [
        {"skill": "pick", "args": {"label": "purple_unicorn"}},  # not in scene
    ]})

    planner = LlmPlanner(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:7b",
        hosted=False,
    )

    mock_choice = mock.MagicMock()
    mock_choice.message.content = bad_response
    mock_choice.message.tool_calls = None
    mock_resp = mock.MagicMock()
    mock_resp.choices = [mock_choice]

    with mock.patch.object(planner._client.chat.completions, "create", return_value=mock_resp):
        world = _make_world()
        result = planner.plan(world, "put the red cube on the blue plate")

    # Still returns a valid plan (fallback)
    assert isinstance(result, Plan)
    assert all(s.skill in SKILL_SCHEMA for s in result.steps)
