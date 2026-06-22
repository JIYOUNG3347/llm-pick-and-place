"""Isaac Sim integration tests — skipped unless running inside Isaac Lab Python.

Run these only via:
    /home/user/IsaacLab/isaaclab.sh -p -m pytest tests/test_isaac_integration.py -v
"""
from __future__ import annotations

import importlib.util

import pytest

# ── Skip the entire module if pxr (USD / Isaac) is not available ──────────
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pxr") is None,
    reason="Isaac Lab / pxr not available in this Python environment",
)


def test_isaac_env_importable():
    """IsaacEnv module must import without error in any Python."""
    from llm_manip.env.isaac_env import IsaacEnv  # noqa: F401


def test_ik_skill_importable():
    """IK skills must import without error in any Python."""
    from llm_manip.executor.ik_skill import IK_SKILLS  # noqa: F401
    assert "pick" in IK_SKILLS
    assert "place" in IK_SKILLS
    assert "move_to" in IK_SKILLS


def test_contracts_ee_target_field():
    """Action.ee_target optional field exists and defaults to None."""
    import numpy as np
    from llm_manip.contracts import Action, Pose

    a = Action(joint_targets=np.zeros(6), gripper=0.0)
    assert a.ee_target is None

    p = Pose.from_xyz(0.4, 0.0, 0.2)
    a2 = Action(joint_targets=np.zeros(6), gripper=0.0, ee_target=p)
    assert a2.ee_target is not None
    assert a2.ee_target.position[0] == pytest.approx(0.4)


# ── Tests that actually require a running Isaac Sim ───────────────────────
# These are marked with the additional marker `isaac_sim` so they can be
# further filtered:  pytest -m "not isaac_sim" skips only them.

@pytest.mark.isaac_sim
def test_isaac_env_boots(tmp_path):
    """IsaacEnv boots, resets, and returns a valid RawObs.

    Only runs inside Isaac Lab Python with simulation available.
    AppLauncher must already be created before this point — run via
    the isaaclab.sh entry point.
    """
    # AppLauncher should already be running (via isaaclab.sh -p -m pytest)
    from llm_manip.env.isaac_env import IsaacEnv
    from llm_manip.robots import get_robot
    from llm_manip.scenes import get_scene

    robot_cfg = get_robot("panda")
    scene_cfg = get_scene("tabletop_rb")
    env = IsaacEnv(robot_cfg=robot_cfg, scene_cfg=scene_cfg)
    raw = env.reset()

    assert raw.joint_positions.shape == (7,)
    assert raw.ee_pos.shape == (3,)
    assert len(raw.gt_objects) == 2
    labels = {label for _, label, _ in raw.gt_objects}
    assert "red_cube" in labels
    assert "blue_cube" in labels
    env.close()
