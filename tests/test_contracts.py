import numpy as np
import pytest

from llm_manip.contracts import (
    Action,
    ActionResult,
    ObjectState,
    Plan,
    Pose,
    RobotState,
    SkillCall,
    SkillStatus,
    WorldState,
)


def test_pose_from_xyz():
    p = Pose.from_xyz(1.0, 2.0, 3.0)
    assert np.allclose(p.position, [1.0, 2.0, 3.0])
    assert np.allclose(p.quaternion, [1.0, 0.0, 0.0, 0.0])


def test_world_state_find():
    obj = ObjectState(id="a", label="red_cube", pose=Pose.from_xyz(0, 0, 0))
    robot = RobotState(
        joint_positions=np.zeros(6),
        ee_pose=Pose.from_xyz(0, 0, 0.3),
        gripper=0.0,
        holding=None,
    )
    world = WorldState(t=0, objects=[obj], robot=robot)
    assert world.find("red_cube") is obj
    assert world.find("blue_plate") is None


def test_skill_status_enum():
    assert SkillStatus.RUNNING.value == "RUNNING"
    assert SkillStatus.SUCCESS != SkillStatus.FAILURE


def test_plan_construction():
    plan = Plan(steps=[
        SkillCall(skill="pick",  args={"label": "red_cube"}),
        SkillCall(skill="place", args={"label": "red_cube", "target": "blue_plate"}),
    ])
    assert len(plan.steps) == 2
    assert plan.steps[0].skill == "pick"


def test_action_fields():
    a = Action(joint_targets=np.ones(6), gripper=0.5)
    assert a.gripper == 0.5
    assert a.joint_targets.shape == (6,)


def test_action_result():
    robot = RobotState(np.zeros(6), Pose.from_xyz(0, 0, 0), 0.0, None)
    world = WorldState(t=0, objects=[], robot=robot)
    r = ActionResult(status=SkillStatus.SUCCESS, message="ok", world_after=world)
    assert r.status == SkillStatus.SUCCESS
