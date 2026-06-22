from __future__ import annotations

import numpy as np

from llm_manip.contracts import Action, SkillStatus, WorldState

_NEAR_DIST = 0.06   # metres — "arrived" threshold
_GRASP_DIST = 0.045 # must match mock_env
_DEFAULT_Z = 0.05   # approach height above object


class MoveToSkill:
    """Move EE to a named object's position."""

    def reset(self, args: dict, world: WorldState) -> None:
        self._label = args["label"]
        self._done = False

    def act(self, world: WorldState) -> Action:
        obj = world.find(self._label)
        if obj is None:
            return _idle_action(world)
        target = obj.pose.position.copy()
        target[2] += _DEFAULT_Z
        joints = np.zeros(world.robot.joint_positions.shape)
        joints[:3] = target
        return Action(joint_targets=joints, gripper=world.robot.gripper)

    def status(self, world: WorldState) -> SkillStatus:
        obj = world.find(self._label)
        if obj is None:
            return SkillStatus.FAILURE
        dist = float(np.linalg.norm(world.robot.ee_pose.position - obj.pose.position))
        if dist < _NEAR_DIST:
            return SkillStatus.SUCCESS
        return SkillStatus.RUNNING


class PickSkill:
    """Approach → close gripper → confirm hold. Timeout → FAILURE."""

    _TIMEOUT = 120

    def reset(self, args: dict, world: WorldState) -> None:
        self._label = args["label"]
        self._steps = 0

    def act(self, world: WorldState) -> Action:
        obj = world.find(self._label)
        if obj is None:
            return _idle_action(world)
        target = obj.pose.position.copy()
        dist = float(np.linalg.norm(world.robot.ee_pose.position - target))
        gripper = 1.0 if dist < _GRASP_DIST * 2 else 0.0
        joints = np.zeros(world.robot.joint_positions.shape)
        joints[:3] = target
        return Action(joint_targets=joints, gripper=gripper)

    def status(self, world: WorldState) -> SkillStatus:
        self._steps += 1
        obj = world.find(self._label)
        if obj is None:
            return SkillStatus.FAILURE
        dist = float(np.linalg.norm(world.robot.ee_pose.position - obj.pose.position))
        if dist < _GRASP_DIST and world.robot.holding is not None:
            return SkillStatus.SUCCESS
        if self._steps >= self._TIMEOUT:
            return SkillStatus.FAILURE
        return SkillStatus.RUNNING


class PlaceSkill:
    """Move EE over target → open gripper."""

    _TIMEOUT = 200

    def reset(self, args: dict, world: WorldState) -> None:
        self._target_label = args["target"]
        self._steps = 0
        self._released = False

    def act(self, world: WorldState) -> Action:
        target_obj = world.find(self._target_label)
        joints = np.zeros(world.robot.joint_positions.shape)
        if target_obj is not None:
            dest = target_obj.pose.position.copy()
            dest[2] += _DEFAULT_Z
            joints[:3] = dest
        dist = (
            float(np.linalg.norm(world.robot.ee_pose.position - target_obj.pose.position))
            if target_obj is not None else 999.0
        )
        gripper = 0.0 if dist < _NEAR_DIST else 1.0
        return Action(joint_targets=joints, gripper=gripper)

    def status(self, world: WorldState) -> SkillStatus:
        self._steps += 1
        target_obj = world.find(self._target_label)
        if target_obj is None:
            return SkillStatus.FAILURE
        dist = float(np.linalg.norm(world.robot.ee_pose.position - target_obj.pose.position))
        if dist < _NEAR_DIST and world.robot.holding is None:
            return SkillStatus.SUCCESS
        if self._steps >= self._TIMEOUT:
            return SkillStatus.FAILURE
        return SkillStatus.RUNNING


def _idle_action(world: WorldState) -> Action:
    return Action(
        joint_targets=world.robot.joint_positions.copy(),
        gripper=world.robot.gripper,
    )


MOCK_SKILLS: dict[str, type] = {
    "move_to": MoveToSkill,
    "pick":    PickSkill,
    "place":   PlaceSkill,
}
