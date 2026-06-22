from __future__ import annotations

import numpy as np

from llm_manip.contracts import ObjectState, Pose, RobotState, WorldState
from llm_manip.env.base import RawObs


class OraclePerception:
    """Ground-truth perception using raw.gt_objects.

    Serves as the upper-bound baseline — perfect detection with confidence=1.0.
    """

    def perceive(self, raw: RawObs) -> WorldState:
        objects = [
            ObjectState(
                id=oid,
                label=label,
                pose=Pose.from_xyz(*pos),
                confidence=1.0,
            )
            for oid, label, pos in raw.gt_objects
        ]
        robot = RobotState(
            joint_positions=raw.joint_positions.copy(),
            ee_pose=Pose.from_xyz(*raw.ee_pos),
            gripper=raw.gripper,
            holding=raw.holding,
        )
        return WorldState(t=raw.t, objects=objects, robot=robot)
