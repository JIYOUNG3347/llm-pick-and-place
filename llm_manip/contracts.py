from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


@dataclass
class Pose:
    position: np.ndarray   # (3,)  x, y, z
    quaternion: np.ndarray # (4,)  w, x, y, z

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float) -> "Pose":
        return cls(
            position=np.array([x, y, z], dtype=float),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        )


@dataclass
class ObjectState:
    id: str
    label: str
    pose: Pose
    confidence: float = 1.0
    bbox: Optional[tuple] = None  # (x1, y1, x2, y2) in image coords


@dataclass
class RobotState:
    joint_positions: np.ndarray  # (n_joints,)
    ee_pose: Pose
    gripper: float               # 0.0 = open, 1.0 = closed
    holding: Optional[str] = None  # object id currently grasped


@dataclass
class WorldState:
    t: int
    objects: list[ObjectState]
    robot: RobotState

    def find(self, label: str) -> Optional[ObjectState]:
        for obj in self.objects:
            if obj.label == label:
                return obj
        return None


@dataclass
class SkillCall:
    skill: str
    args: dict


@dataclass
class Plan:
    steps: list[SkillCall]


class SkillStatus(Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@dataclass
class ActionResult:
    status: SkillStatus
    message: str
    world_after: WorldState


@dataclass
class Action:
    joint_targets: np.ndarray  # (n_joints,) — first 3 interpreted as target EE xyz in mock
    gripper: float             # 0.0 = open, 1.0 = closed
    # Optional EE-space target (position-only IK in IsaacEnv).
    # When set, IsaacEnv uses DifferentialIKController instead of direct joint control.
    # MockEnv and mock_skills ignore this field.
    ee_target: Optional[Pose] = None
