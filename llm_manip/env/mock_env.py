from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from llm_manip.contracts import Action
from llm_manip.env.base import RawObs

if TYPE_CHECKING:
    from llm_manip.robots import RobotConfig
    from llm_manip.scenes import SceneConfig

_GRASP_DIST = 0.045   # metres — gripper closes within this radius
_MAX_STEP = 0.05      # metres per sim step
_GRASP_READY_T = 140  # slip_once: grasp blocked before this step

# Default scene when no SceneConfig is provided (preserves Phase 1 behaviour)
_DEFAULT_OBJECTS = [
    ("red_cube",   "red_cube",   (0.40, 0.00, 0.00)),
    ("blue_plate", "blue_plate", (0.40, 0.30, 0.00)),
]


class MockEnv:
    """Toy 6-DOF arm with analytic kinematics.

    joint_targets[:3] is treated directly as the desired EE xyz (no IK).

    Parameters
    ----------
    slip_once:
        When True (default), grasp is blocked until step _GRASP_READY_T so that
        the first pick attempt times out, triggering a re-plan demonstration.
    robot:
        Optional RobotConfig.  Used for n_joints and reach; USD not loaded.
        Defaults to a 6-joint arm when None.
    scene:
        Optional SceneConfig.  Defines initial object positions.
        Falls back to two_objects (red_cube + blue_plate) when None.
    """

    def __init__(
        self,
        slip_once: bool = True,
        robot: Optional["RobotConfig"] = None,
        scene: Optional["SceneConfig"] = None,
    ) -> None:
        self._slip_once = slip_once
        self._robot = robot
        self._scene = scene

        n = robot.n_joints if robot is not None else 6
        self.n_joints: int = n

        self._t = 0
        self._ee = np.zeros(3, dtype=float)
        self._joints = np.zeros(n, dtype=float)
        self._gripper = 0.0
        self._holding: str | None = None
        self._obj_pos: dict[str, np.ndarray] = {}
        self._obj_labels: dict[str, str] = {}
        # Populated during reset — stored here so _make_obs works before first reset
        self._scene_objects: list[tuple[str, str, tuple]] = []

    def reset(self) -> RawObs:
        self._t = 0
        self._ee = np.array([0.0, 0.0, 0.3], dtype=float)
        self._joints = np.zeros(self.n_joints, dtype=float)
        self._gripper = 0.0
        self._holding = None

        # Build object table from scene or fallback defaults
        source = (
            self._scene.objects
            if self._scene is not None
            else _DEFAULT_OBJECTS
        )
        self._obj_pos = {}
        self._obj_labels = {}
        for oid, label, xyz in source:
            self._obj_pos[oid] = np.array(xyz, dtype=float)
            self._obj_labels[oid] = label

        return self._make_obs()

    def observe(self) -> RawObs:
        return self._make_obs()

    def step(self, action: Action) -> RawObs:
        target_xyz = action.joint_targets[:3].copy()
        delta = target_xyz - self._ee
        dist = float(np.linalg.norm(delta))
        if dist > _MAX_STEP:
            self._ee += delta * (_MAX_STEP / dist)
        else:
            self._ee = target_xyz.copy()

        self._joints[:3] = self._ee
        self._gripper = float(action.gripper)

        # gripper close: attempt grasp
        if self._gripper > 0.5 and self._holding is None:
            for oid, opos in self._obj_pos.items():
                d = float(np.linalg.norm(self._ee - opos))
                if d < _GRASP_DIST:
                    if self._slip_once and self._t < _GRASP_READY_T:
                        break
                    self._holding = oid
                    break

        # gripper open: release
        if self._gripper <= 0.5 and self._holding is not None:
            self._holding = None

        # held object follows EE
        if self._holding is not None:
            self._obj_pos[self._holding] = self._ee.copy()

        self._t += 1
        return self._make_obs()

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    def _make_obs(self) -> RawObs:
        gt = [
            (oid, self._obj_labels[oid], pos.copy())
            for oid, pos in self._obj_pos.items()
        ]
        return RawObs(
            rgb=None,
            depth=None,
            joint_positions=self._joints.copy(),
            ee_pos=self._ee.copy(),
            gripper=self._gripper,
            holding=self._holding,
            gt_objects=gt,
            t=self._t,
        )
