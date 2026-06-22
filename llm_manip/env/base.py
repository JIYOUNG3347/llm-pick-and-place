from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from llm_manip.contracts import Action


@dataclass
class RawObs:
    rgb: Optional[np.ndarray]           # (H, W, 3) uint8
    depth: Optional[np.ndarray]         # (H, W) float32
    joint_positions: np.ndarray         # (n_joints,)
    ee_pos: np.ndarray                  # (3,)  xyz of end-effector
    gripper: float                      # 0.0 open … 1.0 closed
    holding: Optional[str]              # object id being held (None if empty)
    gt_objects: list[tuple[str, str, np.ndarray]]  # [(id, label, xyz), ...]
    cam: Optional[np.ndarray] = None    # (4, 4) camera extrinsic
    t: int = 0                          # environment step counter


@runtime_checkable
class Env(Protocol):
    n_joints: int

    def reset(self) -> RawObs: ...
    def observe(self) -> RawObs: ...
    def step(self, action: Action) -> RawObs: ...
    def close(self) -> None: ...
