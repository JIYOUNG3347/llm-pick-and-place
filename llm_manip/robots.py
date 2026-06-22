"""Robot configuration registry for the Franka Panda arm."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Franka Panda default values ───────────────────────────────────────────────
_PANDA_HOME_Q = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])

_PANDA_ARM_JOINTS = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]


@dataclass
class RobotConfig:
    """Static configuration for a robot variant.

    New robots: fill all fields; set isaaclab_cfg to enable Isaac Sim.
    """

    name: str
    reach: float                                    # metres — arm reach radius
    usd_path: str                                   # Isaac Nucleus USD path
    n_joints: int = 7                               # arm DOF count
    joint_limits: list[tuple[float, float]] = field(
        default_factory=lambda: [(-2 * math.pi, 2 * math.pi)] * 7
    )
    ee_link: str = "panda_hand"                     # EE body name in Isaac Lab USD
    home_q: np.ndarray = field(default_factory=lambda: _PANDA_HOME_Q.copy())
    gripper: str = "panda_hand"                     # gripper model name (informational)
    # Distance from EE body frame origin to TCP (fingertip midpoint) along EE +Z axis
    tcp_offset_z: float = 0.107                     # panda_hand origin → fingertip midpoint

    # ── Isaac Lab integration ─────────────────────────────────────────────────
    # Module that contains the ArticulationCfg symbol.
    isaaclab_module: str = "isaaclab_assets.robots.franka"
    # Attribute name inside that module.  None = no verified Isaac Lab CFG.
    isaaclab_cfg: Optional[str] = None

    # ── Arm joints (for SceneEntityCfg + home-pose dict construction) ─────────
    arm_joint_names: list[str] = field(default_factory=lambda: list(_PANDA_ARM_JOINTS))

    # ── Gripper joints & position convention ──────────────────────────────────
    # gripper_close_sign controls which path isaac_env uses to compute closed_amount:
    #   -1 → closed_amount = abs(finger_val).
    #        Use when open_pos=0 and the joint moves toward ±close_pos when closing.
    #   +1 → closed_amount = open_pos - finger_val.
    #        Use when the joint value DECREASES on closing (e.g. Panda: 0.04 → 0.0 m).
    #        open_pos must be > close_pos.
    gripper_joint_names: list[str] = field(
        default_factory=lambda: ["panda_finger_joint1", "panda_finger_joint2"]
    )
    gripper_open_pos: float = 0.04                  # joint value when fully open
    gripper_close_pos: float = 0.0                  # joint value target when fully closed
    gripper_close_sign: float = 1.0                 # see above
    min_grasp_finger: float = 0.010                 # closed_amount threshold for grasp detection

    # Prim path (relative to robot root) attached to Body0 of the grasp FixedJoint
    gripper_attach_prim: str = "panda_hand"
    # Prim names (relative to robot root) to apply high-friction physics material
    finger_friction_prims: list[str] = field(
        default_factory=lambda: ["panda_leftfinger", "panda_rightfinger"]
    )

    grasp_dist_thresh: float = 0.045       # TCP proximity threshold for grasp detection (m)


ROBOTS: dict[str, RobotConfig] = {
    "panda": RobotConfig(
        name="panda",
        reach=0.855,
        usd_path="",   # resolved via Isaac Nucleus in FRANKA_PANDA_HIGH_PD_CFG
        # Isaac Lab
        isaaclab_module="isaaclab_assets.robots.franka",
        isaaclab_cfg="FRANKA_PANDA_HIGH_PD_CFG",   # stiffness=400, damping=80 — IK-suitable
        arm_joint_names=list(_PANDA_ARM_JOINTS),
        ee_link="panda_hand",
        tcp_offset_z=0.107,   # panda_hand origin to fingertip midpoint (from ik_abs_env_cfg)
        # Gripper: parallel finger gripper
        # Joints go from 0.04 (open) to 0.0 (closed) — decreasing, close_sign=+1
        gripper="panda_hand",
        gripper_joint_names=["panda_finger_joint1", "panda_finger_joint2"],
        gripper_open_pos=0.04,
        gripper_close_pos=0.0,
        gripper_close_sign=1.0,    # open_pos - finger_val gives closed_amount
        min_grasp_finger=0.010,    # DexCube blocks finger at joint≈0.027 (closed_amount≈0.013);
                                   # threshold must be below that observed contact value
        gripper_attach_prim="panda_hand",
        finger_friction_prims=["panda_leftfinger", "panda_rightfinger"],
        # Home + limits
        n_joints=7,
        home_q=_PANDA_HOME_Q.copy(),
        joint_limits=[(-2 * math.pi, 2 * math.pi)] * 7,
        grasp_dist_thresh=0.045,
    ),
}

# Robots with a verified Isaac Lab articulation CFG.
# Only these may be used with executor=ik in Isaac Sim.
ISAAC_SUPPORTED_ROBOTS: frozenset[str] = frozenset(
    name for name, cfg in ROBOTS.items() if cfg.isaaclab_cfg is not None
)


def get_robot(name: str) -> RobotConfig:
    """Return RobotConfig by name.  Raises KeyError with available names on miss."""
    if name not in ROBOTS:
        raise KeyError(
            f"Unknown robot {name!r}. Available: {list(ROBOTS)}"
        )
    return ROBOTS[name]
