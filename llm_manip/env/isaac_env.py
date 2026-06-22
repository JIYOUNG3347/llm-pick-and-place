"""Isaac Lab 2.3.0 environment for robot pick-and-place.

All Isaac / Omniverse imports deferred to __init__ — module is importable in
regular Python (mock tests pass), executed only inside Isaac Lab's Python.

Robot-agnostic design
---------------------
All robot-specific constants (joint names, EE body, gripper convention, home
pose, TCP offset) are read from the active RobotConfig.  No robot name is
hardcoded.  Currently verified robot: panda.

Coordinate frames
-----------------
observe()   reports the EE body position (ee_link) in scene-local coordinates.
ik_skill.py provides wrist-frame heights; IsaacEnv.step() passes them
directly as the IK target position.

Internally, _get_tcp_world_pos() computes the fingertip midpoint (TCP) for
grasp detection only; it is NOT exposed to the skill layer.

Gripper convention
------------------
Two conventions are supported (controlled by RobotConfig.gripper_close_sign):
  -1: joint goes NEGATIVE when closing; target is positive.
      closed_amount = abs(finger_val); norm = abs(finger_val) / close_pos
  +1 (Franka Panda):   joint DECREASES from open_pos to close_pos.
      closed_amount = open_pos - finger_val; norm = closed_amount / (open-close)

Grasp modes
-----------
kinematic (default):
    DiffIK (pose mode) drives arm to downward-pointing orientation.
    When TCP is within grasp_dist_thresh of cube AND gripper is sufficiently
    closed, a USD FixedJoint is created between gripper_attach_prim and cube.
    On release (gripper opens), the joint is removed.

physics:
    Same IK + descent, no FixedJoint. Fingers close with high stiffness;
    high-friction material applied to fingertips + cube surface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from llm_manip.contracts import Action, Pose
from llm_manip.env.base import RawObs

if TYPE_CHECKING:
    from llm_manip.robots import RobotConfig
    from llm_manip.scenes import SceneConfig

# ── Scene constants ─────────────────────────────────────────────────────────
_OBJ_HALF_H = 0.021   # cube half-height on floor/table surface (m) — matches 0.042 m side
_NUM_ENVS   = 1


def _sample_positions(
    rng: "np.random.Generator",
    obj_ids: list[str],
    spawn_bounds: tuple[float, float, float, float],
    min_dist: float,
    max_tries: int = 500,
) -> dict[str, tuple[float, float]]:
    """Sample non-overlapping (x, y) positions within spawn_bounds.

    Returns a dict {obj_id: (x, y)}.  Falls back to a deterministic grid if
    max_tries is exhausted (should never happen with reasonable bounds).
    """
    x_min, y_min, x_max, y_max = spawn_bounds
    positions: dict[str, tuple[float, float]] = {}
    for idx, oid in enumerate(obj_ids):
        for _ in range(max_tries):
            x = float(rng.uniform(x_min, x_max))
            y = float(rng.uniform(y_min, y_max))
            if all(
                np.sqrt((x - px) ** 2 + (y - py) ** 2) >= min_dist
                for px, py in positions.values()
            ):
                positions[oid] = (x, y)
                break
        else:
            # Fallback grid (centres at equal intervals)
            n = len(obj_ids)
            x_grid = x_min + (x_max - x_min) * (idx + 1) / (n + 1)
            positions[oid] = (x_grid, (y_min + y_max) / 2.0)
    return positions

# Downward-pointing EE orientation: local +Z → world -Z.
# wxyz: (w=0, x=1, y=0, z=0) = 180° around X-axis.
# quat_apply((0,1,0,0), (0,0,1)) = (0,0,-1) → tool points DOWN.
_Q_DOWN = [0.0, 1.0, 0.0, 0.0]   # wxyz

# Pre-positioning target in reset(): wrist above the workspace, tool pointing down.
_PRE_GRASP_WRIST = [0.45, 0.0, 0.50]   # scene-local coords


class IsaacEnv:
    """Robot-agnostic pick-and-place environment for Isaac Lab 2.3.0."""

    n_joints: int = 7   # updated in __init__ from robot_cfg.n_joints

    def __init__(
        self,
        robot_cfg: "RobotConfig",
        scene_cfg: Optional["SceneConfig"] = None,
        headless: bool = False,
        grasp_mode: str = "kinematic",
        seed: Optional[int] = None,
    ) -> None:
        # ── All Isaac imports here (post-AppLauncher) ──────────────────────
        import copy
        import importlib
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        from isaaclab.utils.math import quat_apply, subtract_frame_transforms

        # ── Robot CFG resolution ───────────────────────────────────────────
        _cfg_name    = robot_cfg.isaaclab_cfg
        _module_name = robot_cfg.isaaclab_module
        print(f"[IsaacEnv] loading robot cfg={_cfg_name!r} from module={_module_name!r}")
        # Defensive guard: a robot without a verified Isaac Lab CFG (isaaclab_cfg=None)
        # cannot run in Isaac Sim.  Unreachable for the currently registered robots
        # (all in ISAAC_SUPPORTED_ROBOTS), kept to fail loudly if a sim-unsupported
        # robot is ever added.
        if not _cfg_name:
            from llm_manip.robots import ISAAC_SUPPORTED_ROBOTS
            raise RuntimeError(
                f"Robot '{robot_cfg.name}' has no Isaac Lab CFG (isaaclab_cfg=None). "
                f"Isaac Sim only supports: {sorted(ISAAC_SUPPORTED_ROBOTS)}."
            )
        _robot_assets = importlib.import_module(_module_name)
        if not hasattr(_robot_assets, _cfg_name):
            raise RuntimeError(
                f"{_module_name} has no attribute '{_cfg_name}'. "
                f"Check robots.py for robot '{robot_cfg.name}'."
            )
        _base_cfg = getattr(_robot_assets, _cfg_name)
        print(f"[IsaacEnv] robot base_cfg loaded OK")

        self._torch = torch
        self._quat_apply = quat_apply
        self._subtract_frame_transforms = subtract_frame_transforms
        self._grasp_mode = grasp_mode

        # ── Store robot-specific constants ────────────────────────────────
        self._tcp_offset_z: float = robot_cfg.tcp_offset_z
        self._gripper_open_pos: float  = robot_cfg.gripper_open_pos
        self._gripper_close_pos: float = robot_cfg.gripper_close_pos
        self._gripper_close_sign: float = robot_cfg.gripper_close_sign
        self._min_grasp_finger: float  = robot_cfg.min_grasp_finger
        self._grasp_dist_thresh: float = robot_cfg.grasp_dist_thresh
        self._gripper_attach_prim: str = robot_cfg.gripper_attach_prim
        self._finger_friction_prims: list = robot_cfg.finger_friction_prims

        # ── Asset URLs ─────────────────────────────────────────────────────
        table_usd = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"

        # ── Random-spawn state ─────────────────────────────────────────────
        self._random_spawn    = bool(scene_cfg and getattr(scene_cfg, "random_spawn", False))
        self._spawn_bounds    = getattr(scene_cfg, "spawn_bounds",     (0.38, -0.22, 0.62, 0.22))
        self._min_object_dist = getattr(scene_cfg, "min_object_dist",  0.12)
        # seed=None → OS entropy (different each run); seed=int → reproducible sequence
        self._rng = np.random.default_rng(seed) if self._random_spawn else None
        if self._random_spawn:
            tag = f"seed={seed}" if seed is not None else "seed=OS-entropy (random each run)"
            print(f"[IsaacEnv] random spawn enabled, {tag}")

        # ── Object size from SceneConfig ───────────────────────────────────
        _obj_size        = float(getattr(scene_cfg, "object_size", 0.042)) if scene_cfg else 0.042
        _obj_half_h      = _obj_size / 2.0
        self._obj_half_h = _obj_half_h   # stored for grasp diagnostics and reset()

        # ── Scene objects ──────────────────────────────────────────────────
        _PARK = (100.0, 100.0, 0.0)
        scene_objs = {
            oid: (label, tuple(xyz))
            for oid, label, xyz in (scene_cfg.objects if scene_cfg else [
                ("red_cube",   "red_cube",   (0.40, 0.00, 0.0)),
                ("blue_plate", "blue_plate", (0.40, 0.30, 0.0)),
            ])
        }

        def _obj_init_pos(oid) -> tuple:
            if oid in scene_objs:
                x, y, _ = scene_objs[oid][1]
                return (x, y, _obj_half_h)
            return _PARK

        # ── Table / ground settings ────────────────────────────────────────
        _use_table = bool(scene_cfg and scene_cfg.use_table)
        _ground_z  = -1.05 if _use_table else 0.0   # hide robot base below visual horizon

        # ── Robot home pose dict (joint_name → angle) ──────────────────────
        # Arm joints only — gripper joints stay in the base config's init_state.joint_pos
        # (FRANKA_PANDA_HIGH_PD_CFG uses "panda_finger_joint.*" wildcard there; adding
        # explicit panda_finger_joint1/2 alongside it causes a "Multiple matches" ValueError)
        _home_dict = dict(zip(robot_cfg.arm_joint_names, robot_cfg.home_q.tolist()))

        robot_cfg_isaaclab = copy.deepcopy(_base_cfg)
        robot_cfg_isaaclab.prim_path = "{ENV_REGEX_NS}/Robot"
        robot_cfg_isaaclab.init_state.pos = (0.0, 0.0, 0.0)
        robot_cfg_isaaclab.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        robot_cfg_isaaclab.init_state.joint_pos.update(_home_dict)
        print(f"[IsaacEnv] articulation init_state built, arm_joints={list(_home_dict)}")

        # Resolved prim path for env_0 — used in FixedJoint creation.
        # root_physx_view.prim_paths gives the articulation ROOT (e.g. /Robot/root_joint
        # for some USD hierarchies), NOT the USD spawn root.  Using the config template
        # avoids that nesting issue.
        self._robot_prim_base = robot_cfg_isaaclab.prim_path.replace(
            "{ENV_REGEX_NS}", "/World/envs/env_0"
        )  # "/World/envs/env_0/Robot"

        # ── Dynamic InteractiveSceneCfg ────────────────────────────────────
        # Build attribute dict so the table is only included when _use_table.
        _cube_cfg = dict(
            size=(_obj_size, _obj_size, _obj_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        _scene_attrs: dict = dict(
            ground=AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, _ground_z)),
            ),
            light=AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
            ),
            robot=robot_cfg_isaaclab,
            red_cube=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RedCube",
                spawn=sim_utils.CuboidCfg(
                    **_cube_cfg,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0), metallic=0.0,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=_obj_init_pos("red_cube"), rot=(1, 0, 0, 0),
                ),
            ),
            blue_cube=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BlueCube",
                spawn=sim_utils.CuboidCfg(
                    **_cube_cfg,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.0, 1.0), metallic=0.0,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=_obj_init_pos("blue_cube"), rot=(1, 0, 0, 0),
                ),
            ),
        )
        print(f"[IsaacEnv] adding table use_table={_use_table}, ground_z={_ground_z}")
        if _use_table:
            _scene_attrs["table"] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.UsdFileCfg(usd_path=table_usd),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(0.5, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711),
                ),
            )

        print(f"[IsaacEnv] spawning objects={list(scene_objs)}, scene_attrs_keys={list(_scene_attrs)}")
        PickPlaceSceneCfg = configclass(
            type("PickPlaceSceneCfg", (InteractiveSceneCfg,), _scene_attrs)
        )
        print(f"[IsaacEnv] scene class built OK")

        # ── Simulation ─────────────────────────────────────────────────────
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self._sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.4, 0.0, 0.1])

        print(f"[IsaacEnv] instantiating PickPlaceSceneCfg num_envs={_NUM_ENVS}")
        scene_inst = PickPlaceSceneCfg(num_envs=_NUM_ENVS, env_spacing=4.0)
        print(f"[IsaacEnv] creating InteractiveScene ...")
        self._scene = InteractiveScene(scene_inst)
        print(f"[IsaacEnv] sim.reset() ...")
        self._sim.reset()
        print("[IsaacEnv] Scene loaded.")

        # ── Handles ────────────────────────────────────────────────────────
        self._robot = self._scene["robot"]
        self._obj_map: dict = {
            "red_cube":  self._scene["red_cube"],
            "blue_cube": self._scene["blue_cube"],
        }
        self._active_ids: set = set(scene_objs.keys())

        # ── Arm entity config (EE body for DiffIK) ─────────────────────────
        _ee_body  = robot_cfg.ee_link
        _arm_jnts = robot_cfg.arm_joint_names
        print(f"[IsaacEnv] resolving arm entity: joints={_arm_jnts}, ee_body={_ee_body!r}")
        self._arm_entity_cfg = SceneEntityCfg(
            "robot", joint_names=_arm_jnts, body_names=[_ee_body],
        )
        self._arm_entity_cfg.resolve(self._scene)
        self._arm_joint_ids = self._arm_entity_cfg.joint_ids

        if self._robot.is_fixed_base:
            self._ee_jacobi_idx = self._arm_entity_cfg.body_ids[0] - 1
        else:
            self._ee_jacobi_idx = self._arm_entity_cfg.body_ids[0]
        self._ee_body_idx = self._arm_entity_cfg.body_ids[0]

        # ── Gripper ────────────────────────────────────────────────────────
        _gripper_jnts = robot_cfg.gripper_joint_names
        gripper_entity = SceneEntityCfg("robot", joint_names=_gripper_jnts)
        gripper_entity.resolve(self._scene)
        self._gripper_joint_ids = gripper_entity.joint_ids
        self._n_gripper_joints  = len(self._gripper_joint_ids)

        # ── DiffIK: pose mode — drives arm to target position + orientation ─
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
        )
        self._diff_ik = DifferentialIKController(
            diff_ik_cfg, num_envs=_NUM_ENVS, device=self._sim.device
        )
        # Downward orientation tensor: (1, 4) wxyz
        self._q_down = torch.tensor([_Q_DOWN], dtype=torch.float32, device=self._sim.device)

        # ── State ──────────────────────────────────────────────────────────
        self._t = 0
        self._holding: Optional[str] = None
        self._grasp_joint_path: Optional[str] = None
        self._holding_offset_fb: Optional[object] = None
        self._scene_cfg  = scene_cfg
        self._scene_objs = scene_objs
        self._sim_dt     = self._sim.get_physics_dt()
        self._device     = self._sim.device

        self.n_joints = robot_cfg.n_joints

        # Warm-up
        for _ in range(60):
            self._scene.write_data_to_sim()
            self._sim.step()
            self._scene.update(self._sim_dt)

        # Report initial state
        ee_pose_w  = self._robot.data.body_pose_w[0, self._ee_body_idx]
        wrist_pos  = ee_pose_w[:3]
        ee_quat    = ee_pose_w[3:7].unsqueeze(0)
        z_local    = torch.tensor([[0.0, 0.0, 1.0]], device=self._device)
        ee_z       = self._quat_apply(ee_quat, z_local)[0]
        tcp_pos    = wrist_pos + self._tcp_offset_z * ee_z
        print(f"[IsaacEnv] Home pose: wrist={wrist_pos.cpu().tolist()}")
        print(f"[IsaacEnv] EE Z-world={ee_z.cpu().tolist()}, TCP={tcp_pos.cpu().tolist()}")
        print("[IsaacEnv] Ready.")

    # ====================================================================== #
    # Env Protocol                                                             #
    # ====================================================================== #

    def reset(self) -> RawObs:
        torch = self._torch

        # Reset robot joints to home
        joint_pos = self._robot.data.default_joint_pos.clone()
        joint_vel = self._robot.data.default_joint_vel.clone()
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self._robot.reset()

        # Reset objects — optionally sample random (x, y) positions
        env_origin = self._scene.env_origins
        if self._random_spawn and self._rng is not None:
            spawn_xy = _sample_positions(
                self._rng, sorted(self._active_ids),
                self._spawn_bounds, self._min_object_dist,
            )
            pos_str = "  ".join(f"{k}@({v[0]:.3f},{v[1]:.3f})" for k, v in sorted(spawn_xy.items()))
            print(f"[Scene] {pos_str}")
        else:
            spawn_xy = {}

        for oid, obj in self._obj_map.items():
            if oid in self._active_ids:
                if oid in spawn_xy:
                    x, y = spawn_xy[oid]
                else:
                    x, y, _ = self._scene_objs[oid][1]
                z = self._obj_half_h
            else:
                x, y, z = 100.0, 100.0, 0.0
            target_pos = torch.tensor([[x, y, z]], dtype=torch.float32, device=self._device)
            target_pos = target_pos + env_origin
            root_state = obj.data.default_root_state.clone()
            root_state[0, :3] = target_pos[0]
            root_state[0, 3:7] = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self._device)
            root_state[0, 7:] = 0.0
            obj.write_root_state_to_sim(root_state)
            obj.reset()

        # Clean up any lingering grasp joint
        self._destroy_grasp_joint()

        # Open gripper
        self._robot.set_joint_position_target(
            torch.full((1, self._n_gripper_joints), self._gripper_open_pos,
                       dtype=torch.float32, device=self._device),
            joint_ids=self._gripper_joint_ids,
        )

        # Initial settle
        for _ in range(30):
            self._scene.write_data_to_sim()
            self._sim.step()
            self._scene.update(self._sim_dt)

        # ── Pre-positioning: drive arm to tool-down stance above workspace ─
        pre_wrist = torch.tensor(
            [_PRE_GRASP_WRIST], dtype=torch.float32, device=self._device
        )  # (1, 3)
        pose_cmd = torch.cat([pre_wrist, self._q_down], dim=1)  # (1, 7)

        self._diff_ik.reset()
        self._diff_ik.set_command(pose_cmd)
        print("[IsaacEnv] Pre-positioning arm to tool-down stance...")

        z_local = torch.tensor([[0.0, 0.0, 1.0]], device=self._device)
        for step_i in range(500):
            jacobian = self._robot.root_physx_view.get_jacobians()[
                :, self._ee_jacobi_idx, :, self._arm_joint_ids
            ]
            ee_pose_w   = self._robot.data.body_pose_w[:, self._ee_body_idx]
            root_pose_w = self._robot.data.root_pose_w
            ee_pos_b, ee_quat_b = self._subtract_frame_transforms(
                root_pose_w[:, :3], root_pose_w[:, 3:7],
                ee_pose_w[:, :3],   ee_pose_w[:, 3:7],
            )
            joint_pos_cur = self._robot.data.joint_pos[:, self._arm_joint_ids]
            joint_pos_des = self._diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_cur)
            self._robot.set_joint_position_target(joint_pos_des, joint_ids=self._arm_joint_ids)
            self._scene.write_data_to_sim()
            self._sim.step()
            self._scene.update(self._sim_dt)

            wrist_pos_w = self._robot.data.body_pos_w[0, self._ee_body_idx]
            wrist_err = float(torch.norm(wrist_pos_w - pre_wrist[0]).item())
            ee_quat_w = self._robot.data.body_pose_w[0, self._ee_body_idx, 3:7].unsqueeze(0)
            ee_z = self._quat_apply(ee_quat_w, z_local)[0]
            orient_err = float((ee_z[2] + 1.0).abs().item())  # 0 when EE_Z=(0,0,-1)
            if wrist_err < 0.03 and orient_err < 0.10:
                print(f"[IsaacEnv] Pre-position converged at step {step_i} "
                      f"(pos_err={wrist_err:.3f}m, orient_err={orient_err:.3f}, "
                      f"EE_Z={ee_z.cpu().tolist()})")
                break
        else:
            wrist_pos_w = self._robot.data.body_pos_w[0, self._ee_body_idx]
            ee_quat_w = self._robot.data.body_pose_w[0, self._ee_body_idx, 3:7].unsqueeze(0)
            ee_z = self._quat_apply(ee_quat_w, z_local)[0]
            print(f"[IsaacEnv] Pre-position finished (500 steps). "
                  f"wrist={wrist_pos_w.cpu().tolist()}, EE_Z={ee_z.cpu().tolist()}")

        self._diff_ik.reset()
        self._t = 0
        self._holding = None
        return self.observe()

    def observe(self) -> RawObs:
        torch = self._torch

        # Arm joint positions
        joint_pos = self._robot.data.joint_pos[0, self._arm_joint_ids].cpu().numpy()

        # Gripper state normalised to [0=open, 1=closed]
        finger_val = self._robot.data.joint_pos[0, self._gripper_joint_ids[0]].item()
        gripper_norm = self._gripper_closed_frac(finger_val)

        # EE position in scene-local frame (robot-base frame)
        wrist_pos_w = self._robot.data.body_pos_w[0, self._ee_body_idx]
        env_origin  = self._scene.env_origins[0]
        ee_pos = (wrist_pos_w - env_origin).cpu().numpy()

        # GT object positions (scene-local)
        env_origin_np = env_origin.cpu().numpy()
        gt_objects = []
        for oid in self._active_ids:
            obj   = self._obj_map[oid]
            label = self._scene_objs[oid][0]
            pos_w = obj.data.root_pos_w[0].cpu().numpy()
            gt_objects.append((oid, label, (pos_w - env_origin_np).copy()))

        return RawObs(
            rgb=None, depth=None,
            joint_positions=joint_pos,
            ee_pos=ee_pos,
            gripper=gripper_norm,
            holding=self._holding,
            gt_objects=gt_objects,
            t=self._t,
        )

    def step(self, action: Action) -> RawObs:
        torch = self._torch

        # ── Arm: DiffIK pose mode ─────────────────────────────────────────
        if action.ee_target is not None:
            wrist_target = torch.tensor(
                [action.ee_target.position.tolist()],
                dtype=torch.float32, device=self._device,
            )  # (1, 3)
            pose_cmd = torch.cat([wrist_target, self._q_down], dim=1)  # (1, 7)
            self._diff_ik.set_command(pose_cmd)

            jacobian = self._robot.root_physx_view.get_jacobians()[
                :, self._ee_jacobi_idx, :, self._arm_joint_ids
            ]
            ee_pose_w   = self._robot.data.body_pose_w[:, self._ee_body_idx]
            root_pose_w = self._robot.data.root_pose_w
            ee_pos_b, ee_quat_b = self._subtract_frame_transforms(
                root_pose_w[:, :3], root_pose_w[:, 3:7],
                ee_pose_w[:, :3],   ee_pose_w[:, 3:7],
            )
            joint_pos_cur = self._robot.data.joint_pos[:, self._arm_joint_ids]
            joint_pos_des = self._diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_cur)
            self._robot.set_joint_position_target(joint_pos_des, joint_ids=self._arm_joint_ids)

        else:
            # Direct joint control (mock compatibility)
            jt = torch.tensor(
                action.joint_targets[:len(self._arm_joint_ids)].tolist(),
                dtype=torch.float32, device=self._device,
            ).unsqueeze(0)
            self._robot.set_joint_position_target(jt, joint_ids=self._arm_joint_ids)

        # ── Gripper ───────────────────────────────────────────────────────
        gripper_target = (
            self._gripper_open_pos
            + float(action.gripper) * (self._gripper_close_pos - self._gripper_open_pos)
        )
        self._robot.set_joint_position_target(
            torch.full((1, self._n_gripper_joints), gripper_target,
                       dtype=torch.float32, device=self._device),
            joint_ids=self._gripper_joint_ids,
        )

        # ── Physics step ──────────────────────────────────────────────────
        self._scene.write_data_to_sim()
        self._sim.step()
        self._scene.update(self._sim_dt)
        self._t += 1

        self._update_holding(float(action.gripper))
        return self.observe()

    def close(self) -> None:
        self._destroy_grasp_joint()

    # ====================================================================== #
    # Internal helpers                                                         #
    # ====================================================================== #

    def _gripper_closed_frac(self, finger_val: float) -> float:
        """Normalised gripper closedness in [0=open, 1=closed]."""
        if self._gripper_close_sign < 0:
            # close_sign < 0: joint goes negative when closing → use abs
            return float(np.clip(abs(finger_val) / self._gripper_close_pos, 0.0, 1.0))
        else:
            # Panda: joint decreases from open_pos to close_pos
            rng = self._gripper_open_pos - self._gripper_close_pos  # e.g. 0.04
            if rng == 0.0:
                return 0.0
            return float(np.clip((self._gripper_open_pos - finger_val) / rng, 0.0, 1.0))

    def _gripper_closed_amount(self, finger_val: float) -> float:
        """Raw closed amount (in joint units) used for grasp detection threshold."""
        if self._gripper_close_sign < 0:
            return abs(finger_val)
        else:
            return self._gripper_open_pos - finger_val

    def _get_tcp_world_pos(self) -> "torch.Tensor":
        """TCP (fingertip midpoint) in world frame = EE_body + tcp_offset * EE_Z_world."""
        ee_pose_w  = self._robot.data.body_pose_w[0, self._ee_body_idx]
        wrist_pos  = ee_pose_w[:3]
        ee_quat    = ee_pose_w[3:7].unsqueeze(0)
        z_local    = self._torch.tensor([[0.0, 0.0, 1.0]], device=self._device)
        ee_z_world = self._quat_apply(ee_quat, z_local)[0]
        return wrist_pos + self._tcp_offset_z * ee_z_world

    def _update_holding(self, gripper_cmd: float) -> None:
        """Detect grasp / release; manage FixedJoint (kinematic) or friction (physics)."""
        torch = self._torch

        # ── Release ───────────────────────────────────────────────────────
        if gripper_cmd <= 0.5 and self._holding is not None:
            print(f"[IsaacEnv t={self._t}] Released {self._holding!r}")
            self._destroy_grasp_joint()
            self._holding = None
            return

        if gripper_cmd <= 0.5 or self._holding is not None:
            return

        # ── Grasp attempt ─────────────────────────────────────────────────
        finger_val    = self._robot.data.joint_pos[0, self._gripper_joint_ids[0]].item()
        closed_amount = self._gripper_closed_amount(finger_val)
        tcp_pos       = self._get_tcp_world_pos()

        if closed_amount < self._min_grasp_finger:
            if self._t % 30 == 0:
                for oid in self._active_ids:
                    obj_pos_w = self._obj_map[oid].data.root_pos_w[0]
                    dist = float(torch.norm(tcp_pos - obj_pos_w).item())
                    print(f"[IsaacEnv t={self._t}] finger={finger_val:.3f}"
                          f" closed={closed_amount:.3f} (need>={self._min_grasp_finger:.3f}), "
                          f"TCP->{oid}={dist:.3f}m")
            return

        env_origin = self._scene.env_origins[0]
        for oid in self._active_ids:
            obj_pos_w = self._obj_map[oid].data.root_pos_w[0]
            dist = float(torch.norm(tcp_pos - obj_pos_w).item())
            # Diagnostic: TCP Z vs cube bounding box (scene-local)
            tcp_z_local   = float((tcp_pos[2]      - env_origin[2]).item())
            cube_z_local  = float((obj_pos_w[2]    - env_origin[2]).item())
            obj_half_h    = self._obj_half_h
            cube_top_z    = cube_z_local + obj_half_h
            cube_bot_z    = cube_z_local - obj_half_h
            if self._t % 10 == 0:
                print(f"[IsaacEnv t={self._t}] finger={finger_val:.3f},"
                      f" TCP→{oid}={dist:.3f}m (threshold={self._grasp_dist_thresh}m) "
                      f"| TCP_z={tcp_z_local:.3f}  cube=[bot={cube_bot_z:.3f} "
                      f"ctr={cube_z_local:.3f} top={cube_top_z:.3f}]")
            if dist < self._grasp_dist_thresh:
                self._holding = oid
                if self._grasp_mode == "kinematic":
                    self._create_grasp_joint(oid)
                else:
                    self._apply_physics_friction(oid)
                print(f"[IsaacEnv] Grasped {oid!r} dist={dist:.3f}m "
                      f"finger={finger_val:.3f} mode={self._grasp_mode} "
                      f"| TCP_z={tcp_z_local:.3f} cube_ctr={cube_z_local:.3f} "
                      f"cube_top={cube_top_z:.3f}")
                break

    def _create_grasp_joint(self, obj_id: str) -> None:
        """Create a USD FixedJoint between gripper_attach_prim and the object prim.

        local_pos0 / local_rot0 are computed from PhysX body-pose tensors (body_pose_w /
        root_pos_w) rather than UsdGeom.ComputeLocalToWorldTransform.  The USD prim
        transform is relative to the prim PIVOT which may differ from the PhysX body
        CENTER-OF-MASS frame; using stale USD pivots caused a "disjointed body transforms"
        snap warning that threw the arm when LIFT started.
        """
        try:
            import omni.usd
            from pxr import Gf, Sdf, UsdPhysics

            stage = omni.usd.get_context().get_stage()

            # Primary path from robot config's prim_path template (avoids root_joint nesting)
            gripper_path  = f"{self._robot_prim_base}/{self._gripper_attach_prim}"
            obj_prim_path = self._obj_map[obj_id].root_physx_view.prim_paths[0]
            joint_path    = "/World/GraspJoint"

            if stage.GetPrimAtPath(joint_path).IsValid():
                stage.RemovePrim(joint_path)

            gp = stage.GetPrimAtPath(gripper_path)
            # Fallback: some USD hierarchies nest links under root_joint
            if not gp.IsValid():
                alt_root     = self._robot.root_physx_view.prim_paths[0]
                alt_path     = f"{alt_root}/{self._gripper_attach_prim}"
                gp           = stage.GetPrimAtPath(alt_path)
                if gp.IsValid():
                    gripper_path = alt_path
            if not gp.IsValid():
                print(f"[IsaacEnv] WARNING: gripper prim not found: {gripper_path!r}")
                return

            op = stage.GetPrimAtPath(obj_prim_path)
            if not op.IsValid():
                print(f"[IsaacEnv] WARNING: object prim not found: {obj_prim_path!r}")
                return

            # Physics-accurate relative transform via Isaac Lab body-pose tensors.
            # body_pose_w = PhysX body center in world frame (not USD pivot) → no snap.
            torch       = self._torch
            grip_pose   = self._robot.data.body_pose_w[0, self._ee_body_idx]  # (7,) wxyz
            grip_pos_t  = grip_pose[:3].unsqueeze(0)    # (1, 3)
            grip_quat_t = grip_pose[3:7].unsqueeze(0)   # (1, 4) wxyz

            obj_pos_t  = self._obj_map[obj_id].data.root_pos_w[0].unsqueeze(0)  # (1, 3)
            obj_quat_t = torch.tensor([[1.0, 0.0, 0.0, 0.0]],
                                      dtype=torch.float32, device=self._device)

            pos_rel, quat_rel = self._subtract_frame_transforms(
                grip_pos_t, grip_quat_t, obj_pos_t, obj_quat_t
            )
            pos0 = Gf.Vec3f(*pos_rel[0].cpu().numpy().tolist())
            q    = quat_rel[0].cpu().numpy()  # wxyz
            rot0 = Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

            joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
            joint.GetBody0Rel().SetTargets([Sdf.Path(gripper_path)])
            joint.GetBody1Rel().SetTargets([Sdf.Path(obj_prim_path)])
            joint.CreateLocalPos0Attr().Set(pos0)
            joint.CreateLocalRot0Attr().Set(rot0)
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

            self._grasp_joint_path = joint_path
            print(f"[IsaacEnv] FixedJoint: {gripper_path} ↔ {obj_prim_path}"
                  f"  local_pos0={[round(v,4) for v in pos_rel[0].cpu().tolist()]}")

        except Exception as e:
            import traceback as _tb
            _tb.print_exc()
            print(f"[IsaacEnv] WARNING: FixedJoint failed ({e}). Using offset fallback.")
            tcp_pos = self._get_tcp_world_pos()
            obj_pos = self._obj_map[obj_id].data.root_pos_w[0]
            self._holding_offset_fb = (obj_pos - tcp_pos).clone()

    def _destroy_grasp_joint(self) -> None:
        if not self._grasp_joint_path:
            self._holding_offset_fb = None
            return
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            if stage.GetPrimAtPath(self._grasp_joint_path).IsValid():
                stage.RemovePrim(self._grasp_joint_path)
            print("[IsaacEnv] FixedJoint removed.")
        except Exception as e:
            print(f"[IsaacEnv] WARNING: joint removal failed ({e})")
        finally:
            self._grasp_joint_path = None
            self._holding_offset_fb = None

    def _apply_physics_friction(self, obj_id: str) -> None:
        """Apply high-friction material to fingertips + object (physics grasp mode)."""
        try:
            import omni.usd
            from pxr import UsdPhysics, UsdShade

            stage      = omni.usd.get_context().get_stage()
            robot_root = self._robot.root_physx_view.prim_paths[0]

            mat_path = "/World/HighFrictionMat"
            if not stage.GetPrimAtPath(mat_path).IsValid():
                mat_prim = stage.DefinePrim(mat_path, "Material")
                phys_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
                phys_api.CreateStaticFrictionAttr().Set(8.0)
                phys_api.CreateDynamicFrictionAttr().Set(6.0)
                phys_api.CreateRestitutionAttr().Set(0.0)

            mat  = UsdShade.Material(stage.GetPrimAtPath(mat_path))
            bind = UsdShade.Tokens.weakerThanDescendants

            for finger in self._finger_friction_prims:
                fp = stage.GetPrimAtPath(f"{robot_root}/{finger}")
                if fp.IsValid():
                    UsdShade.MaterialBindingAPI.Apply(fp).Bind(mat, bind, "physics")

            op = stage.GetPrimAtPath(self._obj_map[obj_id].root_physx_view.prim_paths[0])
            if op.IsValid():
                UsdShade.MaterialBindingAPI.Apply(op).Bind(mat, bind, "physics")

            print("[IsaacEnv] High-friction material applied (physics grasp).")
        except Exception as e:
            print(f"[IsaacEnv] WARNING: friction material failed ({e})")
