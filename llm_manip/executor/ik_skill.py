"""IK-based skills for real Isaac Sim execution.

Coordinate frame
----------------
All heights in this file are in WRIST FRAME (EE body position).
IsaacEnv.observe() reports the EE body (panda_hand) as ee_pos, so
world.robot.ee_pose.position == EE body (scene-local).

When the tool points straight down (EE local Z → world -Z):
    TCP_z = wrist_z - _TCP_OFFSET_Z          (_TCP_OFFSET_Z below wrist)

The DiffIK in IsaacEnv.step() is in pose mode: it drives the wrist to the
target position AND simultaneously reorients the tool to point down.
Grasp detection uses actual TCP distance (computed inside IsaacEnv), so the
skill layer only needs to track wrist position for phase transitions.

Phase state machines
--------------------
PickSkill :  APPROACH → DESCEND → CLOSE → WAIT_GRASP → LIFT → done
PlaceSkill:  HOVER    → DESCEND → OPEN  → WAIT_RELEASE → RETREAT → done
MoveToSkill: MOVE → done
"""
from __future__ import annotations

import numpy as np

from llm_manip.contracts import Action, Pose, SkillStatus, WorldState

# Panda hand: distance from EE body (panda_hand) to fingertip midpoint (TCP).
# Overridden at runtime by configure() from the active RobotConfig.tcp_offset_z.
_TCP_OFFSET_Z = 0.107   # m

# DexCube half-height (cube centre to floor = 0.021 m)
_OBJ_HALF_H   = 0.021   # m

# ── Wrist-frame heights ───────────────────────────────────────────────────────
# APPROACH / LIFT / HOVER: wrist at 0.40 m
#   → TCP at 0.40 - 0.107 = 0.293 m  (well above cube)
_EE_APPROACH_Z = 0.40

# GRASP DESCENT: TCP must reach cube equator (0.021 m)
#   wrist_z = 0.021 + tcp_offset_z  (Panda: 0.128 m)
_EE_GRASP_Z    = _OBJ_HALF_H + _TCP_OFFSET_Z   # fallback; PickSkill overrides dynamically

_EE_LIFT_Z     = 0.40
_EE_HOVER_Z    = 0.40


def configure(tcp_offset_z: float = 0.145, obj_half_h: float = 0.021) -> None:
    """Set robot/scene constants; recomputes dependent heights.

    tcp_offset_z: wrist-to-fingertip distance (Panda hand: 0.107)
    obj_half_h:   cube half-height = object_size / 2 from SceneConfig
    """
    global _TCP_OFFSET_Z, _OBJ_HALF_H, _EE_GRASP_Z
    _TCP_OFFSET_Z = tcp_offset_z
    _OBJ_HALF_H   = obj_half_h
    _EE_GRASP_Z   = _OBJ_HALF_H + tcp_offset_z   # fallback; PickSkill overrides dynamically


# ── Tolerances ────────────────────────────────────────────────────────────────
_TOL_COARSE = 0.05   # m — approach / lift / hover
_TOL_FINE   = 0.010  # m — grasp descend (tight: wrist within 10 mm before closing)

# ── Timeouts (sim steps at 100 Hz) ───────────────────────────────────────────
# Pick budget covers APPROACH + DESCEND (tight 5 mm convergence, up to
# PickSkill._DESCEND_MAX_STEPS) + CLOSE + WAIT_GRASP + LIFT with margin.
_PICK_TIMEOUT  = 1600
_PLACE_TIMEOUT = 1200
_MOVE_TIMEOUT  = 400

# ── Gripper commands ──────────────────────────────────────────────────────────
_OPEN  = 0.0
_CLOSE = 1.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ee_dist(world: WorldState, target: np.ndarray) -> float:
    """Distance from EE body (== ee_pos after IsaacEnv.observe()) to target."""
    return float(np.linalg.norm(world.robot.ee_pose.position - target))


def _action(world: WorldState,
            tx: float, ty: float, tz: float,
            gripper: float) -> Action:
    """Build Action with ee_target=(tx,ty,tz) in wrist-frame scene-local coords."""
    return Action(
        joint_targets=world.robot.joint_positions.copy(),
        gripper=gripper,
        ee_target=Pose.from_xyz(tx, ty, tz),
    )


def _ee_place_z(tgt_z: float) -> float:
    """Wrist Z when placing: held cube lands on top of target with 15 mm clearance.

    With tool pointing down:
        cube_centre_z = wrist_z - _TCP_OFFSET_Z
    We want:
        cube_bottom_z = tgt_z + _OBJ_HALF_H + 0.015   (15 mm above plate top)
        cube_centre_z = cube_bottom_z + _OBJ_HALF_H
    → wrist_z = tgt_z + 2 * _OBJ_HALF_H + 0.015 + _TCP_OFFSET_Z
    """
    return tgt_z + 2 * _OBJ_HALF_H + 0.015 + _TCP_OFFSET_Z


# ── MoveToSkill ───────────────────────────────────────────────────────────────

class MoveToSkill:
    """Move wrist to directly above a named object."""

    def reset(self, args: dict, world: WorldState) -> None:
        self._label = args["label"]
        self._steps = 0

    def act(self, world: WorldState) -> Action:
        obj = world.find(self._label)
        if obj is None:
            return _action(world, *world.robot.ee_pose.position, world.robot.gripper)
        ox, oy, oz = obj.pose.position
        return _action(world, ox, oy, _EE_APPROACH_Z, world.robot.gripper)

    def status(self, world: WorldState) -> SkillStatus:
        self._steps += 1
        obj = world.find(self._label)
        if obj is None:
            return SkillStatus.FAILURE
        ox, oy, oz = obj.pose.position
        if _ee_dist(world, np.array([ox, oy, _EE_APPROACH_Z])) < _TOL_COARSE:
            return SkillStatus.SUCCESS
        if self._steps >= _MOVE_TIMEOUT:
            return SkillStatus.FAILURE
        return SkillStatus.RUNNING


# ── PickSkill ─────────────────────────────────────────────────────────────────

class PickSkill:
    """
    APPROACH  — wrist above cube at _EE_APPROACH_Z (0.40 m); TCP well above cube
    DESCEND   — wrist → grasp_wrist_z so TCP reaches cube CENTRE (oz)
    CLOSE     — gripper closes (50 sim steps to let joints travel)
    WAIT_GRASP— IsaacEnv detects TCP proximity + finger closure → sets holding
    LIFT      — wrist back to _EE_LIFT_Z; cube follows via FixedJoint
    """

    # Descent velocity in wrist Z: 3 mm / sim-step.
    # Commanding a slowly-descending target keeps the IK error small
    # (avoids DLS stagnation that occurs when error >> IK step size).
    _DESCEND_RATE = 0.003   # m / step

    # DESCEND→CLOSE gate. The wrist trails the commanded cmd_z (DLS IK lag ~25 mm
    # observed during motion); a loose gate fires CLOSE before the wrist actually
    # reaches grasp_wrist_z, so the TCP grips above the cube centre. Require tight
    # convergence (5 mm) so TCP lands within ±5 mm of centre before closing.
    _DESCEND_TOL = 0.005    # m — wrist must be this close to grasp_wrist_z
    # Safety net: if convergence stalls, force CLOSE after this many DESCEND steps
    # rather than running out the whole pick budget. Sized well above the typical
    # convergence time (~cmd travel ≈90 steps + catch-up), so the normal exit is
    # _DESCEND_TOL, not the budget.
    _DESCEND_MAX_STEPS = 400

    def reset(self, args: dict, world: WorldState) -> None:
        self._label = args["label"]
        self._phase = "APPROACH"
        self._steps = 0
        self._wait  = 0
        self._cmd_z: float = _EE_APPROACH_Z
        # grasp_wrist_z is set at APPROACH→DESCEND from observed cube Z so it's
        # independent of scene origin; initialised to the module-level fallback.
        self._grasp_wrist_z: float = _EE_GRASP_Z

    def act(self, world: WorldState) -> Action:
        obj = world.find(self._label)
        if obj is None:
            return _action(world, *world.robot.ee_pose.position, _OPEN)

        ox, oy, oz = obj.pose.position
        ph = self._phase

        if ph == "APPROACH":
            return _action(world, ox, oy, _EE_APPROACH_Z, _OPEN)
        elif ph == "DESCEND":
            # Velocity-controlled descent toward self._grasp_wrist_z (set at transition).
            # TCP lands at cube centre (oz) so fingers wrap around the sides, not the top.
            self._cmd_z = max(self._grasp_wrist_z, self._cmd_z - PickSkill._DESCEND_RATE)
            return _action(world, ox, oy, self._cmd_z, _OPEN)
        elif ph in ("CLOSE", "WAIT_GRASP"):
            return _action(world, ox, oy, self._grasp_wrist_z, _CLOSE)
        elif ph == "LIFT":
            ox2, oy2 = obj.pose.position[:2]
            return _action(world, ox2, oy2, _EE_LIFT_Z, _CLOSE)
        else:
            return _action(world, *world.robot.ee_pose.position, _CLOSE)

    def status(self, world: WorldState) -> SkillStatus:
        self._steps += 1
        if self._steps >= _PICK_TIMEOUT:
            print(f"[PickSkill] TIMEOUT in phase {self._phase} at step {self._steps}")
            return SkillStatus.FAILURE

        obj = world.find(self._label)
        if obj is None:
            return SkillStatus.FAILURE

        ox, oy, oz = obj.pose.position
        wrist = world.robot.ee_pose.position   # (3,) wrist position

        if self._phase == "APPROACH":
            dist = _ee_dist(world, np.array([ox, oy, _EE_APPROACH_Z]))
            if self._steps % 50 == 0:
                print(f"[PickSkill APPROACH t={self._steps}] wrist={wrist}, dist={dist:.3f}m")
            if dist < _TOL_COARSE:
                # Target: TCP (fingertip midpoint) at cube CENTRE (oz) so the fingers
                # wrap the cube's mid-height, not the top edge. With the tool pointing
                # down, TCP_z = wrist_z - _TCP_OFFSET_Z, so:
                #     grasp_wrist_z = oz + _TCP_OFFSET_Z  →  TCP_z = oz (centre)
                # (Previously this added _OBJ_HALF_H, which parked the TCP at the cube
                #  TOP — the cause of the high grip.)
                self._grasp_wrist_z = float(oz) + _TCP_OFFSET_Z
                self._cmd_z = float(wrist[2])
                self._phase = "DESCEND"
                self._wait  = 0
                _cube_top_z = float(oz) + _OBJ_HALF_H
                print(f"[PickSkill] APPROACH done → DESCEND (step {self._steps}) "
                      f"cube_z={float(oz):.3f} cube_top={_cube_top_z:.3f} "
                      f"grasp_wrist_z={self._grasp_wrist_z:.3f}")

        elif self._phase == "DESCEND":
            self._wait += 1   # per-phase step counter (reset at APPROACH→DESCEND)
            wrist_z = float(wrist[2])
            dist_z = abs(wrist_z - self._grasp_wrist_z)
            if self._steps % 30 == 0:
                print(f"[PickSkill DESCEND t={self._steps}] "
                      f"wrist_z={wrist_z:.3f} cmd_z={self._cmd_z:.3f} "
                      f"target={self._grasp_wrist_z:.3f} dist_z={dist_z:.3f}m "
                      f"(wait={self._wait}/{PickSkill._DESCEND_MAX_STEPS})")
            # Gate CLOSE on tight wrist convergence; fall back to the step budget so a
            # stalled descent still proceeds instead of burning the whole pick timeout.
            converged  = dist_z < PickSkill._DESCEND_TOL
            budget_hit = self._wait >= PickSkill._DESCEND_MAX_STEPS
            if converged or budget_hit:
                # CLOSE-entry diagnostics: where the TCP sits vs the cube box.
                tcp_z    = wrist_z - _TCP_OFFSET_Z
                cube_ctr = float(oz)
                cube_bot = cube_ctr - _OBJ_HALF_H
                cube_top = cube_ctr + _OBJ_HALF_H
                reason   = "converged" if converged else f"budget({PickSkill._DESCEND_MAX_STEPS})"
                print(f"[PickSkill] DESCEND done → CLOSE (step {self._steps}, {reason}) "
                      f"dist_z={dist_z:.4f}m | TCP_z={tcp_z:.3f} "
                      f"cube=[bot={cube_bot:.3f} ctr={cube_ctr:.3f} top={cube_top:.3f}]")
                self._phase = "CLOSE"
                self._wait  = 0

        elif self._phase == "CLOSE":
            self._wait += 1
            if self._wait >= 50:
                print(f"[PickSkill] CLOSE done → WAIT_GRASP (step {self._steps})")
                self._phase = "WAIT_GRASP"
                self._wait  = 0

        elif self._phase == "WAIT_GRASP":
            if world.robot.holding is not None:
                print(f"[PickSkill] Grasp confirmed → LIFT (step {self._steps})")
                self._phase = "LIFT"
            else:
                self._wait += 1
                if self._wait % 10 == 0:
                    print(f"[PickSkill WAIT_GRASP t={self._steps}] "
                          f"holding={world.robot.holding!r} wait={self._wait}/60")
                if self._wait >= 60:
                    print("[PickSkill] WAIT_GRASP timeout — grasp not detected")
                    return SkillStatus.FAILURE

        elif self._phase == "LIFT":
            ox2, oy2 = obj.pose.position[:2]
            dist = _ee_dist(world, np.array([ox2, oy2, _EE_LIFT_Z]))
            if self._steps % 50 == 0:
                print(f"[PickSkill LIFT t={self._steps}] wrist-dist={dist:.3f}m")
            if dist < _TOL_COARSE:
                return SkillStatus.SUCCESS

        return SkillStatus.RUNNING


# ── PlaceSkill ────────────────────────────────────────────────────────────────

class PlaceSkill:
    """
    HOVER        — wrist above target at _EE_HOVER_Z
    DESCEND      — wrist at _ee_place_z() so held cube is just above target top
    OPEN         — gripper opens (20 steps)
    WAIT_RELEASE — confirm holding == None (up to 30 steps, then force-proceed)
    RETREAT      — wrist back to hover height
    """

    def reset(self, args: dict, world: WorldState) -> None:
        self._target_label = args["target"]
        self._phase = "HOVER"
        self._steps = 0
        self._wait  = 0

    def act(self, world: WorldState) -> Action:
        tgt = world.find(self._target_label)
        if tgt is None:
            return _action(world, *world.robot.ee_pose.position, _CLOSE)

        tx, ty, tz = tgt.pose.position
        place_z = _ee_place_z(tz)
        ph = self._phase

        if ph == "HOVER":
            return _action(world, tx, ty, _EE_HOVER_Z, _CLOSE)
        elif ph == "DESCEND":
            return _action(world, tx, ty, place_z, _CLOSE)
        elif ph in ("OPEN", "WAIT_RELEASE"):
            return _action(world, tx, ty, place_z, _OPEN)
        elif ph == "RETREAT":
            return _action(world, tx, ty, _EE_HOVER_Z, _OPEN)
        else:
            return _action(world, *world.robot.ee_pose.position, _OPEN)

    def status(self, world: WorldState) -> SkillStatus:
        self._steps += 1
        if self._steps >= _PLACE_TIMEOUT:
            print(f"[PlaceSkill] TIMEOUT in phase {self._phase}")
            return SkillStatus.FAILURE

        tgt = world.find(self._target_label)
        if tgt is None:
            return SkillStatus.FAILURE

        tx, ty, tz = tgt.pose.position
        place_z = _ee_place_z(tz)

        if self._phase == "HOVER":
            dist = _ee_dist(world, np.array([tx, ty, _EE_HOVER_Z]))
            if dist < _TOL_COARSE:
                print(f"[PlaceSkill] HOVER done → DESCEND (step {self._steps})")
                self._phase = "DESCEND"
                self._wait  = 0

        elif self._phase == "DESCEND":
            dist = _ee_dist(world, np.array([tx, ty, place_z]))
            if self._steps % 50 == 0:
                print(f"[PlaceSkill DESCEND t={self._steps}] wrist-dist={dist:.3f}m")
            if dist < _TOL_FINE:
                print(f"[PlaceSkill] DESCEND done → OPEN (step {self._steps})")
                self._phase = "OPEN"
                self._wait  = 0

        elif self._phase == "OPEN":
            self._wait += 1
            if self._wait >= 20:
                self._phase = "WAIT_RELEASE"
                self._wait  = 0

        elif self._phase == "WAIT_RELEASE":
            if world.robot.holding is None:
                self._phase = "RETREAT"
            else:
                self._wait += 1
                if self._wait >= 30:
                    self._phase = "RETREAT"  # force-proceed even if joint not yet removed

        elif self._phase == "RETREAT":
            dist = _ee_dist(world, np.array([tx, ty, _EE_HOVER_Z]))
            if dist < _TOL_COARSE:
                return SkillStatus.SUCCESS

        return SkillStatus.RUNNING


IK_SKILLS: dict[str, type] = {
    "move_to": MoveToSkill,
    "pick":    PickSkill,
    "place":   PlaceSkill,
}
