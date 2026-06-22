#!/usr/bin/env python3
"""Isaac Lab entry point for llm-pick-and-place.

AppLauncher MUST be created before any Isaac/Omni imports.
This script auto-relaunches itself via isaaclab.sh if not already in
Isaac Lab's Python environment (detects via `pxr` availability).

Usage (recommended):
    "$ISAACSIM_PYTHON" scripts/run_sim.py [args]   # e.g. ~/isaacsim/python.sh

Usage (auto-bootstrap):
    python3 scripts/run_sim.py [args]    # re-execs via the resolved Isaac Sim Python
    python  scripts/run_sim.py [args]    # same

Isaac Sim Python is resolved from $ISAACSIM_PYTHON, else $ISAACSIM_PATH/python.sh,
else ~/isaacsim/python.sh.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

# Project root on sys.path (before any llm_manip imports)
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Isaac Sim bundled Python — bypasses VIRTUAL_ENV=/usr system artifact.
# Resolve in order:
#   1. $ISAACSIM_PYTHON   — full path to python.sh
#   2. $ISAACSIM_PATH     — Isaac Sim install dir (python.sh is appended)
#   3. ~/isaacsim/python.sh  (default install location)
def _resolve_isaac_python() -> Path:
    env_py = os.environ.get("ISAACSIM_PYTHON")
    if env_py:
        return Path(env_py).expanduser()
    env_dir = os.environ.get("ISAACSIM_PATH")
    if env_dir:
        return Path(env_dir).expanduser() / "python.sh"
    return Path.home() / "isaacsim" / "python.sh"


_ISAAC_PY = _resolve_isaac_python()


def _in_isaac_env() -> bool:
    """Return True if running inside Isaac Lab's bundled Python.

    Uses `isaaclab` package availability — reliably present in Isaac Sim's
    Python before AppLauncher is created (unlike `pxr` which needs the app).
    """
    return importlib.util.find_spec("isaaclab") is not None


if not _in_isaac_env():
    # ── Re-exec via Isaac Sim's bundled Python ────────────────────────────
    # Do NOT use isaaclab.sh — VIRTUAL_ENV=/usr system artifact causes it to
    # resolve `/usr/bin/python` instead of the bundled python.sh.
    if _ISAAC_PY.exists():
        env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "CONDA_PREFIX")}
        env["PYTHONUNBUFFERED"] = "1"   # force line-buffered output through pipes
        cmd = [str(_ISAAC_PY), "-u", str(Path(__file__).resolve())] + sys.argv[1:]
        print(f"[run_sim] Not in Isaac Lab Python — relaunching via:\n  {cmd[0]} {Path(__file__).name} ...")
        try:
            os.execvpe(cmd[0], cmd, env)   # replace current process — no orphan child
        except OSError as e:
            print(f"[run_sim] execvpe failed: {e}")
            sys.exit(1)
    else:
        print(
            f"ERROR: Isaac Sim python.sh not found at {_ISAAC_PY}\n"
            "Set ISAACSIM_PYTHON (full path to python.sh) or ISAACSIM_PATH "
            "(Isaac Sim install dir), then run:\n"
            '  "$ISAACSIM_PYTHON" scripts/run_sim.py [args]'
        )
        sys.exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
# Running inside Isaac Lab Python — AppLauncher FIRST                         #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

import argparse
import sys as _sys
# Force stdout / stderr to be unbuffered even when piped (e.g. to tee).
_sys.stdout.reconfigure(line_buffering=True)
_sys.stderr.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

from llm_manip.robots import ROBOTS
from llm_manip.scenes import SCENES

parser = argparse.ArgumentParser(description="llm-pick-and-place Isaac Sim runner")
parser.add_argument("--instruction",  default="put the red cube on the blue cube")
parser.add_argument("--robot",        default="panda", choices=list(ROBOTS))
parser.add_argument("--scene",        default="tabletop_rb", choices=list(SCENES))
parser.add_argument("--seed",         type=int, default=None,
                    help="Seed for object spawn positions. Omit for a different random "
                         "layout each run; pass an integer for reproducibility.")
parser.add_argument("--perception",   default="oracle", choices=["oracle"])
parser.add_argument("--planner",      default="rule_based", choices=["rule_based", "llm"])
parser.add_argument("--executor",     default="ik", choices=["ik", "mock"])
parser.add_argument("--grasp",        default="kinematic", choices=["kinematic", "physics"],
                    help="kinematic: FixedJoint attachment; physics: friction-based")
parser.add_argument("--max-replans",  type=int, default=3)
parser.add_argument("--strict-llm",  action="store_true",
                    help="Disable LLM→rule_based fallback; fail loudly if LLM path breaks."
                         " Use to confirm the LLM is actually being queried.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim app — MUST happen before any omni/isaac module imports
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Now safe to import everything ─────────────────────────────────────────
import atexit
import signal

from llm_manip.env.isaac_env import IsaacEnv
from llm_manip.executor import ik_skill as _ik_skill_mod
from llm_manip.executor.base import SkillExecutor
from llm_manip.factory import build_perception, build_planner, build_skills
from llm_manip.orchestrator import Orchestrator
from llm_manip.robots import get_robot
from llm_manip.scenes import get_scene

# ── Guarantee simulation_app.close() on any exit path ─────────────────────
# Without this, killing the python.sh wrapper orphans the kit/python3 child,
# leaving it alive on the GPU (see nvidia-smi Compute processes).
_app_closed = False

def _close_app() -> None:
    global _app_closed
    if not _app_closed:
        _app_closed = True
        try:
            simulation_app.close()
        except Exception:
            pass

atexit.register(_close_app)

def _signal_handler(sig, frame) -> None:
    _close_app()
    raise SystemExit(0)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT,  _signal_handler)


def main() -> None:
    robot_cfg = get_robot(args_cli.robot)
    scene_cfg = get_scene(args_cli.scene)

    # Configure IK skill with robot TCP offset and cube half-height from SceneConfig.
    _ik_skill_mod.configure(
        tcp_offset_z=robot_cfg.tcp_offset_z,
        obj_half_h=scene_cfg.object_size / 2,
    )

    env        = IsaacEnv(robot_cfg=robot_cfg, scene_cfg=scene_cfg,
                          headless=args_cli.headless,
                          grasp_mode=args_cli.grasp,
                          seed=args_cli.seed)
    perception = build_perception(args_cli.perception)
    planner    = build_planner(args_cli.planner, strict_llm=args_cli.strict_llm)
    skills     = build_skills(args_cli.executor)
    executor   = SkillExecutor(env, perception, skills)

    orchestrator = Orchestrator(
        env=env,
        perception=perception,
        planner=planner,
        executor=executor,
        max_replans=args_cli.max_replans,
    )

    result = orchestrator.run(args_cli.instruction)
    print(f"\nsuccess={result.success}  re-plans={result.n_replans}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
    finally:
        _close_app()
