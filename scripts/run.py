#!/usr/bin/env python3
"""Run the manipulation pipeline end-to-end."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_manip.executor.base import SkillExecutor
from llm_manip.factory import build_env, build_perception, build_planner, build_skills
from llm_manip.orchestrator import Orchestrator
from llm_manip.robots import ROBOTS
from llm_manip.scenes import SCENES


def main() -> None:
    parser = argparse.ArgumentParser(description="llm-pick-and-place runner")
    parser.add_argument("--instruction",   default="put the red cube on the blue cube")
    parser.add_argument("--robot",         default="panda",        choices=list(ROBOTS))
    parser.add_argument("--scene",         default="tabletop_rb",  choices=list(SCENES))
    parser.add_argument("--perception",    default="oracle",     choices=["oracle"])
    parser.add_argument("--planner",       default="rule_based", choices=["rule_based", "llm"])
    parser.add_argument("--llm-backend",   default="ollama",     choices=["ollama", "anthropic"],
                        dest="llm_backend")
    parser.add_argument("--model",         default=None,
                        help="LLM model name (default: qwen2.5:7b for ollama, "
                             "claude-haiku-4-5-20251001 for anthropic)")
    parser.add_argument("--executor",      default="mock",       choices=["mock"])
    parser.add_argument("--max-replans",   type=int, default=3)
    args = parser.parse_args()

    from llm_manip.robots import get_robot
    from llm_manip.scenes import get_scene

    robot  = get_robot(args.robot)
    scene  = get_scene(args.scene)

    env        = build_env("mock", robot=robot, scene=scene)
    perception = build_perception(args.perception)
    planner    = build_planner(args.planner, llm_backend=args.llm_backend, model=args.model)
    skills     = build_skills(args.executor)
    executor   = SkillExecutor(env, perception, skills)

    orchestrator = Orchestrator(
        env=env,
        perception=perception,
        planner=planner,
        executor=executor,
        max_replans=args.max_replans,
    )

    result = orchestrator.run(args.instruction)
    print(f"\nsuccess={result.success}  re-plans={result.n_replans}")


if __name__ == "__main__":
    main()
