#!/usr/bin/env python3
"""Generate demonstration data for imitation learning.

Runs the oracle pipeline and saves (observation, action) pairs to perception_data/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_manip.executor.base import SkillExecutor
from llm_manip.factory import build_env, build_perception, build_planner, build_skills
from llm_manip.orchestrator import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int, default=10)
    parser.add_argument("--instruction", default="put the red cube on the blue plate")
    parser.add_argument("--out",         default="perception_data/demos.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for i in range(args.n):
            env        = build_env("mock", slip_once=False)
            perception = build_perception("oracle")
            planner    = build_planner("rule_based")
            skills     = build_skills("mock")
            executor   = SkillExecutor(env, perception, skills)
            orch       = Orchestrator(env, perception, planner, executor)
            result     = orch.run(args.instruction)
            record = {"episode": i, "success": result.success, "n_steps": len(result.trace)}
            f.write(json.dumps(record) + "\n")
    print(f"Saved {args.n} episodes to {args.out}")


if __name__ == "__main__":
    main()
