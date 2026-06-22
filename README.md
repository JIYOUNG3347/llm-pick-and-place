# llm-pick-and-place

**Natural-language pick-and-place for a Franka Panda in NVIDIA Isaac Sim, driven by an LLM
tool-calling agent.**

Type a command like *"pick up the red cube"* or *"put the red cube on the blue cube"* and the
pipeline perceives the scene, asks an LLM agent to plan a sequence of skills, and executes them
in physics with closed-loop re-planning on failure.

The codebase is built **contract-first**: every stage (Env / Perception / Planner / Executor)
communicates only through the dataclasses in [`contracts.py`](llm_manip/contracts.py), so any
implementation can be swapped without touching the others. The whole loop runs either in a pure
Python mock (no GPU) or in Isaac Sim with a real physics Panda.

---

## Demo

```bash
# Isaac Sim — LLM agent plans, DiffIK executes, FixedJoint grasp
"$ISAACSIM_PYTHON" scripts/run_sim.py \
  --robot panda --scene tabletop_rb --executor ik \
  --planner llm --instruction "pick up the red cube" --seed 0
# → ... success=True  re-plans=<n>
```

![Pick-and-place demo](docs/demo.png)

No GPU? The full pipeline also runs in a mock environment:

```bash
python3 scripts/run.py --instruction "put the red cube on the blue cube"
# → success=True  re-plans=1     (a built-in slip on the first attempt triggers a re-plan)
```

---

## Key features

- **Contract-first modular pipeline** — `Env`, `Perception`, `Planner`, `Executor` are Python
  Protocols wired by a [factory](llm_manip/factory.py); each is independently swappable.
- **LLM tool-calling agent planner** — the planner ([`planner/llm.py`](llm_manip/planner/llm.py))
  runs a tool-calling agent loop (the model queries scene tools, then calls `pick`/`place`), with
  graceful fallbacks: tool-calling → position-prompt JSON → rule-based. Validates every step's
  skill name and object label against the live scene before returning a plan.
- **Robot-agnostic design** — all robot specifics (joint names, EE link, gripper convention, TCP
  offset, Isaac Lab CFG) live in a single `RobotConfig` ([`robots.py`](llm_manip/robots.py));
  **Franka Panda is the reference implementation**.
- **Isaac Sim 5.1 / Isaac Lab 2.3.0 physics** — Differential IK pose control with a downward tool
  orientation, and a kinematic **FixedJoint** grasp (TCP-proximity + gripper-closure detection).
- **PySide6 desktop launcher** — native Qt app with preflight checks (GPU / Isaac Sim / Ollama),
  presets, and a live log stream; never imports Isaac modules itself.
- **Ablation evaluation harness** — [`scripts/eval.py`](scripts/eval.py) scores planner output
  against ground-truth step sequences across instruction difficulties (literal / synonym /
  colour-swap / spatial), no simulation required.

---

## Architecture

`Orchestrator.run(instruction)` drives a perceive → plan → execute → re-plan loop: perception
turns raw observations into a `WorldState`, the planner emits a `Plan` of `SkillCall`s, and the
`SkillExecutor` runs each skill as a closed loop (`act → env.step → perceive`). On a skill
failure it re-perceives and re-plans up to `--max-replans` times.

See **[docs/architecture.md](docs/architecture.md)** for the full data-flow diagrams, the LLM
3-path fallback, and the pick/place state machines.

**Interactive diagrams:** [docs/dataflow.html](docs/dataflow.html) (perceive → plan → execute
data flow) and [docs/pick_algorithm.html](docs/pick_algorithm.html) (pick/place state machine) —
open locally in a browser, or via GitHub Pages.

---

## Usage

Install (see **[docs/INSTALL.md](docs/INSTALL.md)** for the complete guide, including Isaac Sim
and Ollama setup):

```bash
pip install -e .            # core (numpy only)
pip install -e ".[llm]"     # OpenAI-compatible LLM planner
pip install -e ".[ui]"      # PySide6 launcher
pip install -e ".[dev]"     # pytest
```

**Mock runner** ([`scripts/run.py`](scripts/run.py)) — full loop, no Isaac:
```bash
python3 scripts/run.py \
  --instruction "put the red cube on the blue cube" \
  --robot panda --scene tabletop_rb \
  --planner rule_based            # rule_based | llm
# LLM planner against a local Ollama model:
python3 scripts/run.py --planner llm --llm-backend ollama --model qwen2.5:7b \
  --instruction "put the red cube on the blue cube"
```

**Isaac Sim runner** ([`scripts/run_sim.py`](scripts/run_sim.py)) — physics Panda:
```bash
"$ISAACSIM_PYTHON" scripts/run_sim.py \
  --robot panda --scene tabletop_rb \
  --executor ik --planner llm --grasp kinematic \
  --seed 0 --instruction "pick up the red cube"
```
Options: `--executor {ik,mock}`, `--planner {rule_based,llm}`,
`--grasp {kinematic,physics}`, `--seed <int>`, plus Isaac Lab AppLauncher flags such as
`--headless`.
> `run_sim.py` re-executes itself inside Isaac Sim's bundled Python, resolved from
> `$ISAACSIM_PYTHON` → `$ISAACSIM_PATH/python.sh` → `~/isaacsim/python.sh`.

**Desktop launcher** ([`scripts/launcher.py`](scripts/launcher.py)):
```bash
pip install -e ".[ui]"
bash run_app.sh                 # or: python3 scripts/launcher.py
```

![Launcher](docs/launcher.png)

**Ablation evaluation** ([`scripts/eval.py`](scripts/eval.py)) — planner correctness, no sim:
```bash
python3 scripts/eval.py --planners rule_based llm:qwen2.5:7b --runs 3
# writes results/ablation.csv
```

> The rule-based planner only parses `put/move/place/stack the <X> on/onto the <Y>`; free-form
> instructions such as *"pick up the nearest cube"* are handled by the LLM planner.

Run the tests:
```bash
pytest -q
```

---

## Project structure

```
llm_manip/
├── contracts.py          # shared dataclasses (Pose, WorldState, Plan, Action, …)
├── robots.py             # RobotConfig + ROBOTS{panda} + ISAAC_SUPPORTED_ROBOTS
├── scenes.py             # SceneConfig + SCENES{tabletop_rb}
├── orchestrator.py       # perceive → plan → execute → re-plan loop
├── factory.py            # name → implementation builders
├── env/
│   ├── base.py           # Env Protocol + RawObs
│   ├── mock_env.py       # analytic-kinematics mock environment
│   └── isaac_env.py      # Isaac Sim env (DiffIK + FixedJoint grasp)
├── perception/
│   ├── base.py           # Perception Protocol
│   └── oracle.py         # ground-truth perception (upper-bound baseline)
├── planner/
│   ├── base.py           # Planner Protocol + SKILL_SCHEMA
│   ├── rule_based.py     # regex keyword planner (also the LLM fallback)
│   └── llm.py            # LLM tool-calling agent planner (3-path fallback)
└── executor/
    ├── base.py           # Skill Protocol + SkillExecutor (closed loop)
    ├── mock_skills.py    # analytic pick/place/move_to
    └── ik_skill.py       # Isaac Sim DiffIK pick/place/move_to state machines
scripts/   run.py · run_sim.py · launcher.py · eval.py · generate_data.py
tests/     contracts · mock_loop · robots_scenes · llm_planner · isaac_integration
docs/      architecture.md · design.md · INSTALL.md
```

---

## Limitations (honest scope)

This is a research/portfolio pipeline, not a production stack:

- **Perception is oracle-only** — `OraclePerception` uses ground-truth object poses
  (`confidence=1.0`); there is no camera/vision-based detector.
- **Execution has no collision avoidance or motion planning** — the IK executor is Differential
  IK (`ik_skill.py`); it can fail to converge near joint limits or obstacles.
- **Grasping is a kinematic FixedJoint** — on detection, a USD FixedJoint binds the object to the
  gripper. A friction-based `physics` grasp mode exists but is unvalidated.
- **One robot, one scene** — only `panda` is registered in `ROBOTS`, and only `tabletop_rb`
  (red_cube + blue_cube on a table, randomized spawn) in `SCENES`. The design is robot/scene-
  agnostic, so adding more is a config entry, not a code change.

---

## License

MIT — see [LICENSE](LICENSE).
