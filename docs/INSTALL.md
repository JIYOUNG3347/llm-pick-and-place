# Installation

This guide covers two tiers:

- **Mock pipeline** — pure Python, no GPU. Runs the full perceive → plan → execute → re-plan
  loop with an analytic mock environment. Good for trying the planners and tests.
- **Isaac Sim pipeline** — physics simulation with a Franka Panda. Requires an NVIDIA GPU,
  Isaac Sim 5.1.0, and Isaac Lab 2.3.0.

---

## 1. Mock pipeline (no GPU)

### Prerequisites
- Linux (developed on Ubuntu) — also fine on macOS/Windows for the mock loop
- Python ≥ 3.11

### Steps
```bash
git clone <your-fork-url> llm-pick-and-place
cd llm-pick-and-place

# core (numpy only)
pip install -e .

# optional extras
pip install -e ".[llm]"   # OpenAI-compatible LLM planner (openai>=1.0)
pip install -e ".[ui]"    # PySide6 desktop launcher
pip install -e ".[dev]"   # pytest + pytest-cov

# smoke test
python3 scripts/run.py --instruction "put the red cube on the blue cube"
pytest -q
```

---

## 2. Isaac Sim pipeline (GPU)

### Prerequisites
- **Ubuntu** with an **NVIDIA GPU**, recent NVIDIA driver, and CUDA runtime
  (whatever Isaac Sim 5.1 requires)
- **Isaac Sim 5.1.0** — https://docs.isaacsim.omniverse.nvidia.com/
- **Isaac Lab 2.3.0** — https://isaac-sim.github.io/IsaacLab/ (install guide:
  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- Python ≥ 3.11 (the project package; Isaac Sim ships its own bundled Python)

The code imports `isaaclab`, `isaaclab_assets`, `isaaclab.sim`, etc., and uses
`FRANKA_PANDA_HIGH_PD_CFG` from `isaaclab_assets.robots.franka`. These are provided by an
Isaac Lab install — the scripts run inside Isaac Sim's bundled Python, not your system Python.

### 2a. Point the runner at your Isaac Sim install

`scripts/run_sim.py` re-executes itself inside Isaac Sim's bundled `python.sh`. It resolves
that interpreter in this order:

1. `$ISAACSIM_PYTHON` — full path to `python.sh`
2. `$ISAACSIM_PATH/python.sh` — Isaac Sim install directory
3. `~/isaacsim/python.sh` — default install location

Set whichever fits your machine, e.g.:
```bash
export ISAACSIM_PYTHON="$HOME/isaacsim/python.sh"
# or
export ISAACSIM_PATH="$HOME/isaacsim"
```
The PySide6 launcher (`scripts/launcher.py`) discovers Isaac Sim from `$ISAACSIM_PATH`,
`~/isaacsim`, or `/opt/isaacsim`.

### 2b. Install the project into Isaac Sim's Python

The pipeline modules need to be importable from Isaac Sim's bundled Python:
```bash
"$ISAACSIM_PYTHON" -m pip install -e .
# LLM planner inside Isaac Sim's Python (optional):
"$ISAACSIM_PYTHON" -m pip install openai
```

### 2c. Run
```bash
"$ISAACSIM_PYTHON" scripts/run_sim.py \
  --robot panda --scene tabletop_rb --executor ik \
  --planner llm --instruction "pick up the red cube" --seed 0
```
Add `--headless` (an Isaac Lab AppLauncher flag) to run without a viewport.

---

## 3. Ollama (local LLM planner)

The LLM planner talks to an OpenAI-compatible endpoint. The default is a local Ollama server.

```bash
# install Ollama: https://ollama.com/download
ollama serve                  # starts the server on http://localhost:11434
ollama pull qwen2.5:7b        # default model used by the project
```

Selecting the backend / model:
- `--llm-backend ollama` (default) → `http://localhost:11434/v1`, default model `qwen2.5:7b`
- `--llm-backend anthropic` → `https://api.anthropic.com/v1`, default model
  `claude-haiku-4-5-20251001` (requires `ANTHROPIC_API_KEY`)
- `--model <name>` overrides the model; the `OLLAMA_MODEL` env var also overrides it

The planner reads API keys only from the environment (`OPENAI_API_KEY`, then
`ANTHROPIC_API_KEY`, falling back to the literal `"ollama"` for local servers). No keys are
stored in the repo. If Ollama is unreachable or the LLM output is invalid, the planner falls
back to the rule-based planner instead of crashing.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ERROR: Isaac Sim python.sh not found at …` | Set `ISAACSIM_PYTHON` or `ISAACSIM_PATH` (see 2a). |
| `ModuleNotFoundError: isaaclab` | You are not in Isaac Sim's Python, or Isaac Lab isn't installed. Run via `run_sim.py` (it re-execs) and verify Isaac Lab. |
| `PySide6 is not installed` | `pip install -e ".[ui]"` |
| LLM planner falls back to rule-based | Ollama not running / model not pulled — see section 3. |
