#!/usr/bin/env python3
"""Plan-quality ablation across planners and instruction difficulty.

Evaluates planners on plan correctness only — no simulation needed.
World state is built from MockEnv + OraclePerception, then planner.plan()
is called and the output is compared against a ground-truth step sequence.

LLM planners run in strict mode (no rule_based fallback) so scores reflect
the raw LLM output, not a safety net.

Usage:
    # single planner
    python3 scripts/eval.py --planners rule_based
    python3 scripts/eval.py --planners llm:qwen2.5:7b --runs 3

    # comparison table (shows capability spread)
    python3 scripts/eval.py --planners rule_based llm:qwen2.5:7b llm:llama3.1:8b --runs 3
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_manip.contracts import SkillCall
from llm_manip.factory import build_env, build_perception, build_planner
from llm_manip.scenes import get_scene

SC = SkillCall   # alias for compact eval-case definitions


# ── ground-truth evaluation cases ────────────────────────────────────────────

@dataclass
class EvalCase:
    id: str
    instruction: str
    scene: str
    expected: list[SkillCall]
    compare: str   # "ordered" | "set"
    note: str = ""


EVAL_CASES: list[EvalCase] = [
    # ── literal ───────────────────────────────────────────────────────────────
    # Baseline: exact object labels. Both rule_based and LLM should pass.
    EvalCase(
        id="literal",
        instruction="put the red cube on the blue cube",
        scene="tabletop_rb",
        expected=[
            SC("pick",  {"label": "red_cube"}),
            SC("place", {"label": "red_cube", "target": "blue_cube"}),
        ],
        compare="ordered",
        note="direct literal instruction; baseline sanity check",
    ),

    # ── synonym ───────────────────────────────────────────────────────────────
    # Tests lexical understanding: "block"→cube synonyms.
    # rule_based: produces box/block labels (not in scene) → FAIL.
    # LLM: must infer synonym from scene context.
    EvalCase(
        id="synonym",
        instruction="put the red block on the blue block",
        scene="tabletop_rb",
        expected=[
            SC("pick",  {"label": "red_cube"}),
            SC("place", {"label": "red_cube", "target": "blue_cube"}),
        ],
        compare="ordered",
        note="block/box synonyms for cube; rule_based FAIL expected",
    ),

    # ── colour_swap ───────────────────────────────────────────────────────────
    # Tests that LLM respects colour when the instruction inverts the default.
    # "move the blue one onto the red one" — opposite of place_targets default.
    EvalCase(
        id="colour_swap",
        instruction="move the blue cube onto the red cube",
        scene="tabletop_rb",
        expected=[
            SC("pick",  {"label": "blue_cube"}),
            SC("place", {"label": "blue_cube", "target": "red_cube"}),
        ],
        compare="ordered",
        note="colour disambiguation: inverts default source/target",
    ),

    # ── spatial ───────────────────────────────────────────────────────────────
    # Tests spatial reasoning via tool-calling (LLM agent path).
    # rule_based: no spatial pattern → ValueError → FAIL.
    # LLM: should call get_object_position to resolve nearest/farthest.
    EvalCase(
        id="spatial",
        instruction="pick up the nearest cube to the robot and place it on the other one",
        scene="tabletop_rb",
        expected=[
            SC("pick",  {"label": "red_cube"}),   # depends on spawn; red_cube is typical nearest
            SC("place", {"label": "red_cube", "target": "blue_cube"}),
        ],
        compare="ordered",
        note="spatial ref resolved via get_object_position; rule_based FAIL expected",
    ),
]


# ── comparison helpers ───────────────────────────────────────────────────────

def _step_key(s: SkillCall) -> tuple:
    return (s.skill, tuple(sorted(s.args.items())))


def _correct(produced: list[SkillCall], expected: list[SkillCall], compare: str) -> bool:
    if compare == "ordered":
        if len(produced) != len(expected):
            return False
        return all(p.skill == e.skill and p.args == e.args
                   for p, e in zip(produced, expected))
    # set: both skill AND args must match, order irrelevant
    return ({_step_key(s) for s in produced} ==
            {_step_key(s) for s in expected})


def _steps_repr(steps: list[SkillCall]) -> str:
    return "[" + ", ".join(
        f"{s.skill}({', '.join(f'{k}={v}' for k,v in s.args.items())})"
        for s in steps
    ) + "]"


# ── world builder ────────────────────────────────────────────────────────────

def _world_for_scene(scene_name: str):
    """Build a WorldState from MockEnv + OraclePerception (no simulation)."""
    scene_cfg = get_scene(scene_name)
    env  = build_env("mock", scene=scene_cfg, slip_once=False)
    raw  = env.reset()
    return build_perception("oracle").perceive(raw)


# ── planner factory ──────────────────────────────────────────────────────────

def _parse_spec(spec: str) -> tuple[str, Optional[str]]:
    """'llm:qwen2.5:7b' → ('llm', 'qwen2.5:7b').  'rule_based' → ('rule_based', None)."""
    if spec == "rule_based":
        return "rule_based", None
    if spec.startswith("llm"):
        model = spec.split(":", 1)[1] if ":" in spec else None
        return "llm", model
    raise ValueError(f"Unknown planner spec {spec!r}. Use 'rule_based' or 'llm[:model]'.")


def _make_planner(spec: str):
    kind, model = _parse_spec(spec)
    if kind == "rule_based":
        return build_planner("rule_based")
    # Strict mode: RuntimeError instead of rule_based fallback → raw LLM score
    return build_planner("llm", model=model, strict_llm=True)


# ── single evaluation run ────────────────────────────────────────────────────

def _run_one(planner, world, case: EvalCase) -> tuple[bool, str]:
    """Return (correct, note_str).  Never raises."""
    try:
        plan = planner.plan(world, case.instruction)
        ok   = _correct(plan.steps, case.expected, case.compare)
        if ok:
            return True, "ok"
        got = _steps_repr(plan.steps)
        exp = _steps_repr(case.expected)
        return False, f"got {got[:120]}"
    except RuntimeError as exc:
        # strict-mode LLM failure
        return False, f"llm-strict: {str(exc)[:100]}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:100]}"


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan-quality ablation: correctness of planner output "
                    "vs. ground-truth steps."
    )
    parser.add_argument(
        "--planners", nargs="+", default=["rule_based"],
        metavar="SPEC",
        help=(
            "One or more planner specs: 'rule_based' | 'llm' | 'llm:MODEL'. "
            "Example: --planners rule_based llm:qwen2.5:7b llm:llama3.1:8b"
        ),
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Trials per (planner × case). Averages LLM stochasticity. Default: 1.",
    )
    parser.add_argument(
        "--out", default="results/ablation.csv",
        help="CSV output path. Default: results/ablation.csv",
    )
    args = parser.parse_args()

    # Pre-build world states once per scene (deterministic, safe to reuse)
    world_cache: dict[str, object] = {}
    for case in EVAL_CASES:
        if case.scene not in world_cache:
            world_cache[case.scene] = _world_for_scene(case.scene)

    rows: list[dict] = []
    # {spec: {case_id: [correct, ...]}}
    all_results: dict[str, dict[str, list[bool]]] = {}

    case_ids = [c.id for c in EVAL_CASES]

    for spec in args.planners:
        print(f"\n{'═'*64}")
        print(f"  Planner: {spec}")
        print(f"{'═'*64}")

        planner = _make_planner(spec)
        spec_res: dict[str, list[bool]] = {c.id: [] for c in EVAL_CASES}

        for case in EVAL_CASES:
            world = world_cache[case.scene]
            for run_i in range(args.runs):
                ok, note = _run_one(planner, world, case)
                spec_res[case.id].append(ok)
                rows.append({
                    "planner": spec,
                    "case":    case.id,
                    "run":     run_i,
                    "correct": int(ok),
                    "note":    note[:160],
                })
                icon    = "✓" if ok else "✗"
                run_tag = f"[{run_i+1}/{args.runs}] " if args.runs > 1 else ""
                print(f"  {icon} {run_tag}{case.id:<13} {note[:80]}")

        all_results[spec] = spec_res

        # per-planner summary line
        rates = {cid: sum(v) / len(v) for cid, v in spec_res.items()}
        avg   = sum(rates.values()) / len(rates)
        print()
        hdr  = " ".join(f"{cid:>10}" for cid in case_ids)
        vals = " ".join(f"{rates[cid]:>9.0%}" for cid in case_ids)
        print(f"  {'':20}{hdr}     avg")
        print(f"  {spec:<20}{vals}     {avg:.0%}")

    # ── cross-planner summary table ───────────────────────────────────────────
    if len(args.planners) > 1:
        print(f"\n{'═'*64}")
        print("  SUMMARY — success rate  (planner × instruction difficulty)")
        print(f"{'═'*64}")
        col_w = max(10, max(len(cid) for cid in case_ids) + 2)
        hdr   = f"  {'planner':<22}" + "".join(f"{cid:>{col_w}}" for cid in case_ids) + f"{'avg':>8}"
        print(hdr)
        print(f"  {'-'*22}" + "-" * (col_w * len(case_ids) + 8))
        for spec in args.planners:
            spec_res = all_results[spec]
            rates    = {cid: sum(v) / len(v) for cid, v in spec_res.items()}
            avg      = sum(rates.values()) / len(rates)
            cells    = "".join(f"{rates[cid]:>{col_w}.0%}" for cid in case_ids)
            print(f"  {spec:<22}{cells}{avg:>8.0%}")

    # ── CSV output ────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["planner", "case", "run", "correct", "note"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV → {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
