# llm-pick-and-place — Design Document

> ⚠️ **초기 설계 비전(v0).** YOLO/cuRobo/MoveIt는 의도적으로 descope됨.
> 구현된 실제 시스템은 [docs/architecture.md](architecture.md)를 참조.

## Architecture

```
instruction
    │
    ▼
Orchestrator
    │
    ├─ Env (reset / observe / step)
    │     MockEnv ←→ IsaacEnv                       # implemented
    │
    ├─ Perception (perceive: RawObs → WorldState)
    │     OraclePerception ←→ YoloPerception        # (not implemented — descoped)
    │
    ├─ Planner (plan: WorldState × str → Plan)
    │     RuleBasedPlanner ←→ LlmPlanner            # implemented
    │
    └─ SkillExecutor (run: SkillCall × WorldState → ActionResult)
          MOCK_SKILLS ←→ IK_SKILLS (DiffIK)         # implemented
                      ←→ CuroboSkill                # (not implemented — descoped)
                      ←→ MoveitSkill                # (not implemented — descoped)
```

## Data Contracts

All modules communicate exclusively through `contracts.py` types.
No module imports another module's implementation class directly.

| Type | Role |
|---|---|
| `Pose` | 3D pose (position + quaternion) |
| `ObjectState` | Perceived object with label, pose, confidence |
| `RobotState` | Joint positions, EE pose, gripper state |
| `WorldState` | Full scene snapshot at time t |
| `Plan` / `SkillCall` | Structured task plan |
| `Action` | Canonical robot command (joint targets + gripper) |
| `ActionResult` | Post-skill world state + status |

## Closed-Loop Execution

```
SkillExecutor.run(call, world):
    skill.reset(args, world)
    loop:
        if skill.status(world) != RUNNING: return ActionResult
        action = skill.act(world)
        raw    = env.step(action)
        world  = perception.perceive(raw)   # ← closed-loop update
```

## Re-plan Flow

```
Orchestrator.run():
    reset → perceive → plan
    for each step:
        executor.run(step) → FAILURE? → re-perceive + re-plan (budget)
    RunResult(success, n_replans, trace)
```
