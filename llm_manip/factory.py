from __future__ import annotations

from typing import Optional


def build_env(
    name: str,
    robot=None,         # RobotConfig | None
    scene=None,         # SceneConfig | None
    grasp_mode: str = "kinematic",
    **kwargs,
):
    """Instantiate an Env by name."""
    if name == "mock":
        from llm_manip.env.mock_env import MockEnv
        return MockEnv(robot=robot, scene=scene, **kwargs)
    if name == "isaac":
        from llm_manip.env.isaac_env import IsaacEnv
        return IsaacEnv(robot_cfg=robot, scene_cfg=scene, grasp_mode=grasp_mode, **kwargs)
    raise ValueError(f"Unknown env: {name!r}. Options: mock, isaac")


def build_perception(name: str, **kwargs):
    """Instantiate a Perception by name."""
    if name == "oracle":
        from llm_manip.perception.oracle import OraclePerception
        return OraclePerception(**kwargs)
    raise ValueError(f"Unknown perception: {name!r}. Options: oracle")


def build_planner(
    name: str,
    llm_backend: str = "ollama",
    model: Optional[str] = None,
    strict_llm: bool = False,
    **kwargs,
):
    """Instantiate a Planner by name.

    For name='llm', llm_backend selects the endpoint:
      'ollama'    → http://localhost:11434/v1  (default model: qwen2.5:7b)
      'anthropic' → https://api.anthropic.com/v1  (default model: claude-haiku-4-5-20251001)

    strict_llm: when True, LlmPlanner raises instead of falling back to RuleBasedPlanner.
    """
    if name == "rule_based":
        from llm_manip.planner.rule_based import RuleBasedPlanner
        return RuleBasedPlanner(**kwargs)
    if name == "llm":
        from llm_manip.planner.llm import LlmPlanner
        if llm_backend == "anthropic":
            return LlmPlanner(
                base_url="https://api.anthropic.com/v1",
                model=model or "claude-haiku-4-5-20251001",
                hosted=True,
                strict_llm=strict_llm,
            )
        # Default: Ollama — model overridden by OLLAMA_MODEL env var inside LlmPlanner
        return LlmPlanner(
            base_url="http://localhost:11434/v1",
            model=model or "qwen2.5:7b",
            hosted=False,
            strict_llm=strict_llm,
        )
    raise ValueError(f"Unknown planner: {name!r}. Options: rule_based, llm")


def build_skills(name: str) -> dict[str, type]:
    """Return skill registry by name."""
    if name == "mock":
        from llm_manip.executor.mock_skills import MOCK_SKILLS
        return MOCK_SKILLS
    if name == "ik":
        from llm_manip.executor.ik_skill import IK_SKILLS
        return IK_SKILLS
    raise ValueError(f"Unknown executor: {name!r}. Options: mock, ik")
