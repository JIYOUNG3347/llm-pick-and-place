"""LLM-based planner via OpenAI-compatible API (Ollama or hosted).

Plan strategy (tried in order, no crash on failure):
  1. Tool-calling agent loop  — LLM queries scene tools then calls pick/place.
  2. Position-prompt JSON     — positions embedded in system prompt, JSON output.
  3. rule_based fallback.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from llm_manip.contracts import Plan, SkillCall, WorldState
from llm_manip.planner.base import SKILL_SCHEMA

# ── Path-2 (prompt) JSON schema & tool def ───────────────────────────────────

_PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "enum": list(SKILL_SCHEMA)},
                    "args":  {"type": "object"},
                },
                "required": ["skill", "args"],
            },
        }
    },
    "required": ["steps"],
}

_TOOL_DEF: dict = {
    "type": "function",
    "function": {
        "name": "plan",
        "description": "Return a robot manipulation plan as a sequence of skill calls.",
        "parameters": _PLAN_SCHEMA,
    },
}

_SYSTEM_TEMPLATE = """\
You are a robot manipulation planner. Return ONLY a JSON object in this exact format:

{{"steps": [{{"skill": "pick", "args": {{"label": "red_cube"}}}}, {{"skill": "place", "args": {{"label": "red_cube", "target": "blue_cube"}}}}]}}

Available skills: {skills}
Objects in scene (label, world position in metres x,y,z): {objects_with_pos}
Robot base at (0, 0, 0).

Rules:
- Use ONLY the exact skill names and object labels listed above.
- "pick"    needs args: {{"label": "<object>"}}.
- "place"   needs args: {{"label": "<object>", "target": "<destination>"}}.
- "move_to" needs args: {{"label": "<object>"}}.
- If the instruction uses spatial relations (nearest/closest/farthest/leftmost/rightmost/between), use the positions to choose the correct object. Output the chosen exact label.
- Return ONLY the JSON object — no markdown, no explanation.
"""

# ── Path-1 (agent) tool definitions ──────────────────────────────────────────

_AGENT_SYSTEM = """\
You are a robot manipulation agent. Use the provided tools to plan manipulation tasks.

Instructions may be in Korean or English. Map Korean object references to English labels:
  빨간/빨강/적색 + 박스/상자/큐브 → red_cube
  파란/파랑/청색 + 박스/상자/큐브 → blue_cube
Always call tools with exact English labels (e.g. "red_cube", "blue_cube").

Workflow:
1. Call list_objects() to see what is in the scene.
2. Call get_object_position(query) to resolve object references or compute spatial relations.
3. Call get_robot_state() if you need the robot end-effector position for spatial reasoning.
4. Call pick(object) and place(object, target) to build the manipulation plan.

Always use the exact object labels returned by the scene tools.
"""

_AGENT_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_objects",
            "description": "List all object labels currently visible in the scene.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_object_position",
            "description": (
                "Return the world position (x, y, z in metres) of an object. "
                "Accepts free-form descriptions such as 'white box' or 'nearest cube'; "
                "normalises to the exact scene label automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Object name or description (e.g. 'white box', 'red cube').",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_robot_state",
            "description": (
                "Return the robot's current end-effector pose (position + quaternion), "
                "joint positions, and gripper state."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pick",
            "description": "Pick up an object. Appends a pick step to the manipulation plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object": {
                        "type": "string",
                        "description": "Exact scene label of the object to pick up.",
                    }
                },
                "required": ["object"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place",
            "description": "Place the held object onto a target. Appends a place step to the plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object": {
                        "type": "string",
                        "description": "Exact scene label of the object being placed.",
                    },
                    "target": {
                        "type": "string",
                        "description": "Exact scene label of the destination object or surface.",
                    },
                },
                "required": ["object", "target"],
            },
        },
    },
]

_MAX_AGENT_ITERS = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_json(text: str) -> str:
    """Strip markdown fences and extract the outermost JSON object."""
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.startswith("```")
        ).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end >= start:
        text = text[start : end + 1]
    return text


# Korean → English token substitutions applied before label matching.
_KR_MAP: dict[str, str] = {
    "빨간": "red", "빨강": "red", "적색": "red",
    "파란": "blue", "파랑": "blue", "청색": "blue",
    "박스": "cube", "상자": "cube", "큐브": "cube",
}


def _normalize_label(text: str, valid_labels: list[str]) -> Optional[str]:
    """Map a model-produced label (English or Korean) to a valid scene label, or None.

    Steps: lowercase → Korean substitution → space-to-underscore → exact / substring match.
    """
    t = text.strip().lower()
    for kr, en in _KR_MAP.items():
        t = t.replace(kr, en)
    t = t.replace(" ", "_")
    for label in valid_labels:
        if t == label.lower().replace(" ", "_"):
            return label
    for label in valid_labels:
        lc = label.lower().replace(" ", "_")
        if t in lc or lc in t:
            return label
    return None


def _plan_str(steps: list[SkillCall]) -> str:
    """Human-readable one-liner: 'pick red_cube -> place blue_plate'."""
    parts = []
    for s in steps:
        if s.skill == "pick":
            parts.append(f"pick {s.args.get('label', '?')}")
        elif s.skill == "place":
            parts.append(f"place {s.args.get('target', s.args.get('label', '?'))}")
        else:
            parts.append(f"{s.skill}({s.args})")
    return " -> ".join(parts)


# ── LlmPlanner ────────────────────────────────────────────────────────────────

class LlmPlanner:
    """LLM planner using an OpenAI-compatible API (Ollama by default).

    Model selection priority:
      1. OLLAMA_MODEL env var  (set by the launcher UI)
      2. `model` constructor argument
      3. Default "qwen2.5:7b"
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
        hosted: bool = False,
        strict_llm: bool = False,
    ) -> None:
        self._model    = os.environ.get("OLLAMA_MODEL") or model
        self._base_url = base_url
        self._hosted   = hosted
        self._strict   = strict_llm
        self._client   = None
        self._no_openai: Optional[str] = None

        print(f"[LlmPlanner] using model={self._model} base_url={base_url}")

        try:
            from openai import OpenAI
            api_key = (
                os.environ.get("OPENAI_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
                or "ollama"
            )
            self._client = OpenAI(base_url=base_url, api_key=api_key)
        except ImportError as exc:
            self._no_openai = str(exc)
            print(f"[LlmPlanner] WARNING: openai not available ({exc})")

        from llm_manip.planner.rule_based import RuleBasedPlanner
        self._fallback_planner = RuleBasedPlanner()

    # ── Public API ────────────────────────────────────────────────────────────

    def plan(self, world: WorldState, instruction: str) -> Plan:
        labels = [o.label for o in world.objects]

        if self._client is None:
            return self._do_fallback(world, instruction,
                                     f"openai 미설치: {self._no_openai}")

        # ── Path 1: Tool-calling agent loop ──────────────────────────────────
        agent_plan = self._run_agent_loop(world, instruction, labels)
        if agent_plan is not None:
            print(f"[LlmPlanner] plan: {_plan_str(agent_plan.steps)}")
            return agent_plan

        # ── Path 2: Position-prompt JSON output ───────────────────────────────
        print(f"[LlmPlanner] agent loop yielded no plan — trying prompt path")
        objects_with_pos = ", ".join(
            f"{o.label} at ({o.pose.position[0]:.2f},{o.pose.position[1]:.2f},{o.pose.position[2]:.2f})"
            for o in world.objects
        )
        system_msg = _SYSTEM_TEMPLATE.format(
            skills=", ".join(SKILL_SCHEMA),
            objects_with_pos=objects_with_pos,
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": instruction},
        ]

        print(f"[LlmPlanner] querying ollama/{self._model} (prompt path)...")

        try:
            raw = self._invoke(messages)
        except Exception as exc:
            return self._do_fallback(world, instruction, f"연결오류: {exc}")

        preview = (json.dumps(raw) if isinstance(raw, dict) else str(raw))[:200]
        print(f"[LlmPlanner] raw: {preview}")

        plan = self._validate(raw, labels)
        if plan is not None:
            print(f"[LlmPlanner] plan: {_plan_str(plan.steps)}")
            return plan

        # Classify failure for the retry feedback message
        bad_skills = [
            s.get("skill") for s in raw.get("steps", [])
            if s.get("skill") not in SKILL_SCHEMA
        ]
        all_arg_labels = [
            v for s in raw.get("steps", [])
            for k, v in s.get("args", {}).items()
            if k in ("label", "target")
        ]
        bad_labels = [v for v in all_arg_labels
                      if _normalize_label(v, labels) is None]

        if not raw.get("steps"):
            reason_tag = "빈plan"
        elif bad_skills:
            reason_tag = f"JSON무효(skills={bad_skills})"
        else:
            reason_tag = f"라벨매핑실패({bad_labels})"

        feedback = (
            f"Invalid plan. "
            f"Unknown skills: {bad_skills}. "
            f"Unknown labels: {bad_labels}. "
            f"Valid skills: {list(SKILL_SCHEMA)}. "
            f"Valid labels (copy exactly): {labels}."
        )
        messages.append({"role": "assistant", "content": json.dumps(raw)})
        messages.append({"role": "user",      "content": feedback})

        try:
            raw2 = self._invoke(messages)
        except Exception as exc:
            return self._do_fallback(world, instruction, f"연결오류(retry): {exc}")

        preview2 = (json.dumps(raw2) if isinstance(raw2, dict) else str(raw2))[:200]
        print(f"[LlmPlanner] raw(retry): {preview2}")

        plan2 = self._validate(raw2, labels)
        if plan2 is not None:
            print(f"[LlmPlanner] plan: {_plan_str(plan2.steps)}")
            return plan2

        return self._do_fallback(world, instruction, reason_tag)

    # ── Path-1: Tool-calling agent loop ──────────────────────────────────────

    def _run_agent_loop(
        self,
        world: WorldState,
        instruction: str,
        labels: list[str],
    ) -> Optional[Plan]:
        """Run the tool-calling agent loop. Returns Plan on success, None to trigger fallback."""
        messages: list[dict] = [
            {"role": "system", "content": _AGENT_SYSTEM},
            {"role": "user",   "content": instruction},
        ]
        plan_steps: list[SkillCall] = []

        print(f"[LlmPlanner] agent loop start (model={self._model})")

        for iteration in range(_MAX_AGENT_ITERS):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=_AGENT_TOOLS,
                    tool_choice="auto",
                )
            except Exception as exc:
                print(f"[LlmPlanner] agent loop API error (iter {iteration}): {exc}")
                return None

            msg = resp.choices[0].message

            if not msg.tool_calls:
                if iteration == 0:
                    print("[LlmPlanner] model returned no tool calls on first turn "
                          "(tool-calling unsupported or model chose to skip) — falling back")
                else:
                    print(f"[LlmPlanner] agent loop finished after {iteration} iteration(s)")
                break

            # Append assistant turn (with tool_calls) to history
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Dispatch each tool call and append result
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    fn_args = {}

                result_str = self._dispatch_tool(fn_name, fn_args, world, labels, plan_steps)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
        else:
            print(f"[LlmPlanner] agent loop hit iteration limit ({_MAX_AGENT_ITERS})")

        if not plan_steps:
            return None
        return Plan(steps=plan_steps)

    def _dispatch_tool(
        self,
        name: str,
        args: dict,
        world: WorldState,
        labels: list[str],
        plan_steps: list[SkillCall],
    ) -> str:
        """Execute an info tool or collect an action tool. Returns a JSON result string."""
        if name == "list_objects":
            print(f"[LlmPlanner] list_objects() -> {labels}")
            return json.dumps({"objects": labels})

        if name == "get_robot_state":
            ee = world.robot.ee_pose
            pos = [round(float(v), 4) for v in ee.position.tolist()]
            result = {
                "ee_position": pos,
                "ee_quaternion": [round(float(v), 4) for v in ee.quaternion.tolist()],
                "gripper": round(float(world.robot.gripper), 3),
                "holding": world.robot.holding,
            }
            print(f"[LlmPlanner] get_robot_state() -> ee@{pos}")
            return json.dumps(result)

        if name == "get_object_position":
            query = args.get("query", "")
            label = _normalize_label(query, labels)
            if label is None:
                print(f"[LlmPlanner] get_object_position('{query}') -> NOT FOUND "
                      f"(available: {labels})")
                return json.dumps({
                    "error": f"Object '{query}' not found.",
                    "available_labels": labels,
                })
            obj = next(o for o in world.objects if o.label == label)
            pos = [round(float(v), 4) for v in obj.pose.position.tolist()]
            print(f"[LlmPlanner] get_object_position('{query}') -> {label}@{pos}")
            return json.dumps({
                "label": label,
                "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
            })

        if name == "pick":
            raw = args.get("object", "")
            label = _normalize_label(raw, labels)
            if label is None:
                print(f"[LlmPlanner] pick('{raw}') -> UNKNOWN LABEL")
                return json.dumps({"error": f"Unknown object '{raw}'.", "available_labels": labels})
            plan_steps.append(SkillCall(skill="pick", args={"label": label}))
            print(f"[LlmPlanner] pick({label})")
            return json.dumps({"status": "queued", "action": "pick", "object": label})

        if name == "place":
            raw_obj = args.get("object", "")
            raw_tgt = args.get("target", "")
            obj_label = _normalize_label(raw_obj, labels)
            tgt_label = _normalize_label(raw_tgt, labels)
            bad = [v for v, l in [(raw_obj, obj_label), (raw_tgt, tgt_label)] if l is None]
            if bad:
                print(f"[LlmPlanner] place('{raw_obj}', '{raw_tgt}') -> UNKNOWN LABEL {bad}")
                return json.dumps({"error": f"Unknown label(s): {bad}", "available_labels": labels})
            plan_steps.append(SkillCall(skill="place", args={"label": obj_label, "target": tgt_label}))
            print(f"[LlmPlanner] place({obj_label}, {tgt_label})")
            return json.dumps({"status": "queued", "action": "place",
                               "object": obj_label, "target": tgt_label})

        print(f"[LlmPlanner] unknown tool '{name}'")
        return json.dumps({"error": f"Unknown tool '{name}'"})

    # ── Path-2 helpers ────────────────────────────────────────────────────────

    def _do_fallback(self, world: WorldState, instruction: str, reason: str) -> Plan:
        if self._strict:
            raise RuntimeError(
                f"[LlmPlanner] STRICT MODE — LLM path failed (reason: {reason})"
            )
        print(f"[LlmPlanner] FALLBACK -> rule_based (reason: {reason})")
        return self._fallback_planner.plan(world, instruction)

    def _invoke(self, messages: list[dict]) -> dict:
        """Call the LLM for JSON output (Path 2).

        Three sub-paths tried in order:
          1. Ollama format=JSON-schema  (Ollama ≥ 0.4)
          2. Ollama format="json"       (Ollama ≥ 0.1)
          3. OpenAI tool-calling        (hosted APIs / last resort)
        """
        if not self._hosted:
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    extra_body={"format": _PLAN_SCHEMA},
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return json.loads(_clean_json(content))
            except Exception:
                pass

            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    extra_body={"format": "json"},
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return json.loads(_clean_json(content))
            except Exception as exc:
                print(f"[LlmPlanner] json format path failed ({exc}), trying tools path")

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=[_TOOL_DEF],
            tool_choice={"type": "function", "function": {"name": "plan"}},
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            return json.loads(msg.tool_calls[0].function.arguments)
        content = (msg.content or "").strip()
        if content:
            return json.loads(_clean_json(content))
        raise RuntimeError("All LLM API paths returned empty response")

    @staticmethod
    def _validate(raw: dict, valid_labels: list[str]) -> Optional[Plan]:
        """Return a Plan with normalised labels if raw is valid, else None."""
        steps_raw = raw.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            return None
        steps: list[SkillCall] = []
        for s in steps_raw:
            skill = s.get("skill")
            args  = dict(s.get("args", {}))
            if skill not in SKILL_SCHEMA:
                return None
            for k in ("label", "target"):
                if k in args:
                    mapped = _normalize_label(args[k], valid_labels)
                    if mapped is None:
                        return None
                    args[k] = mapped
            steps.append(SkillCall(skill=skill, args=args))
        return Plan(steps=steps)
