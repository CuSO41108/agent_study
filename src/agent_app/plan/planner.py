from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from agent_app.plan.graph import PlanGraph, parse_plan_graph


class PlanPlanningError(ValueError):
    """Raised when the model does not return a safe structured plan."""


class PlanPlanner:
    """Turn a user goal into a validated PlanGraph without executing tools."""

    def __init__(self, model_client: Any) -> None:
        self._model_client = model_client

    def create_plan(self, goal: str) -> PlanGraph:
        if not goal.strip():
            raise ValueError("Plan goal cannot be empty.")
        response = self._model_client.generate(
            system_prompt=(
                "You are a conservative coding-task planner. Return only one JSON object, "
                "with no Markdown and no commentary. The object must contain id, revision=1, "
                "goal, and nodes. Each node must use exactly one kind: inspect, edit, run, "
                "or verify. Each node must include objective, depends_on, allowed_tools, "
                "acceptance, and status=pending. Use a static acyclic dependency graph. "
                "Allowed tools by kind are: inspect=file_read,code_search,web_search,skill_list,"
                "skill_load,skill_read_resource; edit=file_read,code_search,replace_in_file,file_write; "
                "run=shell; verify=file_read,code_search,shell. Keep the plan as small as possible."
            ),
            messages=[{"role": "user", "content": goal}],
            tools=[],
        )
        if getattr(response, "error_type", None):
            raise PlanPlanningError(f"Planner model failed: {response.error_type}")
        if getattr(response, "tool_calls", None):
            raise PlanPlanningError("Planner must return JSON without tool calls.")
        text = getattr(response, "assistant_text", None)
        if not isinstance(text, str) or not text.strip():
            raise PlanPlanningError("Planner returned no JSON plan.")

        payload = _decode_json_object(text)
        payload.setdefault("id", f"plan-{uuid4().hex}")
        payload.setdefault("revision", 1)
        payload["goal"] = goal.strip()
        nodes = payload.get("nodes")
        if isinstance(nodes, list):
            for node in nodes:
                if isinstance(node, dict):
                    node.setdefault("status", "pending")
        try:
            return parse_plan_graph(payload)
        except (TypeError, KeyError, ValueError) as exc:
            raise PlanPlanningError(str(exc)) from exc


def _decode_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise PlanPlanningError("Planner response does not contain a JSON object.")
