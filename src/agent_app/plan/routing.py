from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


ExecutionMode = Literal["react", "plan_only", "plan_and_execute"]


@dataclass(frozen=True, slots=True)
class RouteDecision:
    mode: ExecutionMode
    goal: str
    explicit: bool
    reason: str


def route_request(user_input: str) -> RouteDecision:
    """Choose ReAct or planning without changing the default simple-task path."""

    stripped = user_input.strip()
    if not stripped:
        raise ValueError("Task input cannot be empty.")
    command, _, raw_goal = stripped.partition(" ")
    lowered_command = command.casefold()
    goal = raw_goal.strip()
    if lowered_command == "/plan":
        if not goal:
            raise ValueError("/plan requires a goal.")
        return RouteDecision("plan_only", goal, True, "explicit_plan_command")
    if lowered_command in {"/plan-and-execute", "/plan_execute"}:
        if not goal:
            raise ValueError("/plan-and-execute requires a goal.")
        return RouteDecision("plan_and_execute", goal, True, "explicit_plan_and_execute_command")
    if lowered_command in {"/react", "/reactive"}:
        if not goal:
            raise ValueError("/react requires a goal.")
        return RouteDecision("react", goal, True, "explicit_react_command")
    if _looks_multi_step(stripped):
        return RouteDecision("plan_and_execute", stripped, False, "multi_step_markers")
    return RouteDecision("react", stripped, False, "simple_or_exploratory_task")


def _looks_multi_step(text: str) -> bool:
    lowered = text.casefold()
    if "\n" in text:
        return True
    if any(marker in lowered for marker in ("先", "然后", "再", "并且", "同时", "以及")):
        return True
    if re.search(r"\b(and then|then|after that|before|and verify|and test)\b", lowered):
        return True
    action_markers = (
        "implement",
        "refactor",
        "migrate",
        "update and",
        "fix and",
        "add and",
    )
    return any(marker in lowered for marker in action_markers)
