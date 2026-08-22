from __future__ import annotations

import json
from typing import Any

from agent_app.plan.executor import NodeExecutionResult, PlanNodeContext


class PlanAgentNodeRunner:
    """Adapt the existing AgentLoop to one bounded PlanGraph node."""

    def __init__(self, agent_loop: Any) -> None:
        self._agent_loop = agent_loop

    def __call__(self, context: PlanNodeContext) -> NodeExecutionResult:
        result = self._agent_loop.run_turn(
            user_input=_node_prompt(context),
            session_id=context.session_id,
            _task_id=context.task_id,
            _append_user_message=False,
            allowed_tools=context.node.allowed_tools,
            keep_task_open=True,
        )
        if result.pending_action is not None or result.task_status == "waiting_user":
            pending = result.pending_action
            return NodeExecutionResult(
                status="waiting_approval",
                metadata={
                    "pending_action_id": None if pending is None else pending.id,
                    "kind": None if pending is None else pending.kind,
                },
            )
        if result.success:
            return NodeExecutionResult(
                status="completed",
                output=result.final_text,
                metadata={"stop_reason": result.stop_reason},
            )
        return NodeExecutionResult(
            status="failed",
            error=result.final_text or result.stop_reason or "AgentLoop node execution failed.",
            metadata={"stop_reason": result.stop_reason},
        )


def _node_prompt(context: PlanNodeContext) -> str:
    prior_results = json.dumps(context.node_results, ensure_ascii=False, sort_keys=True)
    acceptance = "\n".join(f"- {item}" for item in context.node.acceptance)
    return (
        "Execute exactly one node from the user's coding plan. Do not broaden the scope "
        "or execute later nodes.\n\n"
        f"Node kind: {context.node.kind}\n"
        f"Objective: {context.node.objective}\n"
        f"Acceptance conditions:\n{acceptance}\n"
        f"Completed node evidence:\n{prior_results}\n"
        "When the acceptance conditions are met, return a concise evidence-backed summary."
    )
