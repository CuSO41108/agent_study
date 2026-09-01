from __future__ import annotations

import json
import re
from typing import Any

from agent_app.plan.executor import NodeExecutionResult, PlanNodeContext
from agent_app.types import ToolResult


_EVIDENCE_TOOLS_BY_KIND: dict[str, frozenset[str]] = {
    "inspect": frozenset({
        "file_read",
        "code_search",
        "web_search",
        "skill_list",
        "skill_load",
        "skill_read_resource",
    }),
    "edit": frozenset({"replace_in_file", "file_write"}),
    "run": frozenset({"shell"}),
    "verify": frozenset({"file_read", "code_search", "shell"}),
}
_UUID_PATTERN = re.compile(
    r"(?<![0-9a-fA-F])"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
    r"(?![0-9a-fA-F])"
)


class PlanAgentNodeRunner:
    """Adapt the existing AgentLoop to one bounded PlanGraph node.

    AgentLoop currently owns mutable turn state and the parent TaskRuntime has
    one optimistic version stream. Keep this adapter serial until a later
    isolated worker/task implementation explicitly opts into concurrency.
    """

    supports_concurrent = False

    def __init__(self, agent_loop: Any) -> None:
        self._agent_loop = agent_loop

    def __call__(self, context: PlanNodeContext) -> NodeExecutionResult:
        result = self._agent_loop.run_turn(
            user_input="",
            session_id=context.session_id,
            _task_id=context.task_id,
            _append_user_message=False,
            allowed_tools=context.node.allowed_tools,
            keep_task_open=True,
            transient_context=build_node_prompt(context),
            plan_revision_id=context.revision.id,
            plan_node_id=context.node.id,
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
        if result.stop_reason == "max_tool_rounds_exceeded" or result.task_status == "paused":
            return NodeExecutionResult(
                status="paused",
                error="Node execution window exhausted; explicit continuation is required.",
                metadata={
                    "stop_reason": result.stop_reason,
                },
            )
        if result.success:
            validation = _validate_acceptance_evidence(context, result.tool_runs)
            if validation["status"] != "passed":
                return NodeExecutionResult(
                    status="failed",
                    error=_acceptance_failure_message(context, validation),
                    evidence_refs=tuple(validation["evidence_refs"]),
                    metadata={
                        "stop_reason": "acceptance_evidence_missing",
                        "failure_category": "acceptance_not_met",
                        "acceptance_validation": validation,
                    },
                )
            return NodeExecutionResult(
                status="completed",
                output=result.final_text,
                evidence_refs=tuple(validation["evidence_refs"]),
                metadata={
                    "stop_reason": result.stop_reason,
                    "acceptance_validation": validation,
                },
            )
        return NodeExecutionResult(
            status="failed",
            error=result.final_text or result.stop_reason or "AgentLoop node execution failed.",
            metadata={"stop_reason": result.stop_reason},
        )


def build_node_prompt(context: PlanNodeContext) -> str:
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


def _validate_acceptance_evidence(
    context: PlanNodeContext,
    tool_runs: list[ToolResult],
) -> dict[str, Any]:
    required_tools = _EVIDENCE_TOOLS_BY_KIND[context.node.kind]
    qualifying_runs = [
        tool_run
        for tool_run in tool_runs
        if tool_run.success
        and tool_run.tool_name in required_tools
        and _is_substantive_evidence(tool_run)
    ]
    evidence_refs = [_evidence_ref(tool_run) for tool_run in qualifying_runs]
    evidence_text = "\n".join(tool_run.content for tool_run in qualifying_runs).lower()
    required_anchors = sorted(
        {
            match.group(0).lower()
            for match in _UUID_PATTERN.finditer(
                "\n".join((context.node.objective, *context.node.acceptance))
            )
        }
    )
    missing_anchors = [anchor for anchor in required_anchors if anchor not in evidence_text]
    status = "passed" if evidence_refs and not missing_anchors else "failed"
    return {
        "status": status,
        "policy": "minimum_tool_evidence_v1",
        "required_tools": sorted(required_tools),
        "evidence_refs": evidence_refs,
        "required_anchors": required_anchors,
        "missing_anchors": missing_anchors,
    }


def _is_substantive_evidence(tool_run: ToolResult) -> bool:
    if tool_run.tool_name == "code_search":
        return bool(tool_run.content.strip()) and tool_run.content.strip().lower() != "no matches found."
    return True


def _evidence_ref(tool_run: ToolResult) -> str:
    if tool_run.observation is not None and tool_run.observation.evidence_ref:
        return tool_run.observation.evidence_ref
    return f"tool_call:{tool_run.tool_call_id}"


def _acceptance_failure_message(
    context: PlanNodeContext,
    validation: dict[str, Any],
) -> str:
    missing_anchors = validation["missing_anchors"]
    if missing_anchors:
        return (
            "acceptance_evidence_missing: successful tool evidence for node "
            f"'{context.node.id}' does not contain required anchor(s): "
            + ", ".join(missing_anchors)
        )
    return (
        "acceptance_evidence_missing: node "
        f"'{context.node.id}' has no successful substantive evidence from the required tools: "
        + ", ".join(validation["required_tools"])
    )
