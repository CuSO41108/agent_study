from __future__ import annotations

from typing import Any

from agent_app.orchestrator.subagent_runner import DelegatedTaskRequest, SubagentRunner, normalize_relevant_paths
from agent_app.tools.base import Tool, ToolExecutionContext
from agent_app.types import ToolResult


class DelegateTaskTool(Tool):
    name = "delegate_task"
    description = (
        "Delegate a bounded subtask to a worker subagent. Use this only when "
        "the subtask boundary is clear and the isolation is worth the extra "
        "round trip."
    )
    has_side_effect = True
    is_idempotent = False
    risk_level = "medium"
    parameters_schema = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "minLength": 1},
            "success_criteria": {"type": "string", "minLength": 1},
            "relevant_paths": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "maxItems": 5,
            },
        },
        "required": ["task", "success_criteria"],
        "additionalProperties": False,
    }

    def __init__(self, *, runner: SubagentRunner) -> None:
        self._runner = runner

    def execute(
        self,
        *,
        tool_call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        validation_error = self.validate_arguments(arguments)
        if validation_error is not None:
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error=validation_error)

        request = DelegatedTaskRequest(
            task=str(arguments["task"]),
            success_criteria=str(arguments["success_criteria"]),
            relevant_paths=normalize_relevant_paths(arguments.get("relevant_paths")),
        )
        return self._runner.run(tool_call_id=tool_call_id, request=request, context=context)
