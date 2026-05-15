from __future__ import annotations

from typing import Any, Callable

from agent_app.runtime.shell_runtime import ShellRuntime
from agent_app.tools.approval import validate_shell_command
from agent_app.tools.base import Tool, ToolExecutionContext
from agent_app.types import ToolResult

Runner = Callable[..., object]


class ShellTool(Tool):
    name = "shell"
    description = (
        "Run a workspace-scoped read-only PowerShell command from the stage 1 "
        "whitelist. Use PowerShell syntax only, not CMD flags or Unix-only "
        "options. Prefer forms like 'Get-ChildItem -Recurse', 'Get-Content', "
        "and 'Get-Location'."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "minLength": 1},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        runtime: ShellRuntime | None = None,
        runner: Runner | None = None,
    ) -> None:
        if runtime is not None and runner is not None:
            raise ValueError("Only one of 'runtime' or 'runner' may be provided.")
        self._runtime = runtime or ShellRuntime(runner=runner)

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

        command = arguments.get("command")
        approved, reason = validate_shell_command(command)
        if not approved:
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error=reason)

        execution = self._runtime.run(
            command,
            workspace_root=context.workspace_root,
            timeout=context.timeout,
        )
        if execution.error_type == "timeout":
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                success=False,
                content=execution.combined_output,
                error="Shell command timed out.",
            )
        if execution.error_type == "nonzero_exit":
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                success=False,
                content=execution.combined_output,
                error=f"Shell command exited with code {execution.exit_code}.",
            )
        if execution.error_type == "runtime_error":
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                success=False,
                content=execution.combined_output,
                error="Shell runtime failed.",
            )
        return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=True, content=execution.combined_output, error=None)
