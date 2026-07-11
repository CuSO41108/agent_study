from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent_app.runtime.shell_runtime import ShellRuntime
from agent_app.tools.approval import ControlledShellCommand, parse_controlled_shell_command, validate_shell_command
from agent_app.tools.base import Tool, ToolExecutionContext
from agent_app.tools._path_utils import PathSafetyError, resolve_workspace_path
from agent_app.types import ToolResult

Runner = Callable[..., object]


class ShellTool(Tool):
    name = "shell"
    description = (
        "Run a workspace-scoped PowerShell command. Read-only allowlisted commands run automatically; "
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
    has_side_effect = False
    is_idempotent = True
    risk_level = "medium"

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

    def inspect(self, *, arguments: dict[str, Any], context: ToolExecutionContext) -> tuple["ShellInspection | None", str | None]:
        controlled = parse_controlled_shell_command(arguments.get("command"))
        if controlled is None:
            return None, "Shell command is not an approved workspace-changing command."
        paths: list[Path] = []
        try:
            for raw_path in controlled.paths:
                path = resolve_workspace_path(context.workspace_root, raw_path)
                relative = path.relative_to(context.workspace_root.resolve())
                if any(part in {".git", ".agent_app"} or part.startswith(".") for part in relative.parts):
                    return None, "Controlled shell actions cannot modify internal or hidden workspace paths."
                paths.append(path)
        except PathSafetyError as exc:
            return None, str(exc)
        if controlled.operation in {"move", "copy"}:
            if not paths[0].is_file():
                return None, "Controlled shell source must be an existing regular file."
            if not paths[1].parent.is_dir():
                return None, "Controlled shell destination parent directory does not exist."
            if paths[1].exists():
                return None, "Controlled shell destination already exists."
        if controlled.operation == "mkdir" and paths[0].exists() and not paths[0].is_dir():
            return None, "Controlled shell directory path is already a file."
        return ShellInspection(controlled=controlled, paths=tuple(paths)), None

    def recovery_metadata(self, *, tool_call_id: str, arguments: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        if parse_controlled_shell_command(arguments.get("command")) is None:
            return {"side_effect": False}
        inspection = context.prepared_edits.get(tool_call_id)
        if not isinstance(inspection, ShellInspection):
            inspection, error = self.inspect(arguments=arguments, context=context)
            if inspection is None:
                raise ValueError(error or "Unable to inspect controlled shell action.")
        return {"side_effect": True, "recovery_kind": "controlled_shell", "operation": inspection.controlled.operation, "paths": [str(path) for path in inspection.paths]}

    def has_side_effect_for(self, arguments: dict[str, Any]) -> bool:
        return parse_controlled_shell_command(arguments.get("command")) is not None

    def is_idempotent_for(self, arguments: dict[str, Any]) -> bool:
        return not self.has_side_effect_for(arguments)


@dataclass(frozen=True, slots=True)
class ShellInspection:
    controlled: ControlledShellCommand
    paths: tuple[Path, ...]
