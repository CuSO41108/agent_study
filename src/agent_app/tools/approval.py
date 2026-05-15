from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

_ALLOWED_SHELL_PREFIXES = (
    "git status",
    "git diff",
    "git log",
    "rg",
    "dir",
    "ls",
    "pwd",
    "get-childitem",
    "get-content",
    "type",
    "get-location",
    "python --version",
    "python -m unittest",
)

_ALWAYS_ALLOWED = {"file_read", "code_search", "todo_read", "todo_write", "delegate_task"}


@dataclass(frozen=True, slots=True)
class ApprovalResult:
    decision: Literal["allow", "confirm", "deny"]
    reason: str | None = None



def approve_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
) -> ApprovalResult:
    if tool_name in _ALWAYS_ALLOWED:
        return ApprovalResult(decision="allow")
    if tool_name == "shell":
        allowed, reason = validate_shell_command(arguments.get("command"))
        if allowed:
            return ApprovalResult(decision="allow")
        return ApprovalResult(decision="deny", reason=reason)
    if tool_name in {"file_write", "replace_in_file"}:
        return ApprovalResult(decision="confirm")
    return ApprovalResult(
        decision="deny",
        reason=f"Tool '{tool_name}' is not available in phase 2.",
    )



def validate_shell_command(command: Any) -> tuple[bool, str | None]:
    if not isinstance(command, str) or not command.strip():
        return False, "Shell command must be a non-empty string."

    normalized = " ".join(command.strip().split())
    lowered = normalized.lower()
    if any(operator in normalized for operator in (">", "|", ";", "&&", "||")):
        return False, "Shell command uses operators that are disallowed in stage 1."

    compatibility_hint = _powershell_compatibility_hint(lowered)
    if compatibility_hint is not None:
        return False, compatibility_hint

    if not any(
        lowered == prefix or lowered.startswith(prefix + " ")
        for prefix in _ALLOWED_SHELL_PREFIXES
    ):
        return False, "Shell command is not in the stage 1 whitelist."

    return True, None


def _powershell_compatibility_hint(normalized_command: str) -> str | None:
    if normalized_command.startswith("dir /"):
        return (
            "Current shell runs in PowerShell, not cmd. Do not use 'dir /b' or "
            "'dir /s'; prefer 'Get-ChildItem -Name' or 'Get-ChildItem -Recurse'."
        )
    if normalized_command == "ls -r" or normalized_command == "ls -recurse" or normalized_command.startswith("ls -r "):
        return (
            "Current shell runs in PowerShell, not Unix shell. Do not use "
            "'ls -R'; prefer 'Get-ChildItem -Recurse'."
        )
    return None
