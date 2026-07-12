from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


_ALWAYS_ALLOWED = {"file_read", "code_search", "web_search", "todo_read", "todo_write", "delegate_task"}
_RECURSIVE_DELETE_COMMANDS = {"remove-item", "rm", "rmdir", "rd", "del"}


@dataclass(frozen=True, slots=True)
class ApprovalResult:
    decision: Literal["allow", "confirm", "deny"]
    reason: str | None = None


def normalize_shell_command(command: str) -> str:
    return " ".join(command.strip().split())


def shell_approval_prefix(command: str) -> str:
    """Return a user-visible convenience prefix; it is not a security parser."""
    tokens = normalize_shell_command(command).split(" ")
    if len(tokens) >= 2 and tokens[0].casefold() == "git":
        return " ".join(tokens[:2])
    if len(tokens) >= 3 and tokens[0].casefold() in {"python", "py"} and tokens[1].casefold() == "-m":
        return " ".join(tokens[:3])
    return tokens[0] if tokens else ""


def command_matches_prefix(command: str, prefix: str) -> bool:
    normalized_command = normalize_shell_command(command).casefold()
    normalized_prefix = normalize_shell_command(prefix).casefold()
    return normalized_command == normalized_prefix or normalized_command.startswith(normalized_prefix + " ")


def approve_tool_call(tool_name: str, arguments: dict[str, Any]) -> ApprovalResult:
    if tool_name in _ALWAYS_ALLOWED:
        return ApprovalResult("allow")
    if tool_name == "shell":
        return classify_shell_command(arguments.get("command"))
    if tool_name in {"file_write", "replace_in_file"}:
        return ApprovalResult("confirm")
    return ApprovalResult("deny", f"Tool '{tool_name}' is not available in phase 2.")


def validate_shell_command(command: Any) -> tuple[bool, str | None]:
    result = classify_shell_command(command)
    return result.decision != "deny", result.reason


def classify_shell_command(command: Any) -> ApprovalResult:
    if not isinstance(command, str) or not command.strip():
        return ApprovalResult("deny", "Shell command must be a non-empty string.")
    if _is_forbidden_recursive_delete(normalize_shell_command(command)):
        return ApprovalResult("deny", "Recursive or batch deletion is forbidden by this workspace's AGENTS.md instructions.")
    return ApprovalResult("confirm", "Shell commands require user approval unless allowed for this session.")


def _is_forbidden_recursive_delete(command: str) -> bool:
    tokens = command.casefold().split(" ")
    if not tokens or tokens[0] not in _RECURSIVE_DELETE_COMMANDS:
        return False
    if tokens[0] in {"del", "rmdir", "rd"}:
        return "/s" in tokens
    return any(token in {"-recurse", "-r", "-rf", "-fr"} for token in tokens[1:])
