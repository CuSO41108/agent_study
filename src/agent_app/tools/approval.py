from __future__ import annotations

from dataclasses import dataclass
import shlex
from typing import Any, Literal

_ALLOWED_SHELL_PREFIXES = ("git status", "git diff", "git log", "rg", "dir", "ls", "pwd", "get-childitem", "get-content", "type", "get-location", "python --version", "python -m unittest")
_ALWAYS_ALLOWED = {"file_read", "code_search", "web_search", "todo_read", "todo_write", "delegate_task"}

@dataclass(frozen=True, slots=True)
class ApprovalResult:
    decision: Literal["allow", "confirm", "deny"]
    reason: str | None = None

@dataclass(frozen=True, slots=True)
class ControlledShellCommand:
    operation: Literal["mkdir", "move", "copy"]
    command: str
    paths: tuple[str, ...]
    risk_level: Literal["medium"] = "medium"

def approve_tool_call(tool_name: str, arguments: dict[str, Any]) -> ApprovalResult:
    if tool_name in _ALWAYS_ALLOWED: return ApprovalResult("allow")
    if tool_name == "shell":
        decision, reason, _ = classify_shell_command(arguments.get("command")); return ApprovalResult(decision, reason)
    if tool_name in {"file_write", "replace_in_file"}: return ApprovalResult("confirm")
    return ApprovalResult("deny", f"Tool '{tool_name}' is not available in phase 2.")

def validate_shell_command(command: Any) -> tuple[bool, str | None]:
    decision, reason, _ = classify_shell_command(command); return decision != "deny", reason

def parse_controlled_shell_command(command: Any) -> ControlledShellCommand | None:
    return classify_shell_command(command)[2]

def classify_shell_command(command: Any) -> tuple[Literal["allow", "confirm", "deny"], str | None, ControlledShellCommand | None]:
    if not isinstance(command, str) or not command.strip(): return "deny", "Shell command must be a non-empty string.", None
    normalized = " ".join(command.strip().split()); lowered = normalized.casefold()
    if any(operator in normalized for operator in (">", "|", ";", "&&", "||")): return "deny", "Shell command uses operators that are disallowed in controlled shell mode.", None
    if lowered.startswith("dir /"):
        return "deny", "Current shell runs in PowerShell, not cmd. Do not use 'dir /b' or 'dir /s'; prefer 'Get-ChildItem -Name' or 'Get-ChildItem -Recurse'.", None
    if lowered == "ls -r" or lowered == "ls -recurse" or lowered.startswith("ls -r "):
        return "deny", "Current shell runs in PowerShell, not Unix shell. Do not use 'ls -R'; prefer 'Get-ChildItem -Recurse'.", None
    if any(lowered == prefix or lowered.startswith(prefix + " ") for prefix in _ALLOWED_SHELL_PREFIXES): return "allow", None, None
    try: tokens = shlex.split(normalized, posix=False)
    except ValueError: return "deny", "Shell command quoting is invalid.", None
    name = tokens[0].casefold() if tokens else ""; values, error = _named_values(tokens[1:])
    if error: return "deny", error, None
    if name == "new-item" and values.get("-itemtype", "").casefold() == "directory" and "-path" in values and set(values) <= {"-path", "-itemtype", "-force"}:
        return "confirm", "Workspace-changing shell action requires user approval.", ControlledShellCommand("mkdir", normalized, (values["-path"],))
    if name in {"move-item", "copy-item"} and set(values) == {"-path", "-destination"}:
        return "confirm", "Workspace-changing shell action requires user approval.", ControlledShellCommand("move" if name == "move-item" else "copy", normalized, (values["-path"], values["-destination"]))
    return "deny", "Shell command is not in the controlled shell whitelist.", None

def _named_values(tokens: list[str]) -> tuple[dict[str, str], str | None]:
    values: dict[str, str] = {}; i = 0
    while i < len(tokens):
        flag = tokens[i].casefold()
        if not flag.startswith("-") or flag in values: return {}, "Shell command contains unsupported or repeated arguments."
        if flag == "-force": values[flag] = "true"; i += 1; continue
        if i + 1 >= len(tokens) or tokens[i + 1].startswith("-"): return {}, f"Shell command argument {tokens[i]} requires a value."
        values[flag] = tokens[i + 1].strip('"'); i += 2
    return values, None
