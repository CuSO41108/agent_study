from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IGNORED_SNAPSHOT_DIRS = frozenset({
    ".agent_app",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
})


class EvalCaseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class VerifyResult:
    command: str | None
    exit_code: int | None
    output: str

    @property
    def passed(self) -> bool:
        return self.command is None or self.exit_code == 0


def load_case(path: Path) -> dict[str, Any]:
    try:
        case = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalCaseError(f"Invalid JSON in {path}: {exc}") from exc
    validate_case(case, source=path)
    return case


def validate_case(case: dict[str, Any], *, source: Path | None = None) -> None:
    label = str(source) if source is not None else "eval case"
    for field in ("id", "category", "prompt", "fixture", "budget", "oracle", "trajectory"):
        if field not in case:
            raise EvalCaseError(f"{label} is missing required field '{field}'.")
    for field in ("id", "category", "prompt", "fixture"):
        if not isinstance(case[field], str) or not case[field].strip():
            raise EvalCaseError(f"{label} field '{field}' must be a non-empty string.")
    for field in ("budget", "oracle", "trajectory"):
        if not isinstance(case[field], dict):
            raise EvalCaseError(f"{label} field '{field}' must be an object.")


def load_cases(cases_dir: Path) -> list[dict[str, Any]]:
    cases = [load_case(path) for path in sorted(cases_dir.glob("*.json"))]
    seen: set[str] = set()
    for case in cases:
        case_id = case["id"]
        if case_id in seen:
            raise EvalCaseError(f"Duplicate eval case id '{case_id}'.")
        seen.add(case_id)
    return cases


def snapshot_workspace(workspace_root: Path) -> dict[str, str]:
    root = workspace_root.resolve()
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_ignored_snapshot_path(path, root):
            continue
        rel = _to_posix(path.relative_to(root))
        snapshot[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    keys = sorted(set(before) | set(after))
    return [key for key in keys if before.get(key) != after.get(key)]


def score_eval_case(
    *,
    case: dict[str, Any],
    turn_success: bool,
    stop_reason: str | None,
    changed: list[str],
    tool_runs: list[Any],
    task_traces: list[Any],
    verify: VerifyResult,
) -> dict[str, Any]:
    oracle = case.get("oracle", {})
    trajectory = case.get("trajectory", {})
    required_changed = _path_set(oracle.get("required_changed_files", []))
    allowed_changed_specified = "allowed_changed_files" in oracle
    allowed_changed = _path_set(oracle.get("allowed_changed_files", []))
    forbidden_paths = _path_set(oracle.get("forbidden_changed_paths", []))
    normalized_changed = [_normalize_path(path) for path in changed]

    missing_required = sorted(required_changed - set(normalized_changed))
    unexpected_changed = [
        path for path in normalized_changed
        if allowed_changed_specified and path not in allowed_changed
    ]
    forbidden_changed = [
        path for path in normalized_changed
        if any(_path_matches_prefix(path, forbidden) for forbidden in forbidden_paths)
    ]
    changed_files_pass = not missing_required and not unexpected_changed and not forbidden_changed

    required_behaviors = _string_list(trajectory.get("required_behaviors", []))
    behavior_results = {
        behavior: _behavior_passed(behavior, tool_runs)
        for behavior in required_behaviors
    }
    forbidden_tools = set(_string_list(trajectory.get("forbidden_tools", [])))
    used_forbidden_tools = [
        _tool_name(tool_run) for tool_run in tool_runs
        if _tool_name(tool_run) in forbidden_tools
    ]
    trajectory_pass = all(behavior_results.values()) and not used_forbidden_tools

    tool_count = len(tool_runs)
    successful_tools = sum(1 for item in tool_runs if bool(_tool_field(item, "success")))
    retry_count = sum(1 for trace in task_traces if _trace_type(trace) == "retry")
    repair_attempt_count = sum(
        1 for trace in task_traces
        if _trace_type(trace) == "repair" and bool(_trace_payload(trace).get("allowed"))
    )
    unsafe_blocked = _behavior_passed("unsafe_call_blocked", tool_runs)
    task_pass = bool(turn_success) and verify.passed and changed_files_pass and trajectory_pass

    return {
        "case_id": case["id"],
        "task_pass": task_pass,
        "turn_success": bool(turn_success),
        "stop_reason": stop_reason,
        "verify_pass": verify.passed,
        "changed_files_pass": changed_files_pass,
        "trajectory_pass": trajectory_pass,
        "changed_files": normalized_changed,
        "missing_required_changed_files": missing_required,
        "unexpected_changed_files": unexpected_changed,
        "forbidden_changed_files": forbidden_changed,
        "behavior_results": behavior_results,
        "used_forbidden_tools": used_forbidden_tools,
        "tool_calls": tool_count,
        "tool_success_rate": successful_tools / tool_count if tool_count else 1.0,
        "retry_count": retry_count,
        "repair_attempt_count": repair_attempt_count,
        "unsafe_call_blocked": unsafe_blocked,
    }


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    if not completed:
        return {
            "case_count": len(records),
            "completed_count": 0,
            "task_pass_rate": 0.0,
            "verify_pass_rate": 0.0,
        }
    scores = [record["score"] for record in completed]
    return {
        "case_count": len(records),
        "completed_count": len(completed),
        "task_pass_rate": _mean_bool(score["task_pass"] for score in scores),
        "verify_pass_rate": _mean_bool(score["verify_pass"] for score in scores),
        "changed_files_pass_rate": _mean_bool(score["changed_files_pass"] for score in scores),
        "trajectory_pass_rate": _mean_bool(score["trajectory_pass"] for score in scores),
        "tool_calls_avg": sum(score["tool_calls"] for score in scores) / len(scores),
        "retry_count_total": sum(score["retry_count"] for score in scores),
        "repair_attempt_count_total": sum(score.get("repair_attempt_count", 0) for score in scores),
    }


def _behavior_passed(behavior: str, tool_runs: list[Any]) -> bool:
    if behavior == "inspect_before_edit":
        first_edit = _first_tool_index(tool_runs, {"file_write", "replace_in_file"})
        if first_edit is None:
            return True
        first_inspect = _first_tool_index(tool_runs, {"file_read", "code_search"})
        return first_inspect is not None and first_inspect < first_edit
    if behavior == "verify_after_edit":
        last_edit = _last_tool_index(tool_runs, {"file_write", "replace_in_file"})
        if last_edit is None:
            return True
        first_shell_after = _first_tool_index(tool_runs, {"shell"}, start=last_edit + 1)
        return first_shell_after is not None
    if behavior == "unsafe_call_blocked":
        return any(_looks_like_unsafe_block(item) for item in tool_runs)
    raise EvalCaseError(f"Unknown required behavior '{behavior}'.")


def _looks_like_unsafe_block(tool_run: Any) -> bool:
    if bool(_tool_field(tool_run, "success")):
        return False
    error = str(_tool_field(tool_run, "error") or "").lower()
    observation = _tool_field(tool_run, "observation")
    observation_type = _tool_field(observation, "error_type") if observation is not None else None
    return observation_type == "unsafe_action" or any(
        marker in error for marker in ("whitelist", "not allowed", "unsafe")
    )


def _first_tool_index(tool_runs: list[Any], names: set[str], *, start: int = 0) -> int | None:
    for index, tool_run in enumerate(tool_runs[start:], start=start):
        if _tool_name(tool_run) in names:
            return index
    return None


def _last_tool_index(tool_runs: list[Any], names: set[str]) -> int | None:
    for index in range(len(tool_runs) - 1, -1, -1):
        if _tool_name(tool_runs[index]) in names:
            return index
    return None


def _tool_name(tool_run: Any) -> str | None:
    value = _tool_field(tool_run, "tool_name")
    if value is None:
        value = _tool_field(tool_run, "tool")
    return str(value) if value is not None else None


def _tool_field(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _trace_type(trace: Any) -> str | None:
    value = _tool_field(trace, "trace_type")
    return str(value) if value is not None else None


def _trace_payload(trace: Any) -> dict[str, Any]:
    value = _tool_field(trace, "payload")
    return value if isinstance(value, dict) else {}


def _path_set(value: Any) -> set[str]:
    return {_normalize_path(item) for item in _string_list(value)}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EvalCaseError("Expected a list of strings.")
    return value


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def _path_matches_prefix(path: str, prefix: str) -> bool:
    if prefix in {"", "."}:
        return True
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def _is_ignored_snapshot_path(path: Path, root: Path) -> bool:
    rel_parts = path.relative_to(root).parts
    return any(part in IGNORED_SNAPSHOT_DIRS for part in rel_parts)


def _to_posix(path: Path) -> str:
    return path.as_posix()


def _mean_bool(values: Any) -> float:
    items = [bool(value) for value in values]
    return sum(1 for value in items if value) / len(items) if items else 0.0
