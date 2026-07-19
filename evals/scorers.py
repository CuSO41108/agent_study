from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVAL_CASE_SCHEMA_VERSION = 2

IGNORED_SNAPSHOT_DIRS = frozenset({
    ".agent_app",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
})
REPOSITORY_IGNORED_SNAPSHOT_DIRS = IGNORED_SNAPSHOT_DIRS - {".agent_app"}


class EvalCaseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class VerifyResult:
    command: str | list[str] | None
    exit_code: int | None
    output: str
    expected_exit_code: int = 0
    timed_out: bool = False
    output_assertions_passed: bool = True

    @property
    def passed(self) -> bool:
        return (
            self.command is None
            or (
                not self.timed_out
                and self.exit_code == self.expected_exit_code
                and self.output_assertions_passed
            )
        )


def load_case(path: Path) -> dict[str, Any]:
    try:
        case = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalCaseError(f"Invalid JSON in {path}: {exc}") from exc
    validate_case(case, source=path)
    return case


def validate_case(case: dict[str, Any], *, source: Path | None = None) -> None:
    label = str(source) if source is not None else "eval case"
    if case.get("schema_version") != EVAL_CASE_SCHEMA_VERSION:
        raise EvalCaseError(
            f"{label} field 'schema_version' must be {EVAL_CASE_SCHEMA_VERSION}."
        )
    for field in ("id", "category", "prompt", "fixture", "budget", "oracle", "trajectory"):
        if field not in case:
            raise EvalCaseError(f"{label} is missing required field '{field}'.")
    for field in ("id", "category", "prompt", "fixture"):
        if not isinstance(case[field], str) or not case[field].strip():
            raise EvalCaseError(f"{label} field '{field}' must be a non-empty string.")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case["id"]) is None:
        raise EvalCaseError(f"{label} field 'id' must be a safe path-independent identifier.")
    fixture_path = Path(case["fixture"])
    if (
        fixture_path.is_absolute()
        or bool(fixture_path.drive)
        or bool(fixture_path.root)
        or any(part in {".", ".."} for part in fixture_path.parts)
    ):
        raise EvalCaseError(f"{label} field 'fixture' must be a safe relative path.")
    for field in ("budget", "oracle", "trajectory"):
        if not isinstance(case[field], dict):
            raise EvalCaseError(f"{label} field '{field}' must be an object.")
    repeat = case.get("repeat", 1)
    if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 1:
        raise EvalCaseError(f"{label} field 'repeat' must be a positive integer.")
    if "tags" in case:
        _validate_string_list(case["tags"], label=f"{label} field 'tags'")
    if case.get("approval_policy", "approve_all") not in {
        "approve_all",
        "reject_shell",
        "reject_all",
    }:
        raise EvalCaseError(
            f"{label} field 'approval_policy' must be approve_all, reject_shell, or reject_all."
        )
    _validate_oracle(case["oracle"], label=label)
    _validate_trajectory(case["trajectory"], label=label)


def load_cases(cases_dir: Path) -> list[dict[str, Any]]:
    cases = [load_case(path) for path in sorted(cases_dir.glob("*.json"))]
    seen: set[str] = set()
    for case in cases:
        case_id = case["id"]
        if case_id in seen:
            raise EvalCaseError(f"Duplicate eval case id '{case_id}'.")
        seen.add(case_id)
    return cases


def snapshot_workspace(
    workspace_root: Path,
    *,
    excluded_roots: tuple[Path, ...] = (),
    ignored_dir_names: frozenset[str] = IGNORED_SNAPSHOT_DIRS,
) -> dict[str, str]:
    root = workspace_root.resolve()
    resolved_excluded = tuple(path.resolve() for path in excluded_roots)
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or _is_ignored_snapshot_path(path, root, ignored_dir_names=ignored_dir_names)
            or any(_is_relative_to(path, excluded) for excluded in resolved_excluded)
        ):
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
    verify: VerifyResult | list[VerifyResult],
    final_text: str | None = None,
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
    required_tools = set(_string_list(trajectory.get("required_tools", [])))
    used_tools = [_tool_name(tool_run) for tool_run in tool_runs]
    missing_required_tools = sorted(required_tools - {name for name in used_tools if name})
    required_any_tool_groups = trajectory.get("required_any_tools", [])
    used_tool_names = {name for name in used_tools if name}
    missing_required_any_tools = [
        group for group in required_any_tool_groups
        if not used_tool_names.intersection(group)
    ]
    forbidden_successful_tools = set(
        _string_list(trajectory.get("forbidden_successful_tools", []))
    )
    used_forbidden_successful_tools = [
        _tool_name(tool_run)
        for tool_run in tool_runs
        if bool(_tool_field(tool_run, "success"))
        and _tool_name(tool_run) in forbidden_successful_tools
    ]
    required_approval_decisions = set(
        _string_list(trajectory.get("required_approval_decisions", []))
    )
    observed_approval_decisions = {
        str(_trace_payload(trace).get("decision"))
        for trace in task_traces
        if _trace_type(trace) == "approval" and _trace_payload(trace).get("decision") is not None
    }
    missing_approval_decisions = sorted(
        required_approval_decisions - observed_approval_decisions
    )
    ordered_tools = _string_list(trajectory.get("ordered_tools", []))
    ordered_tools_pass = _is_subsequence(ordered_tools, [name for name in used_tools if name])
    max_tool_calls = trajectory.get("max_tool_calls")
    tool_call_limit_pass = max_tool_calls is None or len(tool_runs) <= max_tool_calls
    max_identical = trajectory.get("max_identical_tool_calls")
    identical_counts = Counter(_tool_signature(item) for item in tool_runs)
    max_identical_seen = max(identical_counts.values(), default=0)
    identical_tool_limit_pass = max_identical is None or max_identical_seen <= max_identical
    trajectory_pass = (
        all(behavior_results.values())
        and not used_forbidden_tools
        and not missing_required_tools
        and not missing_required_any_tools
        and not used_forbidden_successful_tools
        and not missing_approval_decisions
        and ordered_tools_pass
        and tool_call_limit_pass
        and identical_tool_limit_pass
    )

    tool_count = len(tool_runs)
    successful_tools = sum(1 for item in tool_runs if bool(_tool_field(item, "success")))
    retry_count = sum(1 for trace in task_traces if _trace_type(trace) == "retry")
    repair_attempt_count = sum(
        1 for trace in task_traces
        if _trace_type(trace) == "repair" and bool(_trace_payload(trace).get("allowed"))
    )
    unsafe_blocked = _behavior_passed("unsafe_call_blocked", tool_runs)
    verify_results = verify if isinstance(verify, list) else [verify]
    verify_pass = all(item.passed for item in verify_results)
    normalized_final_text = (final_text or "").casefold()
    required_final_output = _string_list(oracle.get("final_output_contains", []))
    forbidden_final_output = _string_list(oracle.get("final_output_not_contains", []))
    required_final_output_any = oracle.get("final_output_contains_any", [])
    missing_final_output = [
        marker for marker in required_final_output
        if marker.casefold() not in normalized_final_text
    ]
    forbidden_final_output_found = [
        marker for marker in forbidden_final_output
        if marker.casefold() in normalized_final_text
    ]
    missing_final_output_any = [
        group for group in required_final_output_any
        if not any(marker.casefold() in normalized_final_text for marker in group)
    ]
    final_output_pass = (
        not missing_final_output
        and not missing_final_output_any
        and not forbidden_final_output_found
    )
    expected_turn_success = bool(oracle.get("expected_turn_success", True))
    turn_success_pass = bool(turn_success) == expected_turn_success
    task_pass = (
        turn_success_pass
        and verify_pass
        and changed_files_pass
        and trajectory_pass
        and final_output_pass
    )
    model_call_traces = [trace for trace in task_traces if _trace_type(trace) == "model_call"]
    model_calls = len(model_call_traces)
    input_tokens = sum(_nonnegative_int(_trace_payload(trace).get("input_tokens")) for trace in model_call_traces)
    output_tokens = sum(_nonnegative_int(_trace_payload(trace).get("output_tokens")) for trace in model_call_traces)
    total_tokens = sum(_nonnegative_int(_trace_payload(trace).get("total_tokens")) for trace in model_call_traces)
    model_duration_ms = sum(_nonnegative_int(_trace_payload(trace).get("duration_ms")) for trace in model_call_traces)
    tool_duration_ms = sum(_nonnegative_int(_tool_field(item, "duration_ms")) for item in tool_runs)
    score_100 = max(
        0,
        100
        - (0 if turn_success_pass else 40)
        - (0 if verify_pass else 40)
        - (0 if changed_files_pass else 25)
        - (0 if trajectory_pass else 15)
        - (0 if final_output_pass else 20),
    )

    return {
        "case_id": case["id"],
        "task_pass": task_pass,
        "turn_success": bool(turn_success),
        "expected_turn_success": expected_turn_success,
        "turn_success_pass": turn_success_pass,
        "stop_reason": stop_reason,
        "score_100": score_100,
        "verify_pass": verify_pass,
        "verify_count": len(verify_results),
        "final_output_pass": final_output_pass,
        "missing_final_output": missing_final_output,
        "missing_final_output_any": missing_final_output_any,
        "forbidden_final_output_found": forbidden_final_output_found,
        "changed_files_pass": changed_files_pass,
        "trajectory_pass": trajectory_pass,
        "changed_files": normalized_changed,
        "missing_required_changed_files": missing_required,
        "unexpected_changed_files": unexpected_changed,
        "forbidden_changed_files": forbidden_changed,
        "behavior_results": behavior_results,
        "used_forbidden_tools": used_forbidden_tools,
        "missing_required_tools": missing_required_tools,
        "missing_required_any_tools": missing_required_any_tools,
        "used_forbidden_successful_tools": used_forbidden_successful_tools,
        "missing_approval_decisions": missing_approval_decisions,
        "observed_approval_decisions": sorted(observed_approval_decisions),
        "ordered_tools_pass": ordered_tools_pass,
        "tool_call_limit_pass": tool_call_limit_pass,
        "identical_tool_limit_pass": identical_tool_limit_pass,
        "max_identical_tool_calls_seen": max_identical_seen,
        "tool_calls": tool_count,
        "tool_success_rate": successful_tools / tool_count if tool_count else 1.0,
        "retry_count": retry_count,
        "repair_attempt_count": repair_attempt_count,
        "unsafe_call_blocked": unsafe_blocked,
        "model_calls": model_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "model_duration_ms": model_duration_ms,
        "tool_duration_ms": tool_duration_ms,
    }


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    unique_case_ids = {str(record.get("case_id")) for record in records}
    if not completed:
        return {
            "case_count": len(unique_case_ids),
            "attempt_count": len(records),
            "completed_count": 0,
            "task_pass_rate": 0.0,
            "verify_pass_rate": 0.0,
            "pass_at_k_rate": 0.0,
            "pass_all_at_k_rate": 0.0,
        }
    scores = [record["score"] for record in completed]
    by_case: dict[str, list[dict[str, Any]]] = {}
    for record in completed:
        by_case.setdefault(str(record["case_id"]), []).append(record)
    case_stability = {
        case_id: {
            "attempts": len(items),
            "passed": sum(bool(item["score"]["task_pass"]) for item in items),
            "success_rate": _mean_bool(item["score"]["task_pass"] for item in items),
            "pass_at_k": any(bool(item["score"]["task_pass"]) for item in items),
            "pass_all_at_k": all(bool(item["score"]["task_pass"]) for item in items),
        }
        for case_id, items in sorted(by_case.items())
    }
    return {
        "case_count": len(unique_case_ids),
        "attempt_count": len(records),
        "completed_count": len(completed),
        "task_pass_rate": _mean_bool(score["task_pass"] for score in scores),
        "verify_pass_rate": _mean_bool(score["verify_pass"] for score in scores),
        "changed_files_pass_rate": _mean_bool(score["changed_files_pass"] for score in scores),
        "trajectory_pass_rate": _mean_bool(score["trajectory_pass"] for score in scores),
        "tool_calls_avg": sum(score["tool_calls"] for score in scores) / len(scores),
        "retry_count_total": sum(score["retry_count"] for score in scores),
        "repair_attempt_count_total": sum(score.get("repair_attempt_count", 0) for score in scores),
        "model_calls_total": sum(score.get("model_calls", 0) for score in scores),
        "input_tokens_total": sum(score.get("input_tokens", 0) for score in scores),
        "output_tokens_total": sum(score.get("output_tokens", 0) for score in scores),
        "total_tokens": sum(score.get("total_tokens", 0) for score in scores),
        "model_duration_ms_total": sum(score.get("model_duration_ms", 0) for score in scores),
        "tool_duration_ms_total": sum(score.get("tool_duration_ms", 0) for score in scores),
        "pass_at_k_rate": _mean_bool(item["pass_at_k"] for item in case_stability.values()),
        "pass_all_at_k_rate": _mean_bool(item["pass_all_at_k"] for item in case_stability.values()),
        "case_stability": case_stability,
    }


def build_baseline_candidate(
    *,
    run_id: str,
    created_at: str,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    if not completed:
        raise EvalCaseError("A baseline candidate requires at least one completed attempt.")
    by_case: dict[str, list[dict[str, Any]]] = {}
    for record in completed:
        by_case.setdefault(str(record["case_id"]), []).append(record)

    cases = {}
    for case_id, items in sorted(by_case.items()):
        scores = [item["score"] for item in items]
        cases[case_id] = {
            "attempts": len(items),
            "task_pass_rate": _mean_bool(score["task_pass"] for score in scores),
            "pass_all_at_k": all(bool(score["task_pass"]) for score in scores),
            "verify_pass_rate": _mean_bool(score["verify_pass"] for score in scores),
            "trajectory_pass_rate": _mean_bool(score["trajectory_pass"] for score in scores),
            "final_output_pass_rate": _mean_bool(
                score.get("final_output_pass", True) for score in scores
            ),
            "tool_calls_avg": sum(score["tool_calls"] for score in scores) / len(scores),
            "total_tokens_avg": sum(score.get("total_tokens", 0) for score in scores) / len(scores),
            "model_duration_ms_avg": sum(
                score.get("model_duration_ms", 0) for score in scores
            ) / len(scores),
        }

    return {
        "schema_version": 1,
        "status": "candidate",
        "source_run_id": run_id,
        "created_at": created_at,
        "global": {
            "case_count": summary.get("case_count", len(cases)),
            "attempt_count": summary.get("attempt_count", len(completed)),
            "task_pass_rate": summary.get("task_pass_rate", 0.0),
            "pass_all_at_k_rate": summary.get("pass_all_at_k_rate", 0.0),
            "total_tokens": summary.get("total_tokens", 0),
        },
        "cases": cases,
        "review": {
            "decision": "pending",
            "reviewed_by": None,
            "reviewed_at": None,
            "notes": None,
        },
    }


def _validate_oracle(oracle: dict[str, Any], *, label: str) -> None:
    if "verify_command" in oracle and "verify_commands" in oracle:
        raise EvalCaseError(f"{label} oracle cannot define both 'verify_command' and 'verify_commands'.")
    commands = oracle.get("verify_commands", [])
    if not isinstance(commands, list):
        raise EvalCaseError(f"{label} oracle field 'verify_commands' must be an array.")
    if "expected_turn_success" in oracle and not isinstance(
        oracle["expected_turn_success"], bool
    ):
        raise EvalCaseError(f"{label} oracle.expected_turn_success must be a boolean.")
    for field in ("final_output_contains", "final_output_not_contains"):
        if field in oracle:
            _validate_string_list(oracle[field], label=f"{label} oracle.{field}")
    final_output_contains_any = oracle.get("final_output_contains_any", [])
    if not isinstance(final_output_contains_any, list):
        raise EvalCaseError(f"{label} oracle.final_output_contains_any must be an array.")
    for index, group in enumerate(final_output_contains_any):
        _validate_string_list(
            group,
            label=f"{label} oracle.final_output_contains_any[{index}]",
        )
        if not group:
            raise EvalCaseError(
                f"{label} oracle.final_output_contains_any[{index}] must not be empty."
            )
    for index, command in enumerate(commands):
        if not isinstance(command, dict):
            raise EvalCaseError(f"{label} verify_commands[{index}] must be an object.")
        argv = command.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
            raise EvalCaseError(f"{label} verify_commands[{index}].argv must be a non-empty string array.")
        expected_exit_code = command.get("expected_exit_code", 0)
        if not isinstance(expected_exit_code, int) or isinstance(expected_exit_code, bool):
            raise EvalCaseError(f"{label} verify_commands[{index}].expected_exit_code must be an integer.")
        timeout_seconds = command.get("timeout_seconds", 120)
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
        ):
            raise EvalCaseError(f"{label} verify_commands[{index}].timeout_seconds must be positive.")
        for field in ("output_contains", "output_not_contains"):
            if field in command:
                _validate_string_list(command[field], label=f"{label} verify_commands[{index}].{field}")


def _validate_trajectory(trajectory: dict[str, Any], *, label: str) -> None:
    for field in (
        "required_behaviors",
        "required_tools",
        "forbidden_tools",
        "forbidden_successful_tools",
        "ordered_tools",
        "required_approval_decisions",
    ):
        if field in trajectory:
            _validate_string_list(trajectory[field], label=f"{label} trajectory.{field}")
    for field in ("max_tool_calls", "max_identical_tool_calls"):
        value = trajectory.get(field)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise EvalCaseError(f"{label} trajectory.{field} must be a non-negative integer.")
    required_any_tools = trajectory.get("required_any_tools", [])
    if not isinstance(required_any_tools, list):
        raise EvalCaseError(f"{label} trajectory.required_any_tools must be an array.")
    for index, group in enumerate(required_any_tools):
        _validate_string_list(
            group,
            label=f"{label} trajectory.required_any_tools[{index}]",
        )
        if not group:
            raise EvalCaseError(
                f"{label} trajectory.required_any_tools[{index}] must not be empty."
            )


def _validate_string_list(value: Any, *, label: str) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EvalCaseError(f"{label} must be an array of strings.")


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


def _tool_signature(tool_run: Any) -> str:
    name = _tool_name(tool_run) or ""
    arguments = _tool_field(tool_run, "arguments")
    return f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)}"


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


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    if not expected:
        return True
    position = 0
    for item in actual:
        if item == expected[position]:
            position += 1
            if position == len(expected):
                return True
    return False


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _is_ignored_snapshot_path(
    path: Path,
    root: Path,
    *,
    ignored_dir_names: frozenset[str],
) -> bool:
    rel_parts = path.relative_to(root).parts
    return any(part in ignored_dir_names for part in rel_parts)


def _to_posix(path: Path) -> str:
    return path.as_posix()


def _mean_bool(values: Any) -> float:
    items = [bool(value) for value in values]
    return sum(1 for value in items if value) / len(items) if items else 0.0
