from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_app.agent.definition import SINGLE_MAIN_AGENT
from agent_app.config import load_config
from agent_app.model.openai_compatible import OpenAICompatibleModelClient
from agent_app.orchestrator.loop import AgentLoop
from agent_app.orchestrator.subagent_runner import SubagentRunner
from agent_app.runtime.shell_runtime import ShellRuntime
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService
from agent_app.tools.registry import build_root_registry, build_worker_registry
from agent_app.types import TaskBudget
from evals.scorers import (
    REPOSITORY_IGNORED_SNAPSHOT_DIRS,
    VerifyResult,
    build_baseline_candidate,
    changed_files,
    load_cases,
    score_eval_case,
    snapshot_workspace,
    summarize_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run agent-study eval cases and write JSONL results.")
    parser.add_argument("--cases-dir", default="evals/cases", help="Directory containing eval case JSON files.")
    parser.add_argument("--fixtures-dir", default="evals/fixtures", help="Directory containing fixture workspaces.")
    parser.add_argument("--results-dir", help="Optional separate directory for JSONL results. Defaults to the run directory.")
    parser.add_argument("--run-root", help="Root for isolated eval runs. Defaults outside the repository.")
    parser.add_argument("--case", dest="case_id", help="Run a single case id.")
    parser.add_argument("--limit", type=int, help="Run at most this many cases after filtering.")
    parser.add_argument("--repeat", type=int, help="Override the per-case repeat count.")
    parser.add_argument("--gate", action="store_true", help="Return a non-zero exit code when a completed attempt fails.")
    parser.add_argument(
        "--write-baseline-candidate",
        action="store_true",
        help="Write a review-pending baseline candidate for completed attempts.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate cases and emit skipped records without calling a model.")
    parser.add_argument("--live-model", action="store_true", help="Actually call the configured model. Omit for non-network reporting.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeat is not None and args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    repo_root = Path.cwd().resolve()
    cases = load_cases((repo_root / args.cases_dir).resolve())
    if args.case_id:
        cases = [case for case in cases if case["id"] == args.case_id]
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]

    run_id = _run_id()
    run_root = _resolve_path(repo_root, args.run_root) if args.run_root else default_run_root()
    _validate_artifact_root(repo_root, run_root, option="--run-root")
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    results_dir = _resolve_path(repo_root, args.results_dir) if args.results_dir else run_dir
    _validate_artifact_root(repo_root, results_dir, option="--results-dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / (f"{run_id}.jsonl" if args.results_dir else "results.jsonl")
    summary_path = run_dir / "summary.json"
    excluded_roots = tuple({run_root.resolve(), results_dir.resolve()})
    repository_before = snapshot_workspace(
        repo_root,
        excluded_roots=excluded_roots,
        ignored_dir_names=REPOSITORY_IGNORED_SNAPSHOT_DIRS,
    )

    records = []
    quota_exhausted = False
    for case in cases:
        repeats = args.repeat if args.repeat is not None else int(case.get("repeat", 1))
        for attempt_index in range(1, repeats + 1):
            if args.dry_run:
                record = _skipped_record(
                    case=case,
                    run_id=run_id,
                    attempt_index=attempt_index,
                    reason="dry_run",
                )
            elif not args.live_model:
                record = _skipped_record(
                    case=case,
                    run_id=run_id,
                    attempt_index=attempt_index,
                    reason="live_model_not_requested",
                )
            else:
                record = run_eval_case(
                    case=case,
                    run_id=run_id,
                    attempt_index=attempt_index,
                    repo_root=repo_root,
                    fixtures_dir=_resolve_path(repo_root, args.fixtures_dir),
                    run_root=run_root,
                )
            records.append(record)
            _append_jsonl(result_path, record)
            if record.get("status") == "quota_exhausted":
                quota_exhausted = True
                break
        if quota_exhausted:
            break

    summary = summarize_results(records)
    repository_after = snapshot_workspace(
        repo_root,
        excluded_roots=excluded_roots,
        ignored_dir_names=REPOSITORY_IGNORED_SNAPSHOT_DIRS,
    )
    repository_changes = changed_files(repository_before, repository_after)
    summary.update({
        "schema_version": 2,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "result_path": str(result_path),
        "repository_clean": not repository_changes,
        "repository_changes": repository_changes,
        "quota_exhausted": quota_exhausted,
    })
    if quota_exhausted:
        summary["action_required"] = "Model quota is exhausted. Please top up the configured provider account and run the suite again."
        print(summary["action_required"], file=sys.stderr)
    baseline_candidate_path = None
    if args.write_baseline_candidate and not quota_exhausted and not repository_changes:
        completed_records = [record for record in records if record.get("status") == "completed"]
        if completed_records:
            baseline_candidate_path = run_dir / "baseline-candidate.json"
            _write_json(
                baseline_candidate_path,
                build_baseline_candidate(
                    run_id=run_id,
                    created_at=datetime.now(UTC).isoformat(),
                    records=records,
                    summary=summary,
                ),
            )
    summary["baseline_candidate_path"] = (
        str(baseline_candidate_path) if baseline_candidate_path is not None else None
    )
    _write_json(summary_path, summary)
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "result_path": str(result_path), "summary_path": str(summary_path), "summary": summary}, ensure_ascii=False))
    if quota_exhausted:
        return 2
    if args.gate and (
        repository_changes
        or _gate_failed(records)
        or _live_gate_incomplete(live_model=args.live_model, records=records)
    ):
        return 1
    return 0


def run_eval_case(
    *,
    case: dict,
    run_id: str,
    attempt_index: int,
    repo_root: Path,
    fixtures_dir: Path,
    run_root: Path,
    model_client=None,
) -> dict:
    case_id = case["id"]
    resolved_fixtures_dir = fixtures_dir.resolve()
    fixture_source = (resolved_fixtures_dir / case["fixture"]).resolve()
    attempt_dir = _attempt_dir(run_root, run_id, case_id, attempt_index)
    workspace = attempt_dir / "workspace"
    if model_client is None:
        model_config = load_config(workspace_root=repo_root)
        if _missing_model_config(model_config):
            return _skipped_record(
                case=case,
                run_id=run_id,
                attempt_index=attempt_index,
                reason="model_configuration_missing",
            )
    if not fixture_source.is_relative_to(resolved_fixtures_dir) or not fixture_source.is_dir():
        return _error_record(
            case=case,
            run_id=run_id,
            attempt_index=attempt_index,
            workspace=workspace,
            error=f"Fixture not found: {fixture_source}",
        )

    attempt_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(fixture_source, workspace)
    before = snapshot_workspace(workspace)

    config = load_config(workspace_root=workspace)
    initialize_database(config.database_path)
    sessions = SessionService(config.database_path)
    if model_client is None:
        model_client = OpenAICompatibleModelClient(
            base_url=model_config.base_url,
            api_key=model_config.api_key,
            model=model_config.model,
            timeout=model_config.model_timeout,
        )
    shell_runtime = ShellRuntime()
    worker_registry = build_worker_registry(shell_runtime=shell_runtime)
    confirmation_handler = _confirmation_handler_from_case(case)
    subagent_runner = SubagentRunner(
        model_client=model_client,
        session_service=sessions,
        workspace_root=config.workspace_root,
        tool_timeout=config.tool_timeout,
        context_token_budget=config.context_token_budget,
        summary_trigger_tokens=config.summary_trigger_tokens,
        confirmation_handler=confirmation_handler,
        worker_registry=worker_registry,
    )
    loop = AgentLoop(
        agent=SINGLE_MAIN_AGENT,
        model_client=model_client,
        tool_registry=build_root_registry(subagent_runner=subagent_runner, shell_runtime=shell_runtime),
        session_service=sessions,
        workspace_root=config.workspace_root,
        tool_timeout=config.tool_timeout,
        context_token_budget=config.context_token_budget,
        summary_trigger_tokens=config.summary_trigger_tokens,
        confirmation_handler=confirmation_handler,
    )

    result = loop.run_turn(
        user_input=case["prompt"],
        session_id=None,
        budget=_budget_from_case(case),
    )
    after = snapshot_workspace(workspace)
    changed = changed_files(before, after)
    verify = _run_verify_commands(workspace, case.get("oracle", {}))
    task_traces = sessions.list_task_traces(result.task_id) if result.task_id is not None else []
    score = score_eval_case(
        case=case,
        turn_success=result.success,
        stop_reason=result.stop_reason,
        changed=changed,
        tool_runs=result.tool_runs,
        task_traces=task_traces,
        verify=verify,
        final_text=result.final_text,
    )
    status = "quota_exhausted" if _quota_exhausted(task_traces) else "completed"
    record = {
        "run_id": run_id,
        "case_id": case_id,
        "attempt": attempt_index,
        "status": status,
        "workspace": str(workspace),
        "turn_result": _serialize_turn_result(result),
        "verify": [asdict(item) for item in verify],
        "score": score,
    }
    _write_json(attempt_dir / "trace.json", {
        "run_id": run_id,
        "case_id": case_id,
        "attempt": attempt_index,
        "task_id": result.task_id,
        "events": [_dataclass_to_dict(item) for item in task_traces],
    })
    _write_json(attempt_dir / "score.json", score)
    _write_json(attempt_dir / "verify.json", record["verify"])
    return record


def _budget_from_case(case: dict) -> TaskBudget:
    return TaskBudget(**case.get("budget", {}))


def _missing_model_config(config) -> bool:
    return not (config.base_url and config.api_key and config.model)


def _run_verify_commands(workspace: Path, oracle: dict) -> list[VerifyResult]:
    command_specs = oracle.get("verify_commands")
    if command_specs is None:
        legacy_command = oracle.get("verify_command")
        command_specs = [] if not legacy_command else [{"command": legacy_command}]
    if not command_specs:
        return [VerifyResult(command=None, exit_code=None, output="")]
    return [_run_verify_command(workspace, spec) for spec in command_specs]


def _run_verify_command(workspace: Path, spec: dict) -> VerifyResult:
    argv = spec.get("argv")
    command = argv if argv is not None else spec.get("command")
    expected_exit_code = int(spec.get("expected_exit_code", 0))
    timeout_seconds = float(spec.get("timeout_seconds", 120))
    try:
        completed = subprocess.run(
            command,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=argv is None,
        )
    except subprocess.TimeoutExpired as exc:
        output = _join_output(exc.stdout or "", exc.stderr or "")
        return VerifyResult(
            command=command,
            exit_code=None,
            output=output or "Verification timed out.",
            expected_exit_code=expected_exit_code,
            timed_out=True,
            output_assertions_passed=False,
        )
    output = _join_output(completed.stdout, completed.stderr)
    output_assertions_passed = all(
        marker in output for marker in spec.get("output_contains", [])
    ) and all(
        marker not in output for marker in spec.get("output_not_contains", [])
    )
    return VerifyResult(
        command=command,
        exit_code=completed.returncode,
        output=output,
        expected_exit_code=expected_exit_code,
        output_assertions_passed=output_assertions_passed,
    )


def _serialize_turn_result(result) -> dict:
    payload = asdict(result)
    payload["tool_runs"] = [_dataclass_to_dict(item) for item in result.tool_runs]
    return payload


def _dataclass_to_dict(value):
    if is_dataclass(value):
        return asdict(value)
    return value


def _skipped_record(*, case: dict, run_id: str, attempt_index: int, reason: str) -> dict:
    return {
        "run_id": run_id,
        "case_id": case["id"],
        "attempt": attempt_index,
        "status": "skipped",
        "reason": reason,
    }


def _error_record(
    *,
    case: dict,
    run_id: str,
    attempt_index: int,
    workspace: Path,
    error: str,
) -> dict:
    return {
        "run_id": run_id,
        "case_id": case["id"],
        "attempt": attempt_index,
        "status": "error",
        "workspace": str(workspace),
        "error": error,
    }


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str))
        handle.write("\n")


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def _join_output(stdout: str | None, stderr: str | None) -> str:
    return "\n".join(part.rstrip("\r\n") for part in (stdout, stderr) if part)


def default_run_root() -> Path:
    configured = os.environ.get("AGENT_STUDY_EVAL_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        return Path("C:/tmp/agent-study-evals").resolve()
    return Path("/tmp/agent-study-evals").resolve()


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo_root / path).resolve()


def _validate_artifact_root(repo_root: Path, artifact_root: Path, *, option: str) -> None:
    if repo_root == artifact_root or repo_root.is_relative_to(artifact_root):
        raise SystemExit(f"{option} cannot be the repository root or one of its parents")


def _attempt_dir(run_root: Path, run_id: str, case_id: str, attempt_index: int) -> Path:
    case_path = Path(case_id)
    if (
        case_path.is_absolute()
        or bool(case_path.drive)
        or bool(case_path.root)
        or len(case_path.parts) != 1
        or case_id in {".", ".."}
    ):
        raise ValueError("case_id must be a safe path-independent identifier")
    return run_root / run_id / "cases" / case_id / f"attempt-{attempt_index:03d}"


def _quota_exhausted(task_traces: list) -> bool:
    return any(
        getattr(trace, "trace_type", None) == "model_call"
        and getattr(trace, "payload", {}).get("error_type") == "quota_exhausted"
        for trace in task_traces
    )


def _gate_failed(records: list[dict]) -> bool:
    return any(
        record.get("status") == "error"
        or (
            record.get("status") == "completed"
            and not bool(record.get("score", {}).get("task_pass"))
        )
        for record in records
    )


def _live_gate_incomplete(*, live_model: bool, records: list[dict]) -> bool:
    return live_model and (
        not records or any(record.get("status") != "completed" for record in records)
    )


def _confirmation_handler_from_case(case: dict):
    policy = case.get("approval_policy", "approve_all")

    def _confirm(tool_call, _context):
        if policy == "reject_all":
            return False
        if policy == "reject_shell" and tool_call.name == "shell":
            return False
        return True

    return _confirm


if __name__ == "__main__":
    raise SystemExit(main())
