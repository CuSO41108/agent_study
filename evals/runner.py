from __future__ import annotations

import argparse
import json
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
    VerifyResult,
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
    parser.add_argument("--results-dir", default="evals/results", help="Directory for JSONL result files.")
    parser.add_argument("--run-root", default=".eval_runs", help="Directory for preserved per-run workspaces.")
    parser.add_argument("--case", dest="case_id", help="Run a single case id.")
    parser.add_argument("--limit", type=int, help="Run at most this many cases after filtering.")
    parser.add_argument("--dry-run", action="store_true", help="Validate cases and emit skipped records without calling a model.")
    parser.add_argument("--live-model", action="store_true", help="Actually call the configured model. Omit for non-network reporting.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path.cwd().resolve()
    cases = load_cases((repo_root / args.cases_dir).resolve())
    if args.case_id:
        cases = [case for case in cases if case["id"] == args.case_id]
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]

    run_id = _run_id()
    results_dir = (repo_root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{run_id}.jsonl"

    records = []
    for case in cases:
        if args.dry_run:
            record = _skipped_record(case=case, run_id=run_id, reason="dry_run")
        elif not args.live_model:
            record = _skipped_record(case=case, run_id=run_id, reason="live_model_not_requested")
        else:
            record = run_eval_case(
                case=case,
                run_id=run_id,
                repo_root=repo_root,
                fixtures_dir=(repo_root / args.fixtures_dir).resolve(),
                run_root=(repo_root / args.run_root).resolve(),
            )
        records.append(record)
        _append_jsonl(result_path, record)

    summary = summarize_results(records)
    print(json.dumps({"run_id": run_id, "result_path": str(result_path), "summary": summary}, ensure_ascii=False))
    return 0


def run_eval_case(
    *,
    case: dict,
    run_id: str,
    repo_root: Path,
    fixtures_dir: Path,
    run_root: Path,
) -> dict:
    case_id = case["id"]
    fixture_source = fixtures_dir / case["fixture"]
    workspace = run_root / run_id / case_id
    model_config = load_config(workspace_root=repo_root)
    if _missing_model_config(model_config):
        return _skipped_record(case=case, run_id=run_id, reason="model_configuration_missing")
    if not fixture_source.is_dir():
        return _error_record(case=case, run_id=run_id, workspace=workspace, error=f"Fixture not found: {fixture_source}")

    shutil.copytree(fixture_source, workspace)
    before = snapshot_workspace(workspace)

    config = load_config(workspace_root=workspace)
    initialize_database(config.database_path)
    sessions = SessionService(config.database_path)
    model_client = OpenAICompatibleModelClient(
        base_url=model_config.base_url,
        api_key=model_config.api_key,
        model=model_config.model,
        timeout=model_config.model_timeout,
    )
    shell_runtime = ShellRuntime()
    worker_registry = build_worker_registry(shell_runtime=shell_runtime)
    subagent_runner = SubagentRunner(
        model_client=model_client,
        session_service=sessions,
        workspace_root=config.workspace_root,
        tool_timeout=config.tool_timeout,
        context_token_budget=config.context_token_budget,
        summary_trigger_tokens=config.summary_trigger_tokens,
        confirmation_handler=lambda tool_call, context: True,
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
        confirmation_handler=lambda tool_call, context: True,
    )

    result = loop.run_turn(
        user_input=case["prompt"],
        session_id=None,
        budget=_budget_from_case(case),
    )
    after = snapshot_workspace(workspace)
    changed = changed_files(before, after)
    verify = _run_verify_command(workspace, case.get("oracle", {}).get("verify_command"))
    task_traces = sessions.list_task_traces(result.task_id) if result.task_id is not None else []
    score = score_eval_case(
        case=case,
        turn_success=result.success,
        stop_reason=result.stop_reason,
        changed=changed,
        tool_runs=result.tool_runs,
        task_traces=task_traces,
        verify=verify,
    )
    return {
        "run_id": run_id,
        "case_id": case_id,
        "status": "completed",
        "workspace": str(workspace),
        "turn_result": _serialize_turn_result(result),
        "verify": asdict(verify),
        "score": score,
    }


def _budget_from_case(case: dict) -> TaskBudget:
    return TaskBudget(**case.get("budget", {}))


def _missing_model_config(config) -> bool:
    return not (config.base_url and config.api_key and config.model)


def _run_verify_command(workspace: Path, command: str | None) -> VerifyResult:
    if not command:
        return VerifyResult(command=None, exit_code=None, output="")
    try:
        completed = subprocess.run(
            command,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=120,
            shell=True,
        )
    except subprocess.TimeoutExpired as exc:
        output = _join_output(exc.stdout or "", exc.stderr or "")
        return VerifyResult(command=command, exit_code=None, output=output or "Verification timed out.")
    output = _join_output(completed.stdout, completed.stderr)
    return VerifyResult(command=command, exit_code=completed.returncode, output=output)


def _serialize_turn_result(result) -> dict:
    payload = asdict(result)
    payload["tool_runs"] = [_dataclass_to_dict(item) for item in result.tool_runs]
    return payload


def _dataclass_to_dict(value):
    if is_dataclass(value):
        return asdict(value)
    return value


def _skipped_record(*, case: dict, run_id: str, reason: str) -> dict:
    return {
        "run_id": run_id,
        "case_id": case["id"],
        "status": "skipped",
        "reason": reason,
    }


def _error_record(*, case: dict, run_id: str, workspace: Path, error: str) -> dict:
    return {
        "run_id": run_id,
        "case_id": case["id"],
        "status": "error",
        "workspace": str(workspace),
        "error": error,
    }


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str))
        handle.write("\n")


def _run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def _join_output(stdout: str | None, stderr: str | None) -> str:
    return "\n".join(part.rstrip("\r\n") for part in (stdout, stderr) if part)


if __name__ == "__main__":
    raise SystemExit(main())
