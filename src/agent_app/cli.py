from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from agent_app.agent.definition import SINGLE_MAIN_AGENT
from agent_app.config import load_config
from agent_app.orchestrator.subagent_runner import SubagentRunner
from agent_app.model.openai_compatible import OpenAICompatibleModelClient
from agent_app.orchestrator.loop import AgentLoop
from agent_app.state.db import initialize_database
from agent_app.state.session_service import ActiveTaskConflict, SessionService
from agent_app.tools.base import ToolExecutionContext
from agent_app.tools.replace_in_file import ReplaceInFileInspection, inspect_replace_in_file_request
from agent_app.tools.file_write import inspect_file_write_request
from agent_app.tools.registry import build_root_registry
from agent_app.tools.web_search import WebSearchTool
from agent_app.tools.shell import ShellInspection, ShellTool
from agent_app.types import AgentEvent, TaskState, ToolCall, TurnResult


_TERMINAL_TASK_STATUSES = {"completed", "failed", "cancelled", "expired"}
_REPL_HELP = """Commands:
  /task       Show the latest task in this session.
  /tasks      List tasks in this session.
  /approve [task-id-prefix]
              Approve a task waiting for tool approval.
  /reject [task-id-prefix]
              Reject a task waiting for tool approval.
  /cancel     Cancel the latest non-terminal task.
  /new        Start a new session.
  /help       Show this help.
  exit        Leave interactive mode."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 5 CLI-first coordinator with worker subagents.")
    parser.add_argument("prompt", nargs="?", help="User prompt to process.")
    parser.add_argument(
        "--session-id",
        dest="session_id",
        help="Reuse an existing session identifier.",
    )
    parser.add_argument(
        "--workspace-root",
        dest="workspace_root",
        default=os.getcwd(),
        help="Workspace root for file and shell tools (default: current directory).",
    )
    parser.add_argument(
        "--new-session",
        dest="new_session",
        action="store_true",
        help="Start a fresh session instead of reusing the most recent local session.",
    )
    parser.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        help="Explicitly start the interactive REPL (also the default when no prompt is supplied).",
    )
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--task-status", metavar="TASK_ID", help="Show the persisted task state.")
    controls.add_argument("--pause-task", metavar="TASK_ID", help="Pause a running task.")
    controls.add_argument("--resume-task", metavar="TASK_ID", help="Resume a paused task.")
    controls.add_argument("--cancel-task", metavar="TASK_ID", help="Cancel a non-terminal task.")
    controls.add_argument("--approve-task", metavar="TASK_ID", help="Approve a persisted pending action.")
    controls.add_argument("--reject-task", metavar="TASK_ID", help="Reject a persisted pending action.")
    return parser



def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    control = _selected_task_control(args)
    interactive = args.interactive or (args.prompt is None and control is None)
    if args.interactive and args.prompt is not None:
        print("Argument error: prompt cannot be used with --interactive.", file=sys.stderr)
        return 2
    if control is not None and (args.interactive or args.prompt is not None):
        print("Argument error: task controls cannot be combined with a prompt or --interactive.", file=sys.stderr)
        return 2

    try:
        config = load_config(workspace_root=args.workspace_root)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    initialize_database(config.database_path)
    sessions = SessionService(config.database_path)
    session_state_path = _session_state_path(config.database_path)
    resolved_session_id = _resolve_session_id(
        explicit_session_id=args.session_id,
        new_session=args.new_session,
        session_state_path=session_state_path,
    )
    model_client = OpenAICompatibleModelClient.from_config(config)
    subagent_runner = SubagentRunner(
        model_client=model_client,
        session_service=sessions,
        workspace_root=config.workspace_root,
        tool_timeout=config.tool_timeout,
        context_token_budget=config.context_token_budget,
        summary_trigger_tokens=config.summary_trigger_tokens,
        confirmation_handler=prompt_for_tool_confirmation,
    )
    loop = AgentLoop(
        agent=SINGLE_MAIN_AGENT,
        model_client=model_client,
        tool_registry=build_root_registry(
            subagent_runner=subagent_runner,
            web_search_tool=WebSearchTool(
                base_url=config.search_base_url,
                api_key=config.search_api_key,
                timeout=config.search_timeout,
                max_results=config.search_max_results,
            ),
        ),
        session_service=sessions,
        workspace_root=config.workspace_root,
        tool_timeout=config.tool_timeout,
        context_token_budget=config.context_token_budget,
        summary_trigger_tokens=config.summary_trigger_tokens,
        confirmation_handler=prompt_for_tool_confirmation,
    )
    if control is not None:
        action, task_id = control
        try:
            task = loop.get_task(task_id)
        except KeyError:
            print(f"Task error: task '{task_id}' was not found.", file=sys.stderr)
            return 1
        if action == "status":
            print(json.dumps(task, default=_serialize, ensure_ascii=False))
            return 0
        event_type = {
            "pause": "pause_requested",
            "resume": "resume_requested",
            "cancel": "cancel_requested",
            "approve": "user_approved",
            "reject": "user_rejected",
        }[action]
        try:
            event_payload = {}
            if action in {"approve", "reject"} and task.pending_action is not None:
                event_payload["pending_action_id"] = task.pending_action.id
            result = loop.handle_event(
                AgentEvent(
                    id=str(uuid4()),
                    task_id=task.id,
                    session_id=task.session_id,
                    type=event_type,
                    source="cli",
                    payload=event_payload,
                    correlation_id=task.id,
                    expected_version=task.version,
                )
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Task error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, default=_serialize, ensure_ascii=False))
        return 0 if result.success or result.task_status in {"paused", "cancelled", "waiting_user"} else 1
    if interactive:
        interactive_session_id = sessions.get_or_create_session(resolved_session_id)
        _persist_current_session(session_state_path, interactive_session_id)
        return _run_interactive_loop(
            loop=loop,
            session_service=sessions,
            session_id=interactive_session_id,
            session_state_path=session_state_path,
        )

    try:
        result = loop.run_turn(user_input=args.prompt, session_id=resolved_session_id)
    except ActiveTaskConflict as exc:
        print(f"Task error: {exc}", file=sys.stderr)
        return 1
    _persist_current_session(session_state_path, result.session_id)
    print(json.dumps(result, default=_serialize, ensure_ascii=False))
    return 0 if result.success else 1



def prompt_for_tool_confirmation(tool_call: ToolCall, context: ToolExecutionContext) -> bool:
    prompt_text = _build_confirmation_prompt(tool_call, context)
    print(prompt_text)
    decision = input("Approve this action? [y/N]: ").strip().lower()
    return decision in {"y", "yes"}



def _build_confirmation_prompt(tool_call: ToolCall, context: ToolExecutionContext) -> str:
    if tool_call.name == "shell":
        inspection = context.prepared_edits.get(tool_call.id)
        if not isinstance(inspection, ShellInspection):
            inspection, error = ShellTool().inspect(arguments=tool_call.arguments, context=context)
            if inspection is None:
                return f"Shell action confirmation\nValidation unavailable: {error}"
        return (
            "Shell action confirmation\n"
            f"Risk: {inspection.controlled.risk_level} workspace change\n"
            f"Operation: {inspection.controlled.operation}\n"
            f"Command: {inspection.controlled.command}\n"
            "Affected paths:\n" + "\n".join(f"- {path}" for path in inspection.paths)
        )
    if tool_call.name == "file_write":
        inspection, error = _resolve_file_write_confirmation_inspection(tool_call=tool_call, context=context)
        if inspection is None:
            raw_path = str(tool_call.arguments.get("path", "<unknown>"))
            raw_content = tool_call.arguments.get("content", "")
            byte_count = len(raw_content.encode("utf-8")) if isinstance(raw_content, str) else 0
            line_count = raw_content.count("\n") + 1 if isinstance(raw_content, str) and raw_content else 0
            preview = raw_content[:800] if isinstance(raw_content, str) else ""
            return (
                "Text edit confirmation\n"
                f"Operation: unknown\n"
                f"Path: {raw_path}\n"
                f"Size: {byte_count} bytes / {line_count} lines\n"
                f"Validation preview unavailable: {error}\n"
                f"Preview:\n{preview or '<empty>'}"
            )

        diff_summary = inspection.diff_summary()
        detail = diff_summary if inspection.operation == "overwrite" and diff_summary else (inspection.preview() or "<empty>")
        return (
            "Text edit confirmation\n"
            f"Operation: {inspection.operation}\n"
            f"Path: {inspection.relative_path}\n"
            f"Size: {inspection.byte_count} bytes / {inspection.line_count} lines\n"
            f"Details:\n{detail}"
        )

    if tool_call.name == "replace_in_file":
        inspection, error = _resolve_replace_confirmation_inspection(tool_call=tool_call, context=context)
        if inspection is None:
            raw_path = str(tool_call.arguments.get("path", "<unknown>"))
            return (
                "Text edit confirmation\n"
                "Operation: replace\n"
                f"Path: {raw_path}\n"
                f"Validation preview unavailable: {error}"
            )

        return (
            "Text edit confirmation\n"
            "Operation: replace\n"
            f"Path: {inspection.relative_path}\n"
            f"Matches: {inspection.match_count}\n"
            f"Replacements: {inspection.replacement_count}\n"
            f"Size: {inspection.byte_count} bytes / {inspection.line_count} lines\n"
            f"Details:\n{inspection.diff_preview}"
        )

    if tool_call.name in {"file_write", "replace_in_file"}:
        return "Text edit confirmation"
    if tool_call.name != "file_write":
        return f"Tool '{tool_call.name}' requires confirmation."
    return f"Tool '{tool_call.name}' requires confirmation."


def _resolve_file_write_confirmation_inspection(
    *,
    tool_call: ToolCall,
    context: ToolExecutionContext,
):
    cached = context.prepared_edits.get(tool_call.id)
    if cached is not None:
        return cached, None
    return inspect_file_write_request(arguments=tool_call.arguments, context=context)


def _resolve_replace_confirmation_inspection(
    *,
    tool_call: ToolCall,
    context: ToolExecutionContext,
) -> tuple[ReplaceInFileInspection | None, str | None]:
    cached = context.prepared_edits.get(tool_call.id)
    if isinstance(cached, ReplaceInFileInspection):
        return cached, None
    return inspect_replace_in_file_request(arguments=tool_call.arguments, context=context)



def _serialize(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def _selected_task_control(args) -> tuple[str, str] | None:
    for action, attribute in (
        ("status", "task_status"),
        ("pause", "pause_task"),
        ("resume", "resume_task"),
        ("cancel", "cancel_task"),
        ("approve", "approve_task"),
        ("reject", "reject_task"),
    ):
        value = getattr(args, attribute, None)
        if value:
            return action, value
    return None


def _session_state_path(database_path: str | Path) -> Path:
    return Path(database_path).resolve().parent / "current_session.txt"


def _resolve_session_id(
    *,
    explicit_session_id: str | None,
    new_session: bool,
    session_state_path: Path,
) -> str | None:
    if explicit_session_id:
        return explicit_session_id
    if new_session:
        return None
    if not session_state_path.exists():
        return None

    session_id = session_state_path.read_text(encoding="utf-8").strip()
    return session_id or None


def _persist_current_session(session_state_path: Path, session_id: str) -> None:
    session_state_path.parent.mkdir(parents=True, exist_ok=True)
    session_state_path.write_text(session_id, encoding="utf-8")


def _run_interactive_loop(
    *,
    loop: AgentLoop,
    session_service: SessionService,
    session_id: str,
    session_state_path: Path,
) -> int:
    current_session_id = session_id
    print(f"Interactive mode. Session: {current_session_id}")
    print("Type /help for commands; use 'exit' or 'quit' to leave.")

    while True:
        try:
            user_input = input("> ")
        except EOFError:
            print()
            return 0

        stripped = user_input.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered in {"exit", "quit"}:
            return 0
        if lowered in {":new", "/new"}:
            current_session_id = session_service.create_session()
            _persist_current_session(session_state_path, current_session_id)
            print(f"Started a new session. Session: {current_session_id}")
            continue
        if lowered == "/help":
            print(_REPL_HELP)
            continue
        if lowered == "/task":
            _print_latest_task(session_service, current_session_id)
            continue
        if lowered == "/tasks":
            _print_task_list(session_service, current_session_id)
            continue
        command, _, raw_target = stripped.partition(" ")
        command = command.lower()
        if command in {"/approve", "/reject", "/cancel"}:
            result = _run_repl_task_control(
                loop=loop,
                session_service=session_service,
                session_id=current_session_id,
                command=command,
                task_prefix=raw_target.strip() or None,
            )
            if result is not None:
                _print_turn_result(result)
            continue
        if stripped.startswith("/"):
            print(f"Unknown command: {stripped}. Type /help for commands.")
            continue

        try:
            result = loop.run_turn(user_input=stripped, session_id=current_session_id)
        except ActiveTaskConflict as exc:
            print(f"Task error: {exc}")
            continue
        current_session_id = result.session_id
        _persist_current_session(session_state_path, result.session_id)
        _print_turn_result(result)


def _run_repl_task_control(
    *,
    loop: AgentLoop,
    session_service: SessionService,
    session_id: str,
    command: str,
    task_prefix: str | None = None,
) -> TurnResult | None:
    tasks = session_service.list_tasks(session_id)
    if command in {"/approve", "/reject"}:
        candidates = [
            item
            for item in reversed(tasks)
            if item.status == "waiting_user"
            and item.pending_action is not None
            and item.pending_action.kind == "tool_approval"
        ]
        if not candidates:
            print("No task is waiting for tool approval in this session.")
            return None
        task = _select_task_candidate(
            candidates,
            task_prefix=task_prefix,
            command=command,
        )
        if task is None:
            return None
        event_type = "user_approved" if command == "/approve" else "user_rejected"
    else:
        candidates = [
            item
            for item in reversed(tasks)
            if item.status not in _TERMINAL_TASK_STATUSES
        ]
        task = _select_task_candidate(
            candidates,
            task_prefix=task_prefix,
            command=command,
        )
        if task is None:
            if not candidates:
                print("No non-terminal task exists in this session.")
            return None
        event_type = "cancel_requested"

    try:
        event_payload = {}
        if command in {"/approve", "/reject"} and task.pending_action is not None:
            event_payload["pending_action_id"] = task.pending_action.id
        return loop.handle_event(
            AgentEvent(
                id=str(uuid4()),
                task_id=task.id,
                session_id=task.session_id,
                type=event_type,
                source="repl",
                payload=event_payload,
                correlation_id=task.id,
                expected_version=task.version,
            )
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Task error: {exc}")
        return None


def _select_task_candidate(
    candidates: list[TaskState],
    *,
    task_prefix: str | None,
    command: str,
) -> TaskState | None:
    if task_prefix is not None:
        matches = [task for task in candidates if task.id.startswith(task_prefix)]
        if not matches:
            print(f"No task matches prefix '{task_prefix}' for {command}.")
            return None
        if len(matches) == 1:
            return matches[0]
        print(f"Task prefix '{task_prefix}' is ambiguous for {command}:")
        for task in matches:
            _print_task_summary(task)
        return None

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(f"Multiple tasks are eligible for {command}; specify a task-id prefix:")
        for task in candidates:
            _print_task_summary(task)
    return None


def _print_latest_task(session_service: SessionService, session_id: str) -> None:
    task = session_service.get_latest_task(session_id)
    if task is None:
        print("No tasks in this session.")
        return
    _print_task_summary(task)


def _print_task_list(session_service: SessionService, session_id: str) -> None:
    tasks = session_service.list_tasks(session_id)
    if not tasks:
        print("No tasks in this session.")
        return
    for task in tasks:
        _print_task_summary(task)


def _print_task_summary(task: TaskState) -> None:
    print(f"[task: {task.id} | {task.status}] {task.goal}")
    if task.pending_action is not None:
        print(
            f"  pending: {task.pending_action.kind} "
            f"[{task.pending_action.id}] - {task.pending_action.prompt}"
        )
    if task.stop_reason:
        print(f"  stop_reason: {task.stop_reason}")


def _print_turn_result(result: TurnResult) -> None:
    if result.final_text:
        print(result.final_text)
    elif not result.success:
        print(f"Error: {result.stop_reason or 'unknown_error'}")
    if result.task_id is not None:
        print(f"[task: {result.task_id} | {result.task_status or 'unknown'}]")


if __name__ == "__main__":
    raise SystemExit(main())
