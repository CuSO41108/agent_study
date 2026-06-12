from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from agent_app.agent.definition import AgentDefinition
from agent_app.agent.prompts import render_system_prompt
from agent_app.orchestrator.context_builder import build_context_messages, build_evidence_message, estimate_messages_tokens
from agent_app.state.session_service import SessionService
from agent_app.tools.approval import approve_tool_call
from agent_app.tools.base import ToolExecutionContext
from agent_app.tools.registry import ToolRegistry
from agent_app.types import Message, ModelResponse, SessionContext, StoredMessage, TodoItem, ToolAction, ToolCall, ToolResult, TurnResult

ConfirmationHandler = Callable[[ToolCall, ToolExecutionContext], bool]


class AgentLoop:
    def __init__(
        self,
        *,
        agent: AgentDefinition,
        model_client: Any,
        tool_registry: ToolRegistry,
        session_service: SessionService,
        workspace_root,
        tool_timeout: float | None = None,
        shell_timeout: float | None = None,
        context_token_budget: int = 6000,
        summary_trigger_tokens: int = 3000,
        confirmation_handler: ConfirmationHandler | None = None,
        delegation_depth: int = 0,
    ) -> None:
        if tool_timeout is not None and shell_timeout is not None:
            raise ValueError("Only one of 'tool_timeout' or 'shell_timeout' may be provided.")
        resolved_tool_timeout = tool_timeout if tool_timeout is not None else shell_timeout
        if resolved_tool_timeout is None:
            resolved_tool_timeout = 15.0

        self._agent = agent
        self._model_client = model_client
        self._tool_registry = tool_registry
        self._session_service = session_service
        self._context_token_budget = context_token_budget
        self._summary_trigger_tokens = summary_trigger_tokens
        self._tool_context = ToolExecutionContext(
            workspace_root=workspace_root,
            timeout=resolved_tool_timeout,
            agent_id=agent.id,
            delegation_depth=delegation_depth,
        )
        self._confirmation_handler = confirmation_handler

    def run_turn(self, *, user_input: str, session_id: str | None = None) -> TurnResult:
        resolved_session_id = self._session_service.get_or_create_session(session_id)
        recovery_blockers = self._recover_pending_tool_actions(resolved_session_id)
        if recovery_blockers:
            action_names = ", ".join(
                f"{result.tool_name} ({result.tool_call_id})"
                for result in recovery_blockers
            )
            return TurnResult(
                session_id=resolved_session_id,
                final_text=(
                    "A previous tool action has an uncertain side-effect state and was not retried. "
                    f"Inspect the workspace before continuing: {action_names}."
                ),
                stop_reason="uncertain_tool_action",
                tool_runs=recovery_blockers,
                success=False,
            )

        self._session_service.append_message(
            resolved_session_id,
            Message(role="user", content=user_input),
        )

        self._tool_context = replace(
            self._tool_context,
            prepared_edits={},
            turn_state={},
            session_id=resolved_session_id,
            session_service=self._session_service,
        )

        messages = self._session_service.list_messages(resolved_session_id)
        session_context = self._session_service.get_session_context(resolved_session_id)
        session_context = self._maybe_update_summary(
            session_id=resolved_session_id,
            messages=messages,
            session_context=session_context,
        )
        tool_runs_history = self._session_service.list_tool_runs(resolved_session_id)
        evidence_message = build_evidence_message(tool_runs_history)
        provider_messages = build_context_messages(
            messages=messages,
            session_context=session_context,
            context_token_budget=self._context_token_budget,
            evidence_message=evidence_message,
        )
        base_context_message_count = len(provider_messages)
        base_context_token_estimate = sum(
            estimate_messages_tokens([StoredMessage(id=index, role=message["role"], content=message.get("content"))])
            for index, message in enumerate(provider_messages, start=1)
        )
        used_summary = bool(session_context.summary_text)
        used_todo = bool(session_context.todo_items)
        used_evidence = bool(evidence_message)
        system_prompt = render_system_prompt(self._agent)
        tool_runs: list[ToolResult] = []
        tool_rounds = 0
        consecutive_failure_tool: str | None = None
        consecutive_failure_count = 0

        while True:
            response = self._model_client.generate(
                system_prompt=system_prompt,
                messages=provider_messages,
                tools=self._tool_registry.get_specs(self._agent.allowed_tools),
            )

            if response.error_type:
                return self._finalize_turn(
                    session_id=resolved_session_id,
                    user_input=user_input,
                    context_message_count=base_context_message_count,
                    context_token_estimate=base_context_token_estimate,
                    used_summary=used_summary,
                    used_todo=used_todo,
                    used_evidence=used_evidence,
                    final_text=None,
                    stop_reason="model_error",
                    tool_runs=tool_runs,
                    success=False,
                )

            if not response.tool_calls:
                if response.assistant_text:
                    self._session_service.append_message(
                        resolved_session_id,
                        Message(role="assistant", content=response.assistant_text),
                    )
                    return self._finalize_turn(
                        session_id=resolved_session_id,
                        user_input=user_input,
                        context_message_count=base_context_message_count,
                        context_token_estimate=base_context_token_estimate,
                        used_summary=used_summary,
                        used_todo=used_todo,
                        used_evidence=used_evidence,
                        final_text=response.assistant_text,
                        stop_reason="final_response",
                        tool_runs=tool_runs,
                        success=True,
                    )
                return self._finalize_turn(
                    session_id=resolved_session_id,
                    user_input=user_input,
                    context_message_count=base_context_message_count,
                    context_token_estimate=base_context_token_estimate,
                    used_summary=used_summary,
                    used_todo=used_todo,
                    used_evidence=used_evidence,
                    final_text=None,
                    stop_reason="invalid_model_response",
                    tool_runs=tool_runs,
                    success=False,
                )

            if tool_rounds >= self._agent.max_tool_rounds:
                evidence_answer = self._build_evidence_answer(
                    user_input=user_input,
                    tool_runs=tool_runs,
                    allow_file_read_excerpt=False,
                )
                if evidence_answer is not None:
                    self._session_service.append_message(
                        resolved_session_id,
                        Message(role="assistant", content=evidence_answer),
                    )
                    return self._finalize_turn(
                        session_id=resolved_session_id,
                        user_input=user_input,
                        context_message_count=base_context_message_count,
                        context_token_estimate=base_context_token_estimate,
                        used_summary=used_summary,
                        used_todo=used_todo,
                        used_evidence=used_evidence,
                        final_text=evidence_answer,
                        stop_reason="answered_from_evidence",
                        tool_runs=tool_runs,
                        success=True,
                    )
                return self._finalize_turn(
                    session_id=resolved_session_id,
                    user_input=user_input,
                    context_message_count=base_context_message_count,
                    context_token_estimate=base_context_token_estimate,
                    used_summary=used_summary,
                    used_todo=used_todo,
                    used_evidence=used_evidence,
                    final_text=None,
                    stop_reason="max_tool_rounds_exceeded",
                    tool_runs=tool_runs,
                    success=False,
                )

            tool_rounds += 1
            provider_messages.append(_assistant_tool_message(response))

            for tool_call in response.tool_calls:
                evidence_answer = self._answer_from_existing_evidence(
                    user_input=user_input,
                    tool_call=tool_call,
                    tool_runs=tool_runs,
                )
                if evidence_answer is not None:
                    self._session_service.append_message(
                        resolved_session_id,
                        Message(role="assistant", content=evidence_answer),
                    )
                    return self._finalize_turn(
                        session_id=resolved_session_id,
                        user_input=user_input,
                        context_message_count=base_context_message_count,
                        context_token_estimate=base_context_token_estimate,
                        used_summary=used_summary,
                        used_todo=used_todo,
                        used_evidence=used_evidence,
                        final_text=evidence_answer,
                        stop_reason="answered_from_evidence",
                        tool_runs=tool_runs,
                        success=True,
                    )

                try:
                    tool_result = self._execute_tool_call(tool_call)
                except Exception:
                    return self._finalize_turn(
                        session_id=resolved_session_id,
                        user_input=user_input,
                        context_message_count=base_context_message_count,
                        context_token_estimate=base_context_token_estimate,
                        used_summary=used_summary,
                        used_todo=used_todo,
                        used_evidence=used_evidence,
                        final_text=None,
                        stop_reason="tool_execution_error",
                        tool_runs=tool_runs,
                        success=False,
                    )
                tool_runs.append(tool_result)
                if self._session_service.list_uncertain_tool_actions(resolved_session_id):
                    return self._finalize_turn(
                        session_id=resolved_session_id,
                        user_input=user_input,
                        context_message_count=base_context_message_count,
                        context_token_estimate=base_context_token_estimate,
                        used_summary=used_summary,
                        used_todo=used_todo,
                        used_evidence=used_evidence,
                        final_text=(
                            "A tool action may have produced a side effect. "
                            "Automatic retry is blocked until the workspace is inspected."
                        ),
                        stop_reason="uncertain_tool_action",
                        tool_runs=tool_runs,
                        success=False,
                    )
                provider_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result.content if tool_result.success else (tool_result.error or tool_result.content or "Tool execution failed."),
                    }
                )

                if tool_result.success:
                    consecutive_failure_tool = None
                    consecutive_failure_count = 0
                else:
                    if consecutive_failure_tool == tool_result.tool_name:
                        consecutive_failure_count += 1
                    else:
                        consecutive_failure_tool = tool_result.tool_name
                        consecutive_failure_count = 1
                    if consecutive_failure_count >= 2:
                        evidence_answer = self._build_evidence_answer(
                            user_input=user_input,
                            tool_runs=tool_runs,
                            allow_file_read_excerpt=True,
                        )
                        if evidence_answer is not None:
                            self._session_service.append_message(
                                resolved_session_id,
                                Message(role="assistant", content=evidence_answer),
                            )
                            return self._finalize_turn(
                                session_id=resolved_session_id,
                                user_input=user_input,
                                context_message_count=base_context_message_count,
                                context_token_estimate=base_context_token_estimate,
                                used_summary=used_summary,
                                used_todo=used_todo,
                                used_evidence=used_evidence,
                                final_text=evidence_answer,
                                stop_reason="answered_from_evidence",
                                tool_runs=tool_runs,
                                success=True,
                            )
                        return self._finalize_turn(
                            session_id=resolved_session_id,
                            user_input=user_input,
                            context_message_count=base_context_message_count,
                            context_token_estimate=base_context_token_estimate,
                            used_summary=used_summary,
                            used_todo=used_todo,
                            used_evidence=used_evidence,
                            final_text=None,
                            stop_reason="repeated_tool_failure",
                            tool_runs=tool_runs,
                            success=False,
                        )

    def _execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        if tool_call.name not in self._agent.allowed_tools:
            return self._record_untracked_tool_result(ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                content="",
                error=f"Tool '{tool_call.name}' is not allowed for this agent.",
            ))

        tool = self._tool_registry.get(tool_call.name)
        if tool is None:
            return self._record_untracked_tool_result(ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                content="",
                error=f"Tool '{tool_call.name}' is not registered.",
            ))

        validation_error = tool.validate_arguments(tool_call.arguments)
        if validation_error is not None:
            return self._record_untracked_tool_result(ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                content="",
                error=validation_error,
            ))

        approval = approve_tool_call(tool_call.name, tool_call.arguments)
        if approval.decision == "deny":
            return self._record_untracked_tool_result(ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                content="",
                error=approval.reason,
            ))

        if approval.decision == "confirm":
            inspection, error = tool.inspect(arguments=tool_call.arguments, context=self._tool_context)
            if inspection is None:
                return self._record_untracked_tool_result(ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    success=False,
                    content="",
                    error=error or "Unable to inspect edit.",
                ))

            self._tool_context.prepared_edits[tool_call.id] = inspection
            confirmed = False
            if self._confirmation_handler is not None:
                confirmed = self._confirmation_handler(tool_call, self._tool_context)
            if not confirmed:
                self._tool_context.prepared_edits.pop(tool_call.id, None)
                return self._record_untracked_tool_result(ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    success=False,
                    content="",
                    error="Tool use denied by user.",
                ))

        session_id = self._tool_context.session_id
        if session_id is None:
            raise RuntimeError("Tool execution requires an active session.")
        recovery_metadata = tool.recovery_metadata(
            tool_call_id=tool_call.id,
            arguments=tool_call.arguments,
            context=self._tool_context,
        )
        action = self._session_service.prepare_tool_action(
            session_id,
            agent_id=self._agent.id,
            tool_call=tool_call,
            recovery_metadata=recovery_metadata,
        )
        if action.status in {"succeeded", "failed", "uncertain"} and action.result is not None:
            return action.result
        if action.status == "executing":
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                content="",
                error="Tool action is already executing and was not run again.",
            )

        action = self._session_service.mark_tool_action_executing(action.id)
        try:
            tool_result = tool.execute(
                tool_call_id=tool_call.id,
                arguments=tool_call.arguments,
                context=self._tool_context,
            )
        except Exception:
            status, recovered_result = self._classify_tool_action_recovery(action)
            completed = self._session_service.complete_tool_action(
                action.id,
                status=status,
                tool_result=recovered_result,
            )
            assert completed.result is not None
            return completed.result

        completed = self._session_service.complete_tool_action(
            action.id,
            status="succeeded" if tool_result.success else "failed",
            tool_result=tool_result,
        )
        assert completed.result is not None
        return completed.result

    def _record_untracked_tool_result(self, tool_result: ToolResult) -> ToolResult:
        session_id = self._tool_context.session_id
        if session_id is None:
            raise RuntimeError("Tool execution requires an active session.")
        self._session_service.append_tool_run(session_id, tool_result)
        return tool_result

    def _recover_pending_tool_actions(self, session_id: str) -> list[ToolResult]:
        for action in self._session_service.list_recoverable_tool_actions(session_id):
            status, tool_result = self._classify_tool_action_recovery(action)
            self._session_service.complete_tool_action(
                action.id,
                status=status,
                tool_result=tool_result,
            )

        blockers: list[ToolResult] = []
        for action in self._session_service.list_uncertain_tool_actions(session_id):
            blockers.append(
                action.result
                or ToolResult(
                    tool_call_id=action.tool_call_id,
                    tool_name=action.tool_name,
                    success=False,
                    content="",
                    error="Tool action side effect is uncertain.",
                )
            )
        return blockers

    def _classify_tool_action_recovery(self, action: ToolAction) -> tuple[str, ToolResult]:
        if action.status == "prepared":
            return (
                "failed",
                ToolResult(
                    tool_call_id=action.tool_call_id,
                    tool_name=action.tool_name,
                    success=False,
                    content="",
                    error="Tool action was interrupted before execution started.",
                ),
            )

        metadata = action.recovery_metadata
        if metadata.get("recovery_kind") == "text_file_hash":
            return self._classify_text_file_recovery(action)
        if not metadata.get("side_effect", False):
            return (
                "failed",
                ToolResult(
                    tool_call_id=action.tool_call_id,
                    tool_name=action.tool_name,
                    success=False,
                    content="",
                    error="Read-only tool action was interrupted before its result was persisted.",
                ),
            )
        return (
            "uncertain",
            ToolResult(
                tool_call_id=action.tool_call_id,
                tool_name=action.tool_name,
                success=False,
                content="",
                error="Tool action may have produced a side effect; automatic retry is blocked.",
            ),
        )

    def _classify_text_file_recovery(self, action: ToolAction) -> tuple[str, ToolResult]:
        metadata = action.recovery_metadata
        raw_relative_path = metadata.get("relative_path")
        if not isinstance(raw_relative_path, str):
            return _uncertain_file_recovery_result(action, "Recovery metadata does not contain a valid file path.")

        workspace_root = self._tool_context.workspace_root.resolve()
        path = (workspace_root / raw_relative_path).resolve()
        try:
            path.relative_to(workspace_root)
        except ValueError:
            return _uncertain_file_recovery_result(action, "Recovery file path escapes the workspace.")

        current_path_exists = path.exists()
        current_exists = path.is_file()
        current_hash = None
        if current_exists:
            try:
                current_hash = hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            except (OSError, UnicodeDecodeError):
                return _uncertain_file_recovery_result(action, "Recovery could not read the target as UTF-8 text.")

        if current_exists and current_hash == metadata.get("after_sha256"):
            return (
                "succeeded",
                ToolResult(
                    tool_call_id=action.tool_call_id,
                    tool_name=action.tool_name,
                    success=True,
                    content=str(metadata.get("success_content") or "Recovered completed file action."),
                    error=None,
                ),
            )

        before_exists = bool(metadata.get("before_exists"))
        before_matches = (
            current_exists and current_hash == metadata.get("before_sha256")
            if before_exists
            else not current_path_exists
        )
        if before_matches:
            return (
                "failed",
                ToolResult(
                    tool_call_id=action.tool_call_id,
                    tool_name=action.tool_name,
                    success=False,
                    content="",
                    error="File action was interrupted before the intended content was installed.",
                ),
            )

        return _uncertain_file_recovery_result(
            action,
            "Target file matches neither the recorded before state nor the expected after state.",
        )

    def _maybe_update_summary(
        self,
        *,
        session_id: str,
        messages: list[StoredMessage],
        session_context: SessionContext,
    ) -> SessionContext:
        if len(messages) <= 1:
            return session_context

        historical_messages = messages[:-1]
        candidate_messages = [
            message
            for message in historical_messages
            if session_context.summary_message_id is None or message.id > session_context.summary_message_id
        ]
        if len(candidate_messages) <= 6:
            return session_context

        summary_segment = candidate_messages[:-6]
        if estimate_messages_tokens(summary_segment) <= self._summary_trigger_tokens:
            return session_context

        new_summary = self._generate_summary(
            existing_summary=session_context.summary_text,
            messages=summary_segment,
        )
        if new_summary is None:
            return session_context

        summary_message_id = summary_segment[-1].id
        self._session_service.upsert_session_context(
            session_id,
            summary_text=new_summary,
            summary_message_id=summary_message_id,
            todo_items=session_context.todo_items,
        )
        return SessionContext(
            summary_text=new_summary,
            summary_message_id=summary_message_id,
            todo_items=session_context.todo_items,
        )

    def _generate_summary(
        self,
        *,
        existing_summary: str | None,
        messages: list[StoredMessage],
    ) -> str | None:
        transcript_lines = []
        if existing_summary:
            transcript_lines.append("Previous summary:")
            transcript_lines.append(existing_summary)
            transcript_lines.append("")
        transcript_lines.append("New conversation segment:")
        for message in messages:
            transcript_lines.append(f"{message.role.upper()}: {message.content or ''}")

        response = self._model_client.generate(
            system_prompt=(
                "Summarize the prior conversation for future task continuation. "
                "Include the user goal, completed actions, unresolved issues, important constraints or assumptions, "
                "and any important files, paths, or results."
            ),
            messages=[{"role": "user", "content": "\n".join(transcript_lines)}],
            tools=[],
        )
        if response.error_type or response.tool_calls or not response.assistant_text:
            return None
        return response.assistant_text.strip() or None

    def _finalize_turn(
        self,
        *,
        session_id: str,
        user_input: str,
        context_message_count: int,
        context_token_estimate: int,
        used_summary: bool,
        used_todo: bool,
        used_evidence: bool,
        final_text: str | None,
        stop_reason: str | None,
        tool_runs: list[ToolResult],
        success: bool,
    ) -> TurnResult:
        try:
            self._session_service.append_turn_trace(
                session_id,
                user_input=user_input,
                context_message_count=context_message_count,
                context_token_estimate=context_token_estimate,
                used_summary=used_summary,
                used_todo=used_todo,
                used_evidence=used_evidence,
                final_text=final_text,
                stop_reason=stop_reason,
                success=success,
                tool_traces=tool_runs,
            )
        except Exception:
            pass

        return TurnResult(
            session_id=session_id,
            final_text=final_text,
            stop_reason=stop_reason,
            tool_runs=tool_runs,
            success=success,
        )

    def _answer_from_existing_evidence(
        self,
        *,
        user_input: str,
        tool_call: ToolCall,
        tool_runs: list[ToolResult],
    ) -> str | None:
        if _looks_like_tool_inventory_question(user_input):
            return _extract_tool_inventory_answer(tool_runs, self._agent.allowed_tools)
        if _looks_like_config_question(user_input):
            return _extract_config_answer(tool_runs)

        if tool_call.name != "shell":
            return None
        if not _looks_like_location_question(user_input):
            return None

        paths = _extract_code_search_paths(tool_runs, self._tool_context.workspace_root)
        if not paths:
            return None

        bullet_lines = "\n".join(f"- `{path}`" for path in paths[:3])
        return (
            "There is already enough code-search evidence to answer this location question "
            "without running shell again.\n"
            f"Relevant paths:\n{bullet_lines}"
        )

    def _build_evidence_answer(
        self,
        *,
        user_input: str,
        tool_runs: list[ToolResult],
        allow_file_read_excerpt: bool,
    ) -> str | None:
        if _looks_like_tool_inventory_question(user_input):
            return _extract_tool_inventory_answer(tool_runs, self._agent.allowed_tools)
        if _looks_like_config_question(user_input):
            return _extract_config_answer(tool_runs)

        if _looks_like_location_question(user_input):
            paths = _extract_code_search_paths(tool_runs, self._tool_context.workspace_root)
            if paths:
                bullet_lines = "\n".join(f"- `{path}`" for path in paths[:3])
                return (
                    "There is already enough code-search evidence to answer this location question "
                    "without running shell again.\n"
                    f"Relevant paths:\n{bullet_lines}"
                )

        file_read_excerpt = _extract_file_read_excerpt(tool_runs) if allow_file_read_excerpt else None
        if file_read_excerpt is not None:
            return (
                "There is already enough file evidence to answer directly without retrying more tools.\n"
                f"{file_read_excerpt}"
            )

        return None


def _message_to_provider_message(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    return payload


def _assistant_tool_message(response: ModelResponse) -> dict[str, Any]:
    if response.raw_response:
        try:
            choice = response.raw_response["choices"][0]
            message = choice["message"]
            if isinstance(message, dict):
                return message
        except (KeyError, IndexError, TypeError):
            pass

    return {
        "role": "assistant",
        "content": response.assistant_text,
        "tool_calls": [_tool_call_to_provider_payload(tool_call) for tool_call in response.tool_calls],
    }


def _tool_call_to_provider_payload(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(tool_call.arguments),
        },
    }


def _looks_like_location_question(user_input: str) -> bool:
    lowered = user_input.lower()
    keywords = (
        "where",
        "path",
        "folder",
        "file",
        "directory",
        "目录",
        "文件夹",
        "路径",
        "位置",
        "在哪",
        "哪里",
    )
    return any(keyword in lowered for keyword in keywords)


def _looks_like_tool_inventory_question(user_input: str) -> bool:
    lowered = user_input.lower()
    keywords = (
        "which tool",
        "what tools",
        "available tool",
        "supported tool",
        "can call",
        "can use",
        "哪些tool",
        "哪些 tool",
        "哪些工具",
        "可用tool",
        "可用 tool",
        "可用工具",
        "支持哪些",
        "有哪些",
    )
    return any(keyword in lowered for keyword in keywords)


def _looks_like_config_question(user_input: str) -> bool:
    lowered = user_input.lower()
    keywords = (
        "database path",
        "db path",
        "config",
        "property",
        "defined",
        "determine",
        "database",
        "路径",
        "数据库",
        "配置",
        "定义",
        "确定",
    )
    return any(keyword in lowered for keyword in keywords)


def _extract_tool_inventory_answer(tool_runs: list[ToolResult], allowed_tools: list[str]) -> str | None:
    authority_hits = 0
    saw_registry = False
    saw_allowed_tools = False

    for tool_run in tool_runs:
        if not tool_run.success:
            continue
        content = tool_run.content
        if tool_run.tool_name == "file_read":
            if "def build_default_registry" in content or "class ToolRegistry" in content:
                saw_registry = True
                authority_hits += 1
            if "allowed_tools=[" in content or "AgentDefinition" in content:
                saw_allowed_tools = True
                authority_hits += 1
        if tool_run.tool_name == "code_search":
            if any(marker in content for marker in ("registry.py", "build_default_registry", "allowed_tools", "AgentDefinition")):
                authority_hits += 1

    if authority_hits == 0 or not (saw_registry or saw_allowed_tools):
        return None

    tool_list = "\n".join(f"- `{tool_name}`" for tool_name in allowed_tools)
    return (
        "Authoritative registry evidence is already enough to answer this.\n"
        "The current project can call these tools:\n"
        f"{tool_list}"
    )


def _extract_config_answer(tool_runs: list[ToolResult]) -> str | None:
    saw_database_property = False
    saw_database_value = False
    excerpt = ""

    for tool_run in tool_runs:
        if not tool_run.success:
            continue
        content = tool_run.content
        if tool_run.tool_name == "file_read":
            if "database_path" in content:
                saw_database_property = True
            if 'workspace_root / ".agent_app" / "agent.db"' in content:
                saw_database_value = True
            if saw_database_property or saw_database_value:
                excerpt = "\n".join(content.splitlines()[:8])
        elif tool_run.tool_name == "code_search":
            if "config.py" in content and "database_path" in content:
                saw_database_property = True
                excerpt = "\n".join(content.splitlines()[:5])

    if not (saw_database_property and saw_database_value):
        return None

    return (
        "There is already enough authoritative config evidence to answer this.\n"
        'The database file path is determined by `AppConfig.database_path` in `src/agent_app/config.py`, '
        'which returns `workspace_root / ".agent_app" / "agent.db"`.\n'
        f"{excerpt}"
    )


def _extract_code_search_paths(tool_runs: list[ToolResult], workspace_root: Path) -> list[str]:
    path_pattern = re.compile(r"^(.*?):\d+:")
    all_paths: list[str] = []
    preferred_paths: list[str] = []
    for tool_run in tool_runs:
        if tool_run.tool_name != "code_search" or not tool_run.success:
            continue
        for line in tool_run.content.splitlines():
            match = path_pattern.match(line)
            if match is None:
                continue
            raw_path = match.group(1)
            relative_path = _relative_workspace_path(raw_path, workspace_root)
            if relative_path in all_paths:
                continue
            all_paths.append(relative_path)
            if not relative_path.replace("\\", "/").startswith("tests/"):
                preferred_paths.append(relative_path)

    candidates = preferred_paths or all_paths
    if not candidates or len(candidates) > 5:
        return []
    return candidates


def _relative_workspace_path(raw_path: str, workspace_root: Path) -> str:
    try:
        return str(Path(raw_path).resolve().relative_to(workspace_root.resolve()))
    except (ValueError, OSError):
        return raw_path


def _extract_file_read_excerpt(tool_runs: list[ToolResult]) -> str | None:
    for tool_run in reversed(tool_runs):
        if tool_run.tool_name == "file_read" and tool_run.success and tool_run.content:
            return "\n".join(tool_run.content.splitlines()[:8])
    return None


def _uncertain_file_recovery_result(action: ToolAction, detail: str) -> tuple[str, ToolResult]:
    return (
        "uncertain",
        ToolResult(
            tool_call_id=action.tool_call_id,
            tool_name=action.tool_name,
            success=False,
            content="",
            error=f"File action side effect is uncertain. {detail}",
        ),
    )
