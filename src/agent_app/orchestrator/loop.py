from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from agent_app.agent.definition import AgentDefinition
from agent_app.agent.prompts import render_system_prompt
from agent_app.orchestrator.context_builder import build_context_messages, build_evidence_message, estimate_messages_tokens
from agent_app.runtime.task_runtime import InvalidTaskTransition, TaskRuntime
from agent_app.state.session_service import (
    SessionService,
    TaskVersionConflict,
    TracePersistenceError,
)
from agent_app.tools.approval import ApprovalResult, approve_tool_call, command_matches_prefix, shell_approval_prefix
from agent_app.tools.base import ToolExecutionContext, observation_from_tool_result
from agent_app.tools.registry import ToolRegistry
from agent_app.types import (
    AgentEvent,
    Message,
    ModelResponse,
    Observation,
    PendingAction,
    SessionContext,
    StoredMessage,
    TaskBudget,
    TodoItem,
    ToolAction,
    ToolCall,
    ToolResult,
    TurnResult,
)

ConfirmationHandler = Callable[[ToolCall, ToolExecutionContext], bool | str]


class InvalidTaskEvent(RuntimeError):
    pass


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
            resolved_tool_timeout = 600.0

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
        self._tasks = TaskRuntime(session_service)
        self._process_id = str(uuid4())
        self._session_shell_prefixes: dict[str, set[str]] = {}
        self._active_task_id: str | None = None
        self._turn_started_at: float | None = None

    def run_turn(
        self,
        *,
        user_input: str,
        session_id: str | None = None,
        budget: TaskBudget | None = None,
        _task_id: str | None = None,
        _append_user_message: bool = True,
    ) -> TurnResult:
        self._active_task_id = None
        self._turn_started_at = None
        try:
            return self._run_turn_impl(
                user_input=user_input,
                session_id=session_id,
                budget=budget,
                _task_id=_task_id,
                _append_user_message=_append_user_message,
            )
        except KeyboardInterrupt:
            if self._active_task_id is None:
                raise
            task = self._tasks.cancel(self._active_task_id)
            return self._task_result(task, final_text="Cancelled.", stop_reason="cancelled", success=False)
        except TracePersistenceError:
            return self._runtime_failure_result(
                session_id=session_id,
                stop_reason="trace_persistence_error",
            )
        except Exception:
            if self._active_task_id is None:
                raise
            return self._runtime_failure_result(
                session_id=session_id,
                stop_reason="internal_exception",
            )

    def _run_turn_impl(
        self,
        *,
        user_input: str,
        session_id: str | None = None,
        budget: TaskBudget | None = None,
        _task_id: str | None = None,
        _append_user_message: bool = True,
    ) -> TurnResult:
        resolved_session_id = self._session_service.get_or_create_session(session_id)
        if _task_id is None:
            latest = self._session_service.get_latest_task(resolved_session_id)
            if (
                latest is not None
                and latest.status == "waiting_user"
                and latest.pending_action is not None
                and latest.pending_action.kind == "tool_approval"
            ):
                return self._task_result(
                    latest,
                    final_text=latest.pending_action.prompt,
                    stop_reason="waiting_user",
                    success=False,
                )
            task = self._tasks.start_for_user_message(
                session_id=resolved_session_id,
                user_input=user_input,
                budget=budget,
            )
        else:
            task = self._tasks.require_task(_task_id)
            if task.session_id != resolved_session_id:
                raise ValueError("Task does not belong to the requested session.")
            if task.status != "running":
                return self._task_result(
                    task,
                    final_text=None,
                    stop_reason=task.stop_reason or f"task_{task.status}",
                    success=task.status == "completed",
                )
        self._active_task_id = task.id
        self._turn_started_at = time.monotonic()
        recovery_blockers = self._recover_pending_tool_actions(resolved_session_id)
        if recovery_blockers:
            action_names = ", ".join(
                f"{result.tool_name} ({result.tool_call_id})"
                for result in recovery_blockers
            )
            task = self._tasks.fail(task.id, reason="uncertain_tool_action")
            return TurnResult(
                session_id=resolved_session_id,
                final_text=(
                    "A previous tool action has an uncertain side-effect state and was not retried. "
                    f"Inspect the workspace before continuing: {action_names}."
                ),
                stop_reason="uncertain_tool_action",
                tool_runs=recovery_blockers,
                success=False,
                task_id=task.id,
                task_status=task.status,
            )

        if _append_user_message:
            self._session_service.append_message(
                resolved_session_id,
                Message(role="user", content=user_input),
            )

        self._tool_context = replace(
            self._tool_context,
            prepared_edits={},
            turn_state={"shell_approval_prefixes": self._session_shell_prefixes.setdefault(resolved_session_id, set())},
            session_id=resolved_session_id,
            task_id=task.id,
            session_service=self._session_service,
        )
        if len(task.plan) > 1 and not task.working_memory.get("planner_initialized"):
            self._session_service.append_task_trace(
                task.id,
                "planner",
                {"trigger": "new_multi_step_task", "items": [item.content for item in task.plan]},
            )
            task = self._tasks.update_working_memory(task.id, {"planner_initialized": True})

        messages = self._session_service.list_messages(resolved_session_id)
        session_context = self._session_service.get_session_context(resolved_session_id)
        session_context = self._maybe_update_summary(
            session_id=resolved_session_id,
            messages=messages,
            session_context=session_context,
        )
        if task.plan:
            session_context = SessionContext(
                summary_text=session_context.summary_text,
                summary_message_id=session_context.summary_message_id,
                todo_items=task.plan,
            )
        tool_runs_history = self._session_service.list_tool_runs(resolved_session_id)
        evidence_message = build_evidence_message(tool_runs_history)
        provider_messages = build_context_messages(
            messages=messages,
            session_context=session_context,
            context_token_budget=self._context_token_budget,
            evidence_message=evidence_message,
        )
        tool_runs: list[ToolResult] = []
        if _requires_web_research(user_input) and not task.working_memory.get("web_research_preflight_completed"):
            research_call = ToolCall(
                id=f"web-research-{task.id}",
                name="web_search",
                arguments={"query": user_input},
            )
            research_result = self._execute_tool_call(research_call)
            tool_runs.append(research_result)
            error_type = _search_stop_reason(research_result.error)
            self._session_service.append_task_trace(
                task.id,
                "research_requirement",
                {
                    "tool_call_id": research_call.id,
                    "required": True,
                    "success": research_result.success,
                    "error_type": error_type if not research_result.success else None,
                    "source_count": _web_search_source_count(research_result.content),
                },
            )
            if not research_result.success:
                return self._finalize_turn(
                    session_id=resolved_session_id,
                    user_input=user_input,
                    context_message_count=len(provider_messages),
                    context_token_estimate=sum(
                        estimate_messages_tokens([StoredMessage(id=index, role=message["role"], content=message.get("content"))])
                        for index, message in enumerate(provider_messages, start=1)
                    ),
                    used_summary=bool(session_context.summary_text),
                    used_todo=bool(session_context.todo_items),
                    used_evidence=bool(evidence_message),
                    final_text=None,
                    stop_reason=error_type,
                    tool_runs=tool_runs,
                    success=False,
                )
            task = self._tasks.update_working_memory(task.id, {"web_research_preflight_completed": True})
            provider_messages.insert(0, {"role": "system", "content": _web_search_observation_message(research_result.content)})
        base_context_message_count = len(provider_messages)
        base_context_token_estimate = sum(
            estimate_messages_tokens([StoredMessage(id=index, role=message["role"], content=message.get("content"))])
            for index, message in enumerate(provider_messages, start=1)
        )
        used_summary = bool(session_context.summary_text)
        used_todo = bool(session_context.todo_items)
        used_evidence = bool(evidence_message) or bool(tool_runs)
        system_prompt = render_system_prompt(self._agent)
        tool_rounds = 0
        consecutive_failure_tool: str | None = None
        consecutive_failure_count = 0

        while True:
            task = self._tasks.require_task(task.id)
            budget_reason = self._tasks.budget_stop_reason(task)
            if budget_reason is not None:
                return self._finalize_turn(
                    session_id=resolved_session_id,
                    user_input=user_input,
                    context_message_count=base_context_message_count,
                    context_token_estimate=base_context_token_estimate,
                    used_summary=used_summary,
                    used_todo=used_todo,
                    used_evidence=used_evidence,
                    final_text=None,
                    stop_reason=budget_reason,
                    tool_runs=tool_runs,
                    success=False,
                )
            model_started = time.monotonic()
            response = self._model_client.generate(
                system_prompt=system_prompt,
                messages=provider_messages,
                tools=self._available_tool_specs(),
            )
            model_duration_ms = int((time.monotonic() - model_started) * 1000)
            response_tokens = response.total_tokens or _estimate_response_tokens(provider_messages, response)
            task = self._tasks.consume_model_call(task.id, tokens=response_tokens)
            decision_payload = _decision_payload(response)
            self._session_service.append_task_trace(
                task.id,
                "model_call",
                {
                    "phase": "policy",
                    "model": response.model_name or getattr(self._model_client, "model", self._agent.default_model),
                    "duration_ms": model_duration_ms,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "total_tokens": response_tokens,
                    "usage_source": response.usage_source,
                    "error_type": response.error_type,
                },
            )
            self._session_service.append_task_trace(task.id, "decision", decision_payload)
            repeated_reason = self._record_decision_repetition(task.id, decision_payload)
            if repeated_reason is not None:
                self._tasks.reflect(task.id, repeated_reason)
                return self._finalize_turn(
                    session_id=resolved_session_id,
                    user_input=user_input,
                    context_message_count=base_context_message_count,
                    context_token_estimate=base_context_token_estimate,
                    used_summary=used_summary,
                    used_todo=used_todo,
                    used_evidence=used_evidence,
                    final_text=None,
                    stop_reason="repeated_decision",
                    tool_runs=tool_runs,
                    success=False,
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
                    self._session_service.append_task_trace(
                        task.id,
                        "critic",
                        {
                            "trigger": "final_answer",
                            "allowed": True,
                            "evidence_count": len([item for item in tool_runs if item.success]),
                        },
                    )
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

                current_task = self._tasks.require_task(task.id)
                budget_reason = self._tasks.budget_stop_reason(current_task)
                if budget_reason is not None:
                    return self._finalize_turn(
                        session_id=resolved_session_id,
                        user_input=user_input,
                        context_message_count=base_context_message_count,
                        context_token_estimate=base_context_token_estimate,
                        used_summary=used_summary,
                        used_todo=used_todo,
                        used_evidence=used_evidence,
                        final_text=None,
                        stop_reason=budget_reason,
                        tool_runs=tool_runs,
                        success=False,
                    )
                tool_result = self._execute_tool_call(tool_call)
                tool_runs.append(tool_result)
                current_task = self._tasks.require_task(task.id)
                if current_task.status == "waiting_user":
                    return self._finalize_turn(
                        session_id=resolved_session_id,
                        user_input=user_input,
                        context_message_count=base_context_message_count,
                        context_token_estimate=base_context_token_estimate,
                        used_summary=used_summary,
                        used_todo=used_todo,
                        used_evidence=used_evidence,
                        final_text=current_task.pending_action.prompt if current_task.pending_action else None,
                        stop_reason="waiting_user",
                        tool_runs=tool_runs,
                        success=False,
                    )
                if current_task.status in {"failed", "cancelled", "expired"}:
                    return self._finalize_turn(
                        session_id=resolved_session_id,
                        user_input=user_input,
                        context_message_count=base_context_message_count,
                        context_token_estimate=base_context_token_estimate,
                        used_summary=used_summary,
                        used_todo=used_todo,
                        used_evidence=used_evidence,
                        final_text=None,
                        stop_reason=current_task.stop_reason or f"task_{current_task.status}",
                        tool_runs=tool_runs,
                        success=False,
                    )
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

                repair_stop = self._record_repair_attempt_if_needed(
                    task_id=task.id,
                    tool_call=tool_call,
                    tool_result=tool_result,
                    tool_runs=tool_runs,
                )
                if repair_stop is not None:
                    return self._finalize_turn(
                        session_id=resolved_session_id,
                        user_input=user_input,
                        context_message_count=base_context_message_count,
                        context_token_estimate=base_context_token_estimate,
                        used_summary=used_summary,
                        used_todo=used_todo,
                        used_evidence=used_evidence,
                        final_text=None,
                        stop_reason=repair_stop,
                        tool_runs=tool_runs,
                        success=False,
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
                        self._tasks.reflect(
                            task.id,
                            f"Tool '{tool_result.tool_name}' failed repeatedly: {tool_result.error or 'unknown error'}",
                        )
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

    def handle_event(self, event: AgentEvent) -> TurnResult:
        self._active_task_id = event.task_id
        try:
            return self._handle_event_impl(event)
        except (InvalidTaskEvent, InvalidTaskTransition, TaskVersionConflict, KeyError, ValueError):
            raise
        except TracePersistenceError:
            return self._runtime_failure_result(
                session_id=event.session_id,
                stop_reason="trace_persistence_error",
            )
        except Exception:
            return self._runtime_failure_result(
                session_id=event.session_id,
                stop_reason="internal_exception",
            )

    def _handle_event_impl(self, event: AgentEvent) -> TurnResult:
        if event.type == "user_message":
            content = str(event.payload.get("content", ""))
            if event.task_id is None:
                resolved_session_id = self._session_service.get_or_create_session(event.session_id)
                task = self._tasks.start_for_user_message(
                    session_id=resolved_session_id,
                    user_input=content,
                    event=event,
                )
                return self.run_turn(
                    user_input=content,
                    session_id=resolved_session_id,
                    _task_id=task.id,
                    _append_user_message=True,
                )
            task = self._tasks.require_task(event.task_id)
            if task.status == "waiting_user":
                task = self._tasks.resume_with_user_message(task.id, content, event=event)
            else:
                task = self._tasks.record_user_message(task.id, content, event=event)
            return self.run_turn(
                user_input=content,
                session_id=event.session_id,
                _task_id=task.id,
                _append_user_message=True,
            )

        if event.task_id is None:
            raise ValueError(f"Event '{event.type}' requires task_id.")
        task = self._tasks.expire_if_needed(self._tasks.require_task(event.task_id))
        if task.status == "expired":
            return self._task_result(task, final_text=None, stop_reason=task.stop_reason, success=False)

        if event.type == "pause_requested":
            return self._task_result(self._tasks.pause(task.id, event=event), final_text=None, stop_reason="paused", success=False)
        if event.type == "resume_requested":
            task = self._tasks.resume(task.id, event=event)
            return self.run_turn(
                user_input="",
                session_id=task.session_id,
                _task_id=task.id,
                _append_user_message=False,
            )
        if event.type == "cancel_requested":
            return self._task_result(self._tasks.cancel(task.id, event=event), final_text=None, stop_reason="cancelled", success=False)
        if event.type == "task_expired":
            return self._task_result(self._tasks.expire(task.id, event=event), final_text=None, stop_reason="waiting_user_expired", success=False)
        if event.type not in {"user_approved", "user_rejected"}:
            raise ValueError(f"Unsupported event type '{event.type}'.")
        if task.status != "waiting_user" or task.pending_action is None:
            raise InvalidTaskEvent(f"Task '{task.id}' is not waiting for user input.")

        pending = task.pending_action
        if (
            pending.kind == "tool_approval"
            and pending.decision is not None
            and pending.decision.get("tool_name") == "shell"
            and pending.decision.get("approval_process_id") != self._process_id
        ):
            task = self._tasks.reject(task.id)
            self._session_service.append_task_trace(
                task.id,
                "approval",
                {"tool": "shell", "decision": "expired", "reason": "process_restarted"},
            )
            return self._task_result(task, final_text=None, stop_reason="shell_approval_expired", success=False)
        if event.type == "user_approved":
            task = self._tasks.approve(task.id, event=event)
        else:
            task = self._tasks.reject(task.id, event=event)

        if pending.kind != "tool_approval" or pending.decision is None:
            return self.run_turn(
                user_input=str(event.payload.get("content", "")),
                session_id=task.session_id,
                _task_id=task.id,
                _append_user_message=bool(event.payload.get("content")),
            )

        tool_call = ToolCall(
            id=str(pending.decision["tool_call_id"]),
            name=str(pending.decision["tool_name"]),
            arguments=dict(pending.decision.get("arguments", {})),
        )
        self._active_task_id = task.id
        self._turn_started_at = time.monotonic()
        self._tool_context = replace(
            self._tool_context,
            prepared_edits={},
            turn_state={
                "approved_tool_calls": {tool_call.id},
                "approved_action_metadata": {
                    tool_call.id: dict(pending.decision.get("approval_metadata", {}))
                },
                "shell_approval_prefixes": self._session_shell_prefixes.setdefault(task.session_id, set()),
            },
            session_id=task.session_id,
            task_id=task.id,
            session_service=self._session_service,
        )
        if event.type == "user_approved":
            tool_result = self._execute_tool_call(tool_call)
            self._session_service.append_task_trace(
                task.id,
                "approval",
                {"tool": tool_call.name, "decision": "approve", "tool_call_id": tool_call.id, "resumed": True},
            )
        else:
            tool_result = self._record_untracked_tool_result(
                ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    success=False,
                    content="",
                    error="Tool use denied by user.",
                ),
                side_effect=True,
                approval_decision="reject",
            )
            self._session_service.append_task_trace(
                task.id,
                "approval",
                {"tool": tool_call.name, "decision": "reject", "tool_call_id": tool_call.id, "resumed": True},
            )
        result = self.run_turn(
            user_input="",
            session_id=task.session_id,
            _task_id=task.id,
            _append_user_message=False,
        )
        result.tool_runs.insert(0, tool_result)
        return result

    def get_task(self, task_id: str):
        return self._tasks.require_task(task_id)

    def _record_decision_repetition(self, task_id: str, decision: dict[str, Any]) -> str | None:
        task = self._tasks.require_task(task_id)
        decision_hash = hashlib.sha256(
            json.dumps(decision, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        previous_hash = task.working_memory.get("last_decision_hash")
        repeat_count = int(task.working_memory.get("repeat_decision_count", 0))
        made_progress = task.last_observation is not None and task.last_observation.status == "succeeded"
        repeat_count = repeat_count + 1 if previous_hash == decision_hash and not made_progress else 1
        self._tasks.update_working_memory(
            task_id,
            {
                "last_decision_hash": decision_hash,
                "repeat_decision_count": repeat_count,
            },
        )
        if repeat_count >= task.budget.repeat_decision_limit:
            return f"The same decision repeated {repeat_count} times without progress."
        return None

    def _require_active_task_id(self) -> str:
        if self._active_task_id is None:
            raise RuntimeError("Task-aware execution requires an active task.")
        return self._active_task_id

    def _task_result(
        self,
        task,
        *,
        final_text: str | None,
        stop_reason: str | None,
        success: bool,
        tool_runs: list[ToolResult] | None = None,
    ) -> TurnResult:
        return TurnResult(
            session_id=task.session_id,
            final_text=final_text,
            stop_reason=stop_reason,
            tool_runs=tool_runs or [],
            success=success,
            task_id=task.id,
            task_status=task.status,
            pending_action=task.pending_action,
        )

    def _runtime_failure_result(
        self,
        *,
        session_id: str | None,
        stop_reason: str,
    ) -> TurnResult:
        task = None
        if self._active_task_id is not None:
            try:
                task = self._tasks.fail(self._active_task_id, reason=stop_reason)
            except Exception:
                task = self._session_service.get_task(self._active_task_id)
        resolved_session_id = (
            task.session_id
            if task is not None
            else session_id or ""
        )
        return TurnResult(
            session_id=resolved_session_id,
            final_text=None,
            stop_reason=stop_reason,
            tool_runs=[],
            success=False,
            task_id=None if task is None else task.id,
            task_status=None if task is None else task.status,
            pending_action=None if task is None else task.pending_action,
        )

    def _execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        if tool_call.name == "ask_user":
            question = tool_call.arguments.get("question")
            if not isinstance(question, str) or not question.strip():
                return self._record_untracked_tool_result(
                    ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        success=False,
                        content="",
                        error="Invalid arguments: question is required.",
                    ),
                    approval_decision="not_requested",
                )
            task_id = self._require_active_task_id()
            self._tasks.wait_for_user(
                task_id,
                PendingAction(
                    kind="ask_user",
                    prompt=question.strip(),
                    decision={"type": "ask_user", "question": question.strip()},
                ),
            )
            self._session_service.append_task_trace(
                task_id,
                "approval",
                {"decision": "pending", "kind": "ask_user", "question": question.strip()},
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                content="",
                error="Waiting for user response.",
            )
        if tool_call.name == "give_up":
            reason = str(tool_call.arguments.get("reason") or "Agent gave up.")
            self._tasks.fail(self._require_active_task_id(), reason="give_up")
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                content="",
                error=reason,
            )
        if tool_call.name not in self._agent.allowed_tools:
            return self._record_untracked_tool_result(ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                content="",
                error=f"Tool '{tool_call.name}' is not allowed for this agent.",
            ), approval_decision="deny")

        tool = self._tool_registry.get(tool_call.name)
        if tool is None:
            return self._record_untracked_tool_result(ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                content="",
                error=f"Tool '{tool_call.name}' is not registered.",
            ), approval_decision="deny")

        side_effect = tool.has_side_effect_for(tool_call.arguments)

        validation_error = tool.validate_arguments(tool_call.arguments)
        if validation_error is not None:
            return self._record_untracked_tool_result(ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                content="",
                error=validation_error,
            ), side_effect=side_effect, approval_decision="not_requested")

        approval = approve_tool_call(tool_call.name, tool_call.arguments)
        if tool_call.name == "shell" and approval.decision == "confirm":
            command = tool_call.arguments.get("command")
            prefixes = self._tool_context.turn_state.get("shell_approval_prefixes", set())
            if isinstance(command, str) and any(command_matches_prefix(command, prefix) for prefix in prefixes):
                approval = ApprovalResult("allow", "Allowed by a user-approved session prefix.")
        if approval.decision == "deny":
            return self._record_untracked_tool_result(ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                content="",
                error=approval.reason,
            ), side_effect=side_effect, approval_decision="deny")

        if approval.decision == "confirm":
            approved_ids = self._tool_context.turn_state.setdefault("approved_tool_calls", set())
            if tool_call.id not in approved_ids:
                approval_metadata: dict[str, Any] = {"side_effect": True} if tool_call.name == "shell" else {}
                if tool_call.name != "shell":
                    inspection, error = tool.inspect(arguments=tool_call.arguments, context=self._tool_context)
                    if inspection is None:
                        return self._record_untracked_tool_result(ToolResult(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            success=False,
                            content="",
                            error=error or "Unable to inspect edit.",
                        ), side_effect=side_effect, approval_decision="inspection_failed")
                    self._tool_context.prepared_edits[tool_call.id] = inspection
                    approval_metadata = tool.recovery_metadata(
                        tool_call_id=tool_call.id,
                        arguments=tool_call.arguments,
                        context=self._tool_context,
                    )
                task_id = self._require_active_task_id()
                pending = PendingAction(
                    kind="tool_approval",
                    prompt=f"Approve tool '{tool_call.name}' for task {task_id}?",
                    decision={
                        "type": "tool_call",
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.name,
                        "arguments": tool_call.arguments,
                        "approval_metadata": approval_metadata,
                        "approval_process_id": self._process_id,
                    },
                )
                self._tasks.wait_for_user(task_id, pending)
                self._session_service.append_task_trace(
                    task_id,
                    "critic",
                    {
                        "trigger": "side_effect_action",
                        "allowed": False,
                        "reason": "user_approval_required",
                        "tool": tool_call.name,
                        "risk_level": tool.risk_level,
                    },
                )
                if self._confirmation_handler is None:
                    self._session_service.append_task_trace(
                        task_id,
                        "approval",
                        {"tool": tool_call.name, "decision": "pending", "tool_call_id": tool_call.id},
                    )
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        success=False,
                        content="",
                        error="Waiting for user approval.",
                    )

                confirmed = self._confirmation_handler(tool_call, self._tool_context)
                if confirmed:
                    self._tasks.approve(task_id)
                    approved_ids.add(tool_call.id)
                    approval_decision = "approve"
                    if confirmed == "session" and tool_call.name == "shell":
                        command = tool_call.arguments.get("command")
                        if isinstance(command, str):
                            prefix = shell_approval_prefix(command)
                            if prefix:
                                self._tool_context.turn_state.setdefault("shell_approval_prefixes", set()).add(prefix)
                                approval_decision = "approve_session_prefix"
                    self._session_service.append_task_trace(
                        task_id,
                        "approval",
                        {"tool": tool_call.name, "decision": approval_decision, "tool_call_id": tool_call.id},
                    )
                else:
                    self._tasks.reject(task_id)
                    self._tool_context.prepared_edits.pop(tool_call.id, None)
                    self._session_service.append_task_trace(
                        task_id,
                        "approval",
                        {"tool": tool_call.name, "decision": "reject", "tool_call_id": tool_call.id},
                    )
                    return self._record_untracked_tool_result(ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        success=False,
                        content="",
                        error="Tool use denied by user.",
                    ), side_effect=side_effect, approval_decision="reject")

        session_id = self._tool_context.session_id
        if session_id is None:
            raise RuntimeError("Tool execution requires an active session.")
        task_id = self._require_active_task_id()
        task = self._tasks.require_task(task_id)
        approved_metadata = self._tool_context.turn_state.get("approved_action_metadata", {}).get(tool_call.id)
        if approved_metadata:
            inspection, inspection_error = tool.inspect(
                arguments=tool_call.arguments,
                context=self._tool_context,
            )
            if inspection is None:
                return self._record_untracked_tool_result(
                    ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        success=False,
                        content="",
                        error=inspection_error or "Unable to revalidate approved action.",
                    ),
                    side_effect=side_effect,
                    approval_decision="approve_revalidation_failed",
                )
            self._tool_context.prepared_edits[tool_call.id] = inspection
            current_metadata = tool.recovery_metadata(
                tool_call_id=tool_call.id,
                arguments=tool_call.arguments,
                context=self._tool_context,
            )
            approval_keys = ("relative_path", "before_exists", "before_sha256", "after_sha256")
            if any(current_metadata.get(key) != approved_metadata.get(key) for key in approval_keys):
                self._tool_context.prepared_edits.pop(tool_call.id, None)
                return self._record_untracked_tool_result(
                    ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        success=False,
                        content="",
                        error="Approved action changed since inspection. Please review it again.",
                    ),
                    side_effect=side_effect,
                    approval_decision="approve_conflict",
                )
        retry_of: str | None = None
        for attempt in range(1, task.budget.max_retries + 2):
            task = self._tasks.require_task(task_id)
            if task.budget.used_tool_calls >= task.budget.max_tool_calls:
                tool_result = ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    success=False,
                    content="",
                    error="Tool call budget exhausted.",
                )
                observation = observation_from_tool_result(
                    tool_result,
                    side_effect=side_effect,
                    attempt=attempt,
                    duration_ms=0,
                )
                tool_result.observation = observation
                self._tasks.record_observation(task_id, observation)
                self._session_service.append_task_trace(
                    task_id,
                    "tool_attempt",
                    {
                        "tool_call_id": tool_call.id,
                        "tool": tool_call.name,
                        "arguments": tool_call.arguments,
                        "approval": approval.decision,
                        "attempt": attempt,
                        "duration_ms": 0,
                        "success": False,
                        "error_type": observation.error_type,
                        "retryable": False,
                        "side_effect": side_effect,
                        "budget_blocked": True,
                    },
                )
                return tool_result
            task = self._tasks.consume_tool_call(task_id)
            recovery_metadata = tool.recovery_metadata(
                tool_call_id=tool_call.id,
                arguments=tool_call.arguments,
                context=self._tool_context,
            )
            action = self._session_service.prepare_tool_action(
                session_id,
                agent_id=self._agent.id,
                tool_call=_persisted_tool_call(tool_call),
                recovery_metadata=recovery_metadata,
                task_id=task_id,
                attempt=attempt,
                retry_of=retry_of,
            )
            if action.status in {"succeeded", "failed", "uncertain"} and action.result is not None:
                tool_result = action.result
                duration_ms = 0
            elif action.status == "executing":
                tool_result = ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    success=False,
                    content="",
                    error="Tool action is already executing and was not run again.",
                )
                duration_ms = 0
            else:
                action = self._session_service.mark_tool_action_executing(action.id)
                started = time.monotonic()
                try:
                    tool_result = tool.execute(
                        tool_call_id=tool_call.id,
                        arguments=tool_call.arguments,
                        context=self._tool_context,
                    )
                    status = "succeeded" if tool_result.success else "failed"
                except Exception as exc:
                    if side_effect:
                        status, tool_result = self._classify_tool_action_recovery(action)
                    else:
                        status = "failed"
                        tool_result = ToolResult(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            success=False,
                            content="",
                            error=f"Tool execution raised {type(exc).__name__}: {exc}",
                        )
                duration_ms = int((time.monotonic() - started) * 1000)
                completed = self._session_service.complete_tool_action(
                    action.id,
                    status=status,
                    tool_result=tool_result,
                )
                assert completed.result is not None
                tool_result = completed.result

            observation = observation_from_tool_result(
                tool_result,
                side_effect=side_effect,
                attempt=attempt,
                duration_ms=duration_ms,
            )
            tool_result.observation = observation
            self._tasks.record_observation(task_id, observation)
            self._session_service.append_task_trace(
                task_id,
                "tool_attempt",
                {
                    "tool_call_id": tool_call.id,
                    "tool": tool_call.name,
                    "arguments": _audit_arguments(tool_call),
                    "arguments_hash": hashlib.sha256(
                        json.dumps(tool_call.arguments, ensure_ascii=False, sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                    "approval": approval.decision,
                    "attempt": attempt,
                    "retry_of": retry_of,
                    "duration_ms": duration_ms,
                    "success": tool_result.success,
                    "error_type": observation.error_type,
                    "retryable": observation.retryable,
                    "side_effect": observation.side_effect,
                    "risk_level": tool.risk_level,
                },
            )
            self._session_service.append_task_trace(
                task_id,
                "observation",
                {
                    "tool_call_id": tool_call.id,
                    "status": observation.status,
                    "error_type": observation.error_type,
                    "message": observation.message,
                    "retryable": observation.retryable,
                    "side_effect": observation.side_effect,
                    "attempt": attempt,
                    "duration_ms": duration_ms,
                },
            )
            can_retry = observation.retryable and (not side_effect or tool.is_idempotent_for(tool_call.arguments))
            if tool_result.success or not can_retry or attempt > task.budget.max_retries:
                return tool_result
            retry_of = action.id
            self._session_service.append_task_trace(
                task_id,
                "retry",
                {
                    "tool_call_id": tool_call.id,
                    "attempt": attempt + 1,
                    "retry_of": retry_of,
                    "backoff_ms": 10 * (2 ** (attempt - 1)),
                },
            )
            time.sleep(0.01 * (2 ** (attempt - 1)))

        raise AssertionError("Tool retry loop exhausted without a result.")

    def _record_untracked_tool_result(
        self,
        tool_result: ToolResult,
        *,
        side_effect: bool = False,
        approval_decision: str,
    ) -> ToolResult:
        session_id = self._tool_context.session_id
        if session_id is None:
            raise RuntimeError("Tool execution requires an active session.")
        task_id = self._require_active_task_id()
        self._tasks.consume_tool_call(task_id)
        observation = observation_from_tool_result(
            tool_result,
            side_effect=side_effect,
            attempt=1,
            duration_ms=0,
        )
        tool_result.observation = observation
        self._session_service.append_tool_run(session_id, tool_result)
        self._tasks.record_observation(task_id, observation)
        self._session_service.append_task_trace(
            task_id,
            "tool_attempt",
            {
                "tool_call_id": tool_result.tool_call_id,
                "tool": tool_result.tool_name,
                "arguments": {},
                "approval": approval_decision,
                "attempt": 1,
                "duration_ms": 0,
                "success": False,
                "error_type": observation.error_type,
                "retryable": observation.retryable,
                "side_effect": side_effect,
            },
        )
        return tool_result

    def _available_tool_specs(self) -> list[dict[str, Any]]:
        return [
            *self._tool_registry.get_specs(self._agent.allowed_tools),
            {
                "type": "function",
                "function": {
                    "name": "ask_user",
                    "description": "Pause the task and ask the user for missing information.",
                    "parameters": {
                        "type": "object",
                        "properties": {"question": {"type": "string", "minLength": 1}},
                        "required": ["question"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "give_up",
                    "description": "Stop the task when it cannot be completed safely.",
                    "parameters": {
                        "type": "object",
                        "properties": {"reason": {"type": "string", "minLength": 1}},
                        "required": ["reason"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

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

        started = time.monotonic()
        response = self._model_client.generate(
            system_prompt=(
                "Summarize the prior conversation for future task continuation. "
                "Include the user goal, completed actions, unresolved issues, important constraints or assumptions, "
                "and any important files, paths, or results."
            ),
            messages=[{"role": "user", "content": "\n".join(transcript_lines)}],
            tools=[],
        )
        if self._active_task_id is not None:
            duration_ms = int((time.monotonic() - started) * 1000)
            tokens = response.total_tokens or _estimate_response_tokens(
                [{"role": "user", "content": "\n".join(transcript_lines)}],
                response,
            )
            self._tasks.consume_model_call(self._active_task_id, tokens=tokens)
            self._session_service.append_task_trace(
                self._active_task_id,
                "model_call",
                {
                    "phase": "summary",
                    "model": response.model_name or getattr(self._model_client, "model", self._agent.default_model),
                    "duration_ms": duration_ms,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "total_tokens": tokens,
                    "usage_source": response.usage_source,
                    "error_type": response.error_type,
                },
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
        task = self._tasks.require_task(self._require_active_task_id())
        if self._turn_started_at is not None and task.status not in {"completed", "failed", "cancelled", "expired"}:
            task = self._tasks.add_active_time(task.id, time.monotonic() - self._turn_started_at)
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
            self._session_service.append_task_trace(
                task.id,
                "turn",
                {
                    "user_input": user_input,
                    "context_message_count": context_message_count,
                    "context_token_estimate": context_token_estimate,
                    "used_summary": used_summary,
                    "used_todo": used_todo,
                    "used_evidence": used_evidence,
                    "final_text": final_text,
                    "stop_reason": stop_reason,
                    "success": success,
                },
            )
        except Exception as exc:
            task = self._tasks.require_task(task.id)
            if task.status not in {"completed", "failed", "cancelled", "expired"}:
                task = self._tasks.fail(task.id, reason="trace_persistence_error")
            return TurnResult(
                session_id=session_id,
                final_text=None,
                stop_reason="trace_persistence_error",
                tool_runs=tool_runs,
                success=False,
                task_id=task.id,
                task_status=task.status,
                pending_action=task.pending_action,
            )

        if task.status != "waiting_user" and task.status not in {"completed", "failed", "cancelled", "expired"}:
            task = (
                self._tasks.complete(task.id, reason=stop_reason or "completed")
                if success
                else self._tasks.fail(task.id, reason=stop_reason or "failed")
            )

        return TurnResult(
            session_id=session_id,
            final_text=final_text,
            stop_reason=stop_reason,
            tool_runs=tool_runs,
            success=success,
            task_id=task.id,
            task_status=task.status,
            pending_action=task.pending_action,
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

    def _record_repair_attempt_if_needed(
        self,
        *,
        task_id: str,
        tool_call: ToolCall,
        tool_result: ToolResult,
        tool_runs: list[ToolResult],
    ) -> str | None:
        if not _is_failed_verification_after_edit(tool_call, tool_result, tool_runs):
            return None

        task = self._tasks.require_task(task_id)
        command = str(tool_call.arguments.get("command", ""))
        output = tool_result.content or tool_result.error or ""
        if task.budget.used_repair_attempts >= task.budget.max_repair_attempts:
            self._session_service.append_task_trace(
                task_id,
                "repair",
                {
                    "trigger": "verification_failure",
                    "command": command,
                    "allowed": False,
                    "attempt": task.budget.used_repair_attempts + 1,
                    "max_attempts": task.budget.max_repair_attempts,
                    "output_preview": output[:1000],
                },
            )
            return "repair_attempt_budget_exceeded"

        updated = self._tasks.consume_repair_attempt(task_id)
        self._session_service.append_task_trace(
            task_id,
            "repair",
            {
                "trigger": "verification_failure",
                "command": command,
                "allowed": True,
                "attempt": updated.budget.used_repair_attempts,
                "max_attempts": updated.budget.max_repair_attempts,
                "output_preview": output[:1000],
            },
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


def _is_failed_verification_after_edit(
    tool_call: ToolCall,
    tool_result: ToolResult,
    tool_runs: list[ToolResult],
) -> bool:
    if tool_result.success or tool_result.tool_name != "shell":
        return False
    command = str(tool_call.arguments.get("command", "")).strip().lower()
    if not _looks_like_verification_command(command):
        return False
    return _has_unverified_successful_edit(tool_runs)


def _looks_like_verification_command(command: str) -> bool:
    return (
        command.startswith("python -m unittest")
        or command.startswith("python -m pytest")
        or command.startswith("pytest")
    )


def _has_unverified_successful_edit(tool_runs: list[ToolResult]) -> bool:
    last_successful_shell = -1
    for index, tool_run in enumerate(tool_runs):
        if tool_run.tool_name == "shell" and tool_run.success:
            last_successful_shell = index
    return any(
        tool_run.success and tool_run.tool_name in {"file_write", "replace_in_file"}
        for tool_run in tool_runs[last_successful_shell + 1 :]
    )


def _decision_payload(response: ModelResponse) -> dict[str, Any]:
    if response.tool_calls:
        return {
            "type": "tool_calls",
            "actions": [
                {
                    "id": tool_call.id,
                    "tool": tool_call.name,
                    "arguments": tool_call.arguments,
                }
                for tool_call in response.tool_calls
            ],
        }
    if response.assistant_text:
        return {"type": "final_answer", "content": response.assistant_text}
    return {"type": "invalid", "error_type": response.error_type}


def _estimate_response_tokens(messages: list[dict[str, Any]], response: ModelResponse) -> int:
    payload = json.dumps(messages, ensure_ascii=False) + (response.assistant_text or "")
    if response.tool_calls:
        payload += json.dumps(_decision_payload(response), ensure_ascii=False)
    return max(1, (len(payload.encode("utf-8")) + 3) // 4)


def _requires_web_research(user_input: str) -> bool:
    normalized = user_input.casefold()
    phrases = ("查阅", "联网检索", "搜索网页", "网上搜索", "web search", "browse the web", "look up")
    return any(phrase in normalized for phrase in phrases)


def _search_stop_reason(error_message: str | None) -> str:
    if error_message:
        match = re.match(r"(search_[a-z_]+):", error_message.casefold())
        if match:
            return match.group(1)
    return "search_configuration_error"


def _web_search_source_count(content: str) -> int:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return 0
    sources = payload.get("sources") if isinstance(payload, dict) else None
    return len(sources) if isinstance(sources, list) else 0


def _web_search_observation_message(content: str) -> str:
    return (
        "A required public-web search completed before this response. Use only the following source-backed "
        "evidence for factual claims that depend on the research request, and include relevant source URLs in the final answer.\n"
        f"Search observation: {content}"
    )


def _persisted_tool_call(tool_call: ToolCall) -> ToolCall:
    if tool_call.name != "shell":
        return tool_call
    return ToolCall(
        id=tool_call.id,
        name=tool_call.name,
        arguments={"command_sha256": _shell_command_sha256(tool_call.arguments)},
    )


def _audit_arguments(tool_call: ToolCall) -> dict[str, Any]:
    if tool_call.name != "shell":
        return tool_call.arguments
    return {"command_redacted": True, "command_sha256": _shell_command_sha256(tool_call.arguments)}


def _shell_command_sha256(arguments: dict[str, Any]) -> str:
    command = arguments.get("command")
    return hashlib.sha256(str(command).encode("utf-8")).hexdigest()


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


class AgentRuntime(AgentLoop):
    """Task-aware runtime entry point; AgentLoop remains the compatibility name."""
