from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from agent_app.agent.definition import AgentDefinition, WORKER_AGENT
from agent_app.orchestrator.reviewer import (
    ReviewResult,
    WorkerReviewEvidence,
    WorkerReviewer,
    tool_results_to_evidence,
)
from agent_app.runtime.task_runtime import TaskRuntime
from agent_app.state.session_service import SessionService
from agent_app.tools.base import ToolExecutionContext
from agent_app.types import ToolResult

if TYPE_CHECKING:
    from agent_app.orchestrator.loop import ConfirmationHandler
    from agent_app.tools.registry import ToolRegistry

LoopFactory = Callable[..., object]

_MAX_RELEVANT_PATHS = 5


@dataclass(frozen=True, slots=True)
class DelegatedTaskRequest:
    task: str
    success_criteria: str
    relevant_paths: tuple[str, ...] = ()


class SubagentRunner:
    def __init__(
        self,
        *,
        model_client: object,
        session_service: SessionService,
        workspace_root: Path,
        tool_timeout: float,
        context_token_budget: int,
        summary_trigger_tokens: int,
        confirmation_handler: "ConfirmationHandler | None" = None,
        worker_agent: AgentDefinition = WORKER_AGENT,
        worker_registry: "ToolRegistry | None" = None,
        max_delegation_depth: int = 1,
        max_subagents_per_turn: int = 2,
        loop_factory: LoopFactory | None = None,
        skill_registry=None,
        reviewer: WorkerReviewer | None = None,
        max_repair_attempts: int = 2,
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts cannot be negative.")
        self._model_client = model_client
        self._session_service = session_service
        self._workspace_root = workspace_root
        self._tool_timeout = tool_timeout
        self._context_token_budget = context_token_budget
        self._summary_trigger_tokens = summary_trigger_tokens
        self._confirmation_handler = confirmation_handler
        self._worker_agent = worker_agent
        self._skill_registry = skill_registry
        if worker_registry is None:
            from agent_app.tools.registry import build_worker_registry

            worker_registry = build_worker_registry(skill_registry=skill_registry)
        self._worker_registry = worker_registry
        self._max_delegation_depth = max_delegation_depth
        self._max_subagents_per_turn = max_subagents_per_turn
        self._loop_factory = loop_factory
        self._reviewer = reviewer
        self._max_repair_attempts = max_repair_attempts

    def run(
        self,
        *,
        tool_call_id: str,
        request: DelegatedTaskRequest,
        context: ToolExecutionContext,
    ) -> ToolResult:
        session_service, session_error = _require_session_service(context)
        if session_service is None:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name="delegate_task",
                success=False,
                content="",
                error=session_error,
            )

        if context.delegation_depth >= self._max_delegation_depth:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name="delegate_task",
                success=False,
                content="",
                error=f"Delegation depth limit reached ({self._max_delegation_depth}).",
            )

        current_calls = int(context.turn_state.get("subagent_calls", 0))
        if current_calls >= self._max_subagents_per_turn:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name="delegate_task",
                success=False,
                content="",
                error=f"Subagent limit reached for this turn ({self._max_subagents_per_turn}).",
            )
        context.turn_state["subagent_calls"] = current_calls + 1

        child_result = self._run_worker(
            request=request,
            session_service=session_service,
            delegation_depth=context.delegation_depth + 1,
        )
        summary = _format_subagent_summary(
            child_session_id=child_result.session_id,
            agent_id=self._worker_agent.id,
            success=child_result.success,
            tool_runs=child_result.tool_runs,
            final_text=child_result.final_text,
            stop_reason=child_result.stop_reason,
        )
        self._record_subagent_run(
            session_service=session_service,
            parent_session_id=context.session_id or "",
            parent_tool_call_id=tool_call_id,
            request=request,
            child_result=child_result,
            summary=summary,
        )

        if self._reviewer is None:
            return _worker_tool_result(
                tool_call_id=tool_call_id,
                success=child_result.success,
                summary=summary,
                stop_reason=child_result.stop_reason,
            )

        review = self._review_worker(request=request, child_result=child_result)
        self._record_review_trace(
            context=context,
            tool_call_id=tool_call_id,
            child_session_id=child_result.session_id,
            review=review,
            attempt=0,
        )
        if review.decision == "accepted" and child_result.success:
            return _reviewed_tool_result(tool_call_id=tool_call_id, summary=summary, review=review)

        for attempt in range(1, self._max_repair_attempts + 1):
            if not self._consume_repair_attempt(context):
                break
            repair_request = _repair_request(request, review)
            repair_result = self._run_worker(
                request=repair_request,
                session_service=session_service,
                delegation_depth=context.delegation_depth + 1,
            )
            repair_summary = _format_subagent_summary(
                child_session_id=repair_result.session_id,
                agent_id=self._worker_agent.id,
                success=repair_result.success,
                tool_runs=repair_result.tool_runs,
                final_text=repair_result.final_text,
                stop_reason=repair_result.stop_reason,
            )
            self._record_subagent_run(
                session_service=session_service,
                parent_session_id=context.session_id or "",
                parent_tool_call_id=f"{tool_call_id}:repair:{attempt}",
                request=repair_request,
                child_result=repair_result,
                summary=repair_summary,
            )
            review = self._review_worker(request=repair_request, child_result=repair_result)
            self._record_review_trace(
                context=context,
                tool_call_id=tool_call_id,
                child_session_id=repair_result.session_id,
                review=review,
                attempt=attempt,
            )
            if review.decision == "accepted" and repair_result.success:
                return _reviewed_tool_result(
                    tool_call_id=tool_call_id,
                    summary=repair_summary,
                    review=review,
                )

        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="delegate_task",
            success=False,
            content=(
                f"{summary}\nreview_decision={review.decision}\n"
                f"review_feedback={_format_feedback(review)}"
            ),
            error=(
                "Worker review did not accept the result. "
                f"Repair attempts exhausted or blocked (max={self._max_repair_attempts})."
            ),
        )

    def _run_worker(self, *, request: DelegatedTaskRequest, session_service: SessionService, delegation_depth: int):
        child_session_id = session_service.create_session()
        child_loop = self._build_child_loop(delegation_depth=delegation_depth)
        return child_loop.run_turn(
            user_input=_build_child_user_input(request),
            session_id=child_session_id,
        )

    def _record_subagent_run(
        self,
        *,
        session_service: SessionService,
        parent_session_id: str,
        parent_tool_call_id: str,
        request: DelegatedTaskRequest,
        child_result,
        summary: str,
    ) -> None:
        session_service.append_subagent_run(
            parent_session_id=parent_session_id,
            parent_tool_call_id=parent_tool_call_id,
            child_session_id=child_result.session_id,
            agent_id=self._worker_agent.id,
            task=request.task,
            success=child_result.success,
            result_summary=summary,
        )

    def _review_worker(self, *, request: DelegatedTaskRequest, child_result) -> ReviewResult:
        assert self._reviewer is not None
        return self._reviewer.review(
            WorkerReviewEvidence(
                task=request.task,
                success_criteria=request.success_criteria,
                child_session_id=child_result.session_id,
                worker_success=child_result.success,
                final_text=child_result.final_text,
                stop_reason=child_result.stop_reason,
                tool_runs=tool_results_to_evidence(child_result.tool_runs),
            )
        )

    @staticmethod
    def _record_review_trace(
        *,
        context: ToolExecutionContext,
        tool_call_id: str,
        child_session_id: str,
        review: ReviewResult,
        attempt: int,
    ) -> None:
        if context.task_id is None or context.session_service is None:
            return
        context.session_service.append_task_trace(
            context.task_id,
            "worker_review",
            {
                "tool_call_id": tool_call_id,
                "child_session_id": child_session_id,
                "attempt": attempt,
                "decision": review.decision,
                "feedback": list(review.feedback),
                "evidence_refs": list(review.evidence_refs),
                "summary": review.summary,
            },
        )

    def _consume_repair_attempt(self, context: ToolExecutionContext) -> bool:
        if context.task_id is None or context.session_service is None:
            return True
        task = context.session_service.get_task(context.task_id)
        if task is None or task.budget.used_repair_attempts >= task.budget.max_repair_attempts:
            return False
        TaskRuntime(context.session_service).consume_repair_attempt(context.task_id)
        context.session_service.append_task_trace(
            context.task_id,
            "worker_repair",
            {
                "allowed": True,
                "attempt": task.budget.used_repair_attempts + 1,
                "max_attempts": task.budget.max_repair_attempts,
            },
        )
        return True

    def _build_child_loop(self, *, delegation_depth: int):
        if self._loop_factory is not None:
            return self._loop_factory(
                agent=self._worker_agent,
                model_client=self._model_client,
                tool_registry=self._worker_registry,
                session_service=self._session_service,
                workspace_root=self._workspace_root,
                tool_timeout=self._tool_timeout,
                context_token_budget=self._context_token_budget,
                summary_trigger_tokens=self._summary_trigger_tokens,
                confirmation_handler=self._confirmation_handler,
                delegation_depth=delegation_depth,
                skill_registry=self._skill_registry,
            )

        from agent_app.orchestrator.loop import AgentLoop

        return AgentLoop(
            agent=self._worker_agent,
            model_client=self._model_client,
            tool_registry=self._worker_registry,
            session_service=self._session_service,
            workspace_root=self._workspace_root,
            tool_timeout=self._tool_timeout,
            context_token_budget=self._context_token_budget,
            summary_trigger_tokens=self._summary_trigger_tokens,
            confirmation_handler=self._confirmation_handler,
            delegation_depth=delegation_depth,
            skill_registry=self._skill_registry,
        )


def _require_session_service(context: ToolExecutionContext) -> tuple[SessionService | None, str | None]:
    if context.session_id is None or context.session_service is None:
        return None, "Delegate task requires an active session."
    return context.session_service, None


def _build_child_user_input(request: DelegatedTaskRequest) -> str:
    lines = [
        "Delegated task:",
        request.task.strip(),
        "",
        "Success criteria:",
        request.success_criteria.strip(),
    ]
    if request.relevant_paths:
        lines.extend(["", "Relevant targets:"])
        lines.extend(f"- {path}" for path in request.relevant_paths)
    return "\n".join(lines)


def _format_subagent_summary(
    *,
    child_session_id: str,
    agent_id: str,
    success: bool,
    tool_runs: Sequence[ToolResult],
    final_text: str | None,
    stop_reason: str | None,
) -> str:
    tool_sequence = " -> ".join(tool_run.tool_name for tool_run in tool_runs) or "(none)"
    summary_text = _compact_summary(final_text, stop_reason=stop_reason)
    return "\n".join(
        [
            f"child_session_id={child_session_id}",
            f"agent_id={agent_id}",
            f"success={'true' if success else 'false'}",
            f"tool_sequence={tool_sequence}",
            f"final_summary={summary_text}",
        ]
    )


def _compact_summary(final_text: str | None, *, stop_reason: str | None) -> str:
    text = (final_text or "").strip()
    if not text:
        text = f"Subagent completed without a final text response (stop_reason={stop_reason or 'unknown'})."
    one_line = " ".join(text.split())
    return one_line[:400]


def _worker_tool_result(
    *,
    tool_call_id: str,
    success: bool,
    summary: str,
    stop_reason: str | None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name="delegate_task",
        success=success,
        content=summary,
        error=None if success else f"Subagent failed with stop reason '{stop_reason}'.",
    )


def _reviewed_tool_result(*, tool_call_id: str, summary: str, review: ReviewResult) -> ToolResult:
    content = "\n".join(
        [
            summary,
            f"review_decision={review.decision}",
            f"review_summary={review.summary or '(none)'}",
        ]
    )
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name="delegate_task",
        success=True,
        content=content,
        error=None,
    )


def _repair_request(request: DelegatedTaskRequest, review: ReviewResult) -> DelegatedTaskRequest:
    feedback = _format_feedback(review)
    return DelegatedTaskRequest(
        task=(
            f"{request.task.strip()}\n\n"
            "Repair the previous Worker attempt using this read-only review feedback:\n"
            f"{feedback}"
        ),
        success_criteria=request.success_criteria,
        relevant_paths=request.relevant_paths,
    )


def _format_feedback(review: ReviewResult) -> str:
    if review.feedback:
        return "\n".join(f"- {item}" for item in review.feedback)
    return review.summary or "No actionable feedback was provided."


def normalize_relevant_paths(raw_paths: Sequence[str] | None) -> tuple[str, ...]:
    if not raw_paths:
        return ()
    normalized = tuple(path.strip() for path in raw_paths if path.strip())
    return normalized[:_MAX_RELEVANT_PATHS]
