from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from agent_app.agent.definition import AgentDefinition, WORKER_AGENT
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
    ) -> None:
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

        child_session_id = session_service.create_session()
        child_loop = self._build_child_loop(delegation_depth=context.delegation_depth + 1)
        child_result = child_loop.run_turn(
            user_input=_build_child_user_input(request),
            session_id=child_session_id,
        )
        summary = _format_subagent_summary(
            child_session_id=child_result.session_id,
            agent_id=self._worker_agent.id,
            success=child_result.success,
            tool_runs=child_result.tool_runs,
            final_text=child_result.final_text,
            stop_reason=child_result.stop_reason,
        )
        session_service.append_subagent_run(
            parent_session_id=context.session_id or "",
            parent_tool_call_id=tool_call_id,
            child_session_id=child_result.session_id,
            agent_id=self._worker_agent.id,
            task=request.task,
            success=child_result.success,
            result_summary=summary,
        )

        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="delegate_task",
            success=child_result.success,
            content=summary,
            error=None if child_result.success else f"Subagent failed with stop reason '{child_result.stop_reason}'.",
        )

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


def normalize_relevant_paths(raw_paths: Sequence[str] | None) -> tuple[str, ...]:
    if not raw_paths:
        return ()
    normalized = tuple(path.strip() for path in raw_paths if path.strip())
    return normalized[:_MAX_RELEVANT_PATHS]
