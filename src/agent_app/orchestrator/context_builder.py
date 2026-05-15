from __future__ import annotations

from math import ceil

from agent_app.types import SessionContext, StoredMessage, ToolResult


def estimate_message_tokens(*, content: str | None) -> int:
    payload_cost = ceil(len((content or "").encode("utf-8")) / 4)
    return payload_cost + 12


def build_context_messages(
    *,
    messages: list[StoredMessage],
    session_context: SessionContext,
    context_token_budget: int,
    evidence_message: str | None = None,
) -> list[dict[str, str | None]]:
    if not messages:
        return []

    current_user_message = messages[-1]
    recent_messages = [
        message
        for message in messages[:-1]
        if session_context.summary_message_id is None or message.id > session_context.summary_message_id
    ]

    synthetic_messages: list[dict[str, str]] = []
    if session_context.summary_text:
        synthetic_messages.append({"role": "assistant", "content": f"Session summary:\n{session_context.summary_text}"})
    if session_context.todo_items:
        synthetic_messages.append(
            {
                "role": "assistant",
                "content": "Active todo list:\n" + "\n".join(
                    f"{index}. [{item.status}] {item.content}"
                    for index, item in enumerate(session_context.todo_items, start=1)
                ),
            }
        )
    if evidence_message:
        synthetic_messages.append({"role": "assistant", "content": evidence_message})

    provider_messages: list[dict[str, str | None]] = []
    token_budget = context_token_budget
    synthetic_cost = sum(estimate_message_tokens(content=message["content"]) for message in synthetic_messages)
    current_user_payload = {"role": current_user_message.role, "content": current_user_message.content}
    current_user_cost = estimate_message_tokens(content=current_user_message.content)
    token_budget -= synthetic_cost + current_user_cost

    selected_recent_messages: list[dict[str, str | None]] = []
    for message in reversed(recent_messages):
        message_payload = {"role": message.role, "content": message.content}
        message_cost = estimate_message_tokens(content=message.content)
        if token_budget < message_cost:
            continue
        selected_recent_messages.append(message_payload)
        token_budget -= message_cost

    provider_messages.extend(synthetic_messages)
    provider_messages.extend(reversed(selected_recent_messages))
    provider_messages.append(current_user_payload)
    return provider_messages


def estimate_messages_tokens(messages: list[StoredMessage]) -> int:
    return sum(estimate_message_tokens(content=message.content) for message in messages)


def build_evidence_message(tool_runs: list[ToolResult]) -> str | None:
    evidence_items: list[str] = []
    for tool_run in reversed(tool_runs):
        if not tool_run.success:
            continue
        evidence = _summarize_tool_result(tool_run)
        if evidence is None:
            continue
        evidence_items.append(f"- [{tool_run.tool_name}] {evidence}")
        if len(evidence_items) >= 4:
            break

    if not evidence_items:
        return None
    evidence_items.reverse()
    return "Recent successful tool evidence:\n" + "\n".join(evidence_items)


def _summarize_tool_result(tool_run: ToolResult) -> str | None:
    if tool_run.tool_name == "file_read":
        return _truncate_lines_and_chars(tool_run.content, max_lines=40, max_chars=2000)
    if tool_run.tool_name == "code_search":
        return _truncate_lines_and_chars(tool_run.content, max_lines=20, max_chars=2000)
    if tool_run.tool_name in {"replace_in_file", "file_write", "delegate_task"}:
        return tool_run.content.strip() or None
    if tool_run.tool_name == "shell" and tool_run.content:
        return _truncate_lines_and_chars(tool_run.content, max_lines=40, max_chars=2000)
    return None


def _truncate_lines_and_chars(content: str, *, max_lines: int, max_chars: int) -> str:
    preview = content[:max_chars]
    lines = preview.splitlines()[:max_lines]
    return "\n".join(lines)
