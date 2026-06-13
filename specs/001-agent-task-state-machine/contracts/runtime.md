# Runtime and CLI Contract

## Python

`AgentRuntime.handle_event(event: AgentEvent) -> TaskResult`

`AgentLoop.run_turn(user_input: str, session_id: str | None = None) -> TurnResult`
remains supported and delegates to the runtime.

`TurnResult` retains existing fields and adds optional `task_id`, `task_status`,
and `pending_action`.

## CLI

Existing prompt and interactive commands remain valid.

Additional mutually exclusive controls:

```text
--task-status TASK_ID
--pause-task TASK_ID
--resume-task TASK_ID
--cancel-task TASK_ID
--approve-task TASK_ID
--reject-task TASK_ID
```

Control commands emit the same JSON serialization style as normal turns.

## Event Types

`task_created`, `user_message`, `user_approved`, `user_rejected`,
`resume_requested`, `pause_requested`, `cancel_requested`, and `task_expired`.

Duplicate event ids and stale expected versions are rejected without mutation.
