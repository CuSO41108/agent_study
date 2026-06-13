# Data Model

## TaskState

- `id`, `session_id`, optional `parent_task_id`
- `goal`, `status`, `step`, `version`
- `plan_json`, `working_memory_json`
- `pending_action_json`, `last_observation_json`, `reflection_json`
- budget limits and used model/tool/token/active-time/replan counters
- `stop_reason`, `created_at`, `updated_at`, `waiting_deadline`

Terminal states are `completed`, `failed`, `cancelled`, and `expired`.

## AgentEvent

- immutable `id`, `task_id`, `session_id`
- `type`, `source`, `payload_json`
- optional `correlation_id`, `causation_id`
- task-local unique `sequence`, `created_at`

## Observation

- `status`, `error_type`, `message`
- `retryable`, `side_effect`
- `raw_data`, `evidence_ref`
- `attempt`, `duration_ms`

## PendingAction

- `kind`: `ask_user` or `tool_approval`
- user-facing prompt
- optional serialized decision/tool call and inspection summary
- creation and expiry timestamps

## State Transitions

```text
created -> running | cancelled
running -> waiting_user | waiting_tool | paused | completed | failed | cancelled | expired
waiting_user -> running | cancelled | expired
waiting_tool -> running | failed | cancelled | expired
paused -> running | cancelled
```

Terminal states have no outgoing transitions.
