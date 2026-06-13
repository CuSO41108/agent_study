# Feature Specification: Agent Task State Machine

**Feature Branch**: `codex/crash-consistent-tool-actions`

**Created**: 2026-06-12

**Status**: Approved

**Input**: Persist task state and normalized events, implement explicit lifecycle
transitions, semantic observations, budgets, recoverable user waits, and complete
execution traces while retaining the existing ReAct loop.

## User Scenarios & Testing

### User Story 1 - Durable Task Lifecycle (Priority: P1)

A CLI user can start a goal, stop the process, and later inspect or continue the
same task without losing its lifecycle, budget, plan, or latest observation.

**Why this priority**: All other behavior depends on a durable task identity and
validated state transitions.

**Independent Test**: Start a task, reload the database in a new runtime, and
verify the same task state and event sequence are returned.

**Acceptance Scenarios**:

1. **Given** a new goal, **When** it is submitted, **Then** one task is created
   and transitions from `created` to `running`.
2. **Given** a terminal task, **When** a new goal is submitted in the same
   session, **Then** a new task is created.
3. **Given** an illegal or stale transition, **When** it is requested, **Then**
   the transition is rejected without changing the task snapshot or event log.

---

### User Story 2 - Recoverable Human Input (Priority: P1)

A task that needs clarification or edit approval enters `waiting_user` and can be
approved, rejected, or answered from a later CLI process.

**Why this priority**: Human interaction must be a normal resumable state rather
than a blocking callback.

**Independent Test**: Persist a pending edit, restart the runtime, approve it by
task id, and verify execution resumes exactly once.

**Acceptance Scenarios**:

1. **Given** an edit requiring confirmation, **When** no immediate confirmation
   is available, **Then** the task persists the pending action and returns
   `waiting_user`.
2. **Given** a waiting task, **When** the user approves or rejects it, **Then**
   the event is recorded and the task resumes or completes without duplicate
   side effects.

---

### User Story 3 - Semantic ReAct Recovery (Priority: P2)

The ReAct loop receives structured observations, retries only safe transient
failures, and terminates with an explicit reason when recovery is unsafe or its
budget is exhausted.

**Why this priority**: Error strings alone cannot support reliable autonomous
recovery.

**Independent Test**: Make a read-only tool time out twice and verify attempts,
backoff classification, budget usage, final observation, and stop reason.

**Acceptance Scenarios**:

1. **Given** a retryable read-only failure, **When** attempts remain, **Then**
   the executor retries at most twice and records every attempt.
2. **Given** a failed side-effect action, **When** retryability is uncertain,
   **Then** the runtime does not automatically repeat it.
3. **Given** an unexpected tool exception, **When** execution fails, **Then** a
   runtime-error observation and trace are persisted.

---

### User Story 4 - Complete Execution Trace (Priority: P2)

A developer can reconstruct why a task stopped by inspecting model, decision,
approval, tool, observation, budget, and state-transition traces.

**Why this priority**: The state machine must be diagnosable before broader
governance is added.

**Independent Test**: Run a tool task and verify all correlated trace records,
durations, parameters, token usage source, attempts, and transitions.

**Acceptance Scenarios**:

1. **Given** a completed task, **When** traces are queried, **Then** every model
   call, tool attempt, observation, and state transition is linked to its task.
2. **Given** a provider without usage data, **When** a model call completes,
   **Then** estimated token usage is recorded and labelled as estimated.
3. **Given** a required trace write failure, **When** execution continues,
   **Then** the task fails with a trace-persistence reason instead of hiding it.

### Edge Cases

- Duplicate events and stale task versions do not apply twice.
- Terminal tasks reject resume, approval, and new observations.
- Waiting tasks expire after their configured deadline.
- An uncertain side-effect action blocks automatic continuation.
- A model repeatedly emits the same decision and reaches the repetition limit.
- Existing databases without task tables migrate without losing sessions.

## Requirements

### Functional Requirements

- **FR-001**: The system MUST persist one TaskState per active user goal.
- **FR-002**: The system MUST normalize task input and control operations as
  immutable AgentEvents with task-local sequence numbers.
- **FR-003**: The system MUST validate all lifecycle transitions and reject stale
  versions atomically.
- **FR-004**: The system MUST persist task budget consumption and structured
  terminal reasons.
- **FR-005**: The system MUST expose a runtime event entry point while preserving
  the existing `run_turn` API.
- **FR-006**: The system MUST persist pending clarifications and approvals in
  `waiting_user` and resume them across processes.
- **FR-007**: The system MUST convert tool success, validation failures, denials,
  timeouts, conflicts, unexpected exceptions, and uncertain effects into
  structured Observations.
- **FR-008**: The system MUST retry only read-only or explicitly idempotent
  transient actions, at most twice.
- **FR-009**: The system MUST enforce model-call, tool-call, token, elapsed-time,
  repeated-decision, and replan budgets.
- **FR-010**: The system MUST record correlated model, decision, approval, tool,
  observation, budget, and transition traces.
- **FR-011**: Lightweight planning MAY run for a new multi-step task; deterministic
  criticism MUST guard side effects and unsupported final claims; reflection MAY
  run once after repeated failure.
- **FR-012**: Existing session, tool-action recovery, CLI, and JSON result
  behavior MUST remain backward compatible.

### Key Entities

- **TaskState**: Durable snapshot of one goal, lifecycle, plan, memory, pending
  action, latest observation, budget, terminal reason, and version.
- **AgentEvent**: Immutable normalized input or control signal ordered within a
  task.
- **Observation**: Semantic outcome of an action attempt.
- **TaskBudget**: Configured limits and persisted consumption.
- **PendingAction**: Clarification or tool approval awaiting a user event.
- **ExecutionTrace**: Correlated model, tool, observation, approval, and state
  transition evidence.

## Success Criteria

### Measurable Outcomes

- **SC-001**: All supported lifecycle transitions survive process restart with no
  duplicate event application.
- **SC-002**: All tool exceptions produce a persisted observation and terminal or
  recoverable task state.
- **SC-003**: Safe retries never exceed two additional attempts and side-effect
  actions are never automatically retried.
- **SC-004**: Every completed or failed task has a structured terminal reason and
  correlated state-transition trace.
- **SC-005**: Existing tests remain green, new lifecycle scenarios pass, and core
  code coverage remains at least 90 percent.

## Assumptions

- Existing crash-consistent ToolAction changes are retained as the action layer.
- Current tools remain synchronous; `waiting_tool` is reserved in contracts only.
- Planner, critic, and reflection are lightweight triggers, not a separate
  always-on decision pipeline.
- Governance platforms, users, tenants, RBAC, alerts, and external metrics are
  outside this feature.
