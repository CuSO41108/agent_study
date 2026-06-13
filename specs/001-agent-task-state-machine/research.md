# Research Decisions

## Task and Session Boundary

**Decision**: A session contains conversation history; each non-waiting user goal
creates a task. User input received while the latest task is `waiting_user`
resumes that task.

**Rationale**: This preserves conversational memory without making a session one
unbounded lifecycle.

**Alternatives considered**: One task per turn loses clarification continuity;
one task per session makes terminal state and budgets ambiguous.

## Persistence Pattern

**Decision**: Store a mutable task snapshot plus append-only task events in one
SQLite transaction using optimistic task versions and task-local sequences.

**Rationale**: Snapshot reads remain simple while event history supports audit
and transition reconstruction.

## Human Waits

**Decision**: Persist pending actions before returning `waiting_user`. Immediate
CLI confirmation is represented as an approval/rejection event through the same
runtime path.

**Rationale**: One contract supports both current interactive behavior and
cross-process continuation.

## Retry Semantics

**Decision**: Retry only transient failures for tools without side effects or
tools explicitly marked idempotent, with two additional attempts.

**Rationale**: Retryability is a system policy; model prose cannot safely repeat
unknown side effects.

## Lightweight Reasoning

**Decision**: Keep ReAct as the policy. Generate a simple initial plan only for
multi-step requests, use deterministic critic checks for side effects and weak
final evidence, and create one reflection summary after repeated failure.

**Rationale**: This adds control without multiplying model calls on every step.
