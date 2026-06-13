<!--
Sync Impact Report
- Version change: template -> 1.0.0
- Added principles: Persistent State First; Structured Runtime Contracts;
  Crash-Consistent Side Effects; Observable Execution; Compatibility and Tests
- Added sections: Runtime Constraints; Development Workflow
- Removed sections: none
- Templates reviewed: plan-template.md, spec-template.md, tasks-template.md
- Follow-up TODOs: none
-->
# Agent Study Constitution

## Core Principles

### I. Persistent State First
Every user goal MUST execute through a persisted task lifecycle. Runtime state,
budgets, pending human actions, observations, and terminal reasons MUST survive
process restarts. Session history alone is not an acceptable task state model.

### II. Structured Runtime Contracts
External input MUST be normalized as an event, model output MUST become a typed
decision, and tool output or failure MUST become a typed observation. State
transitions MUST be validated by the runtime rather than inferred from prose.

### III. Crash-Consistent Side Effects
Actions with side effects MUST be recorded before execution, use idempotency or
recoverable evidence where possible, and never be automatically repeated when
their outcome is uncertain. State and trace persistence MUST not claim success
unless the corresponding action result is durably recorded.

### IV. Observable Execution
Model decisions, approvals, tool attempts, observations, budget changes, and
state transitions MUST be traceable with task correlation, timing, and failure
semantics. Required trace failures MUST surface as runtime failures and MUST NOT
be silently ignored.

### V. Compatibility and Tests
Existing CLI and Python entry points MUST remain compatible unless a breaking
change is explicitly specified. State migration, recovery, error semantics, and
control-flow changes MUST have unit and integration coverage. Core coverage MUST
remain at or above 90 percent.

## Runtime Constraints

The project remains a Python CLI-first harness using SQLite, the existing
OpenAI-compatible model adapter, and the existing tool registry. New orchestration
frameworks, hosted services, RBAC, tenant management, external metric platforms,
and asynchronous tool workers require a separate approved specification.

Planner, critic, and reflection capabilities MUST remain lightweight and
condition-triggered unless a future specification explicitly adopts a full
decision stack.

## Development Workflow

Features MUST proceed through specification, implementation planning, actionable
tasks, implementation, and verification. Existing uncommitted user work MUST be
preserved. Tests SHOULD be written before or alongside behavior changes, and the
full test suite plus coverage gate MUST pass before a feature is considered
complete.

## Governance

This constitution supersedes conflicting local guidance. Amendments require an
updated Sync Impact Report and semantic version change. Reviews MUST verify each
principle and document any justified exception in the implementation plan.

**Version**: 1.0.0 | **Ratified**: 2026-06-12 | **Last Amended**: 2026-06-12
