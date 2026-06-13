# Implementation Plan: Agent Task State Machine

**Branch**: `codex/crash-consistent-tool-actions` | **Date**: 2026-06-12 |
**Spec**: [spec.md](spec.md)

## Summary

Extend the existing CLI-first ReAct harness with a SQLite-backed TaskState and
append-only Event lifecycle. Reuse the in-progress crash-consistent ToolAction
layer, add semantic observations and hard budgets, persist human waits, and
replace turn-end-only tracing with correlated model/tool/state traces.

## Technical Context

**Language/Version**: Python 3.13

**Primary Dependencies**: Standard library, `jsonschema`

**Storage**: Existing workspace-local SQLite database

**Testing**: `unittest`, coverage.py

**Target Platform**: Local Windows/PowerShell CLI

**Project Type**: Single Python package and CLI

**Performance Goals**: State/event persistence adds less than 50 ms per local
transition under normal SQLite operation.

**Constraints**: Preserve current API and CLI behavior; no orchestration
framework; core coverage at least 90 percent.

**Scale/Scope**: One local process at a time, durable sessions and tasks, bounded
worker subagents, synchronous tools.

## Constitution Check

- Persistent task state and event sequencing: PASS
- Typed decisions and observations: PASS
- Crash-consistent side effects: PASS by extending existing ToolAction
- Required trace persistence: PASS
- Compatibility and test gate: PASS

## Project Structure

```text
src/agent_app/
├── runtime/          # task runtime, state machine, executor
├── orchestrator/     # backward-compatible ReAct adapter
├── state/            # SQLite task/event/trace persistence
├── tools/            # metadata and execution
├── model/            # usage-aware model responses
└── cli.py            # turn and task control commands

tests/
├── unit/
├── integration/
└── regression/
```

**Structure Decision**: Keep the existing package layout. Add task-runtime
modules under `runtime/` and extend existing persistence/types rather than
introducing another service layer.

## Complexity Tracking

No constitution violations.
