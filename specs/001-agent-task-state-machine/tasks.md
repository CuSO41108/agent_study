# Tasks: Agent Task State Machine

## Phase 1: Speckit and Contracts

- [x] T001 Establish constitution and feature specification in `.specify/memory/constitution.md` and `specs/001-agent-task-state-machine/spec.md`
- [x] T002 Create plan, research, data model, runtime contract, and quickstart in `specs/001-agent-task-state-machine/`

## Phase 2: Foundational State

- [x] T003 [P] Add TaskState, AgentEvent, Observation, budget, pending-action, and trace types in `src/agent_app/types.py`
- [x] T004 Add backward-compatible task, event, and trace schema migrations in `src/agent_app/state/db.py`
- [x] T005 Implement atomic task/event/transition persistence in `src/agent_app/state/session_service.py`

## Phase 3: Durable Task Lifecycle

- [x] T006 [US1] Add transition validation and TaskState service behavior in `src/agent_app/runtime/task_runtime.py`
- [x] T007 [US1] Add `AgentRuntime.handle_event` and adapt `AgentLoop.run_turn` in `src/agent_app/orchestrator/loop.py`
- [x] T008 [US1] Migrate session todo into task plan and preserve session summaries in `src/agent_app/state/session_service.py`
- [x] T009 [US1] Add lifecycle, event-order, stale-version, migration, and restart tests in `tests/unit/test_task_runtime.py`

## Phase 4: Recoverable Human Input

- [x] T010 [US2] Persist pending clarification and tool-approval actions in `src/agent_app/runtime/task_runtime.py`
- [x] T011 [US2] Resume approval/rejection across processes and reuse ToolAction inspection/recovery in `src/agent_app/orchestrator/loop.py`
- [x] T012 [US2] Add waiting-user and cross-process approval tests in `tests/integration/test_cli_flow.py`

## Phase 5: Semantic ReAct Recovery

- [x] T013 [P] [US3] Add semantic tool metadata and Observation mapping in `src/agent_app/tools/base.py`
- [x] T014 [US3] Add executor exception boundary, safe retry policy, decision repetition detection, and lightweight reflection in `src/agent_app/orchestrator/loop.py`
- [x] T015 [US3] Enforce persisted model/tool/token/time/replan budgets in `src/agent_app/runtime/task_runtime.py`
- [x] T016 [US3] Add Observation, retry, side-effect, exception, and budget tests in `tests/unit/test_loop.py`

## Phase 6: Complete Trace and Controls

- [x] T017 [US4] Capture provider usage and model identity in `src/agent_app/model/openai_compatible.py`
- [x] T018 [US4] Persist model, decision, approval, tool-attempt, observation, budget, and transition traces in `src/agent_app/state/session_service.py`
- [x] T019 [US4] Add task query/pause/resume/cancel/approve/reject CLI controls in `src/agent_app/cli.py`
- [x] T020 [US4] Add trace completeness and CLI compatibility tests in `tests/unit/test_tracing.py` and `tests/integration/test_cli_flow.py`

## Phase 7: Validation

- [x] T021 Update architecture documentation in `README.md` and `docs/AGENT_CONTEXT_ENGINEERING.md`
- [x] T022 Run all tests, remove ResourceWarning, and keep core coverage at least 90 percent

## Dependencies

T003-T005 block all user stories. US1 establishes lifecycle, US2 adds human
waits, US3 adds semantic execution and budgets, and US4 completes trace and CLI
controls. Documentation and validation follow all stories.
