# Tasks: Web Research Tool

**Input**: Design documents from `/specs/002-web-search/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/web-search-tool.md

**Tests**: Required by the project constitution for runtime and error-semantics changes.

## Phase 1: Setup

**Purpose**: Establish configuration examples and test fixtures without touching user configuration.

- [ ] T001 [P] Add safe `SEARCH_*` configuration documentation to README.md and .agent_app/.env.local example guidance in README.md
- [ ] T002 [P] Add Tavily response/error fixtures to tests/unit/test_web_search.py

---

## Phase 2: Foundational

**Purpose**: Provide typed configuration and a registered, synchronous read-only tool.

- [ ] T003 Add validated `SEARCH_BASE_URL`, `SEARCH_API_KEY`, `SEARCH_TIMEOUT`, and `SEARCH_MAX_RESULTS` fields to src/agent_app/config.py and tests/unit/test_config.py
- [ ] T004 Implement bounded Tavily request, response normalization, and secret-safe error handling in src/agent_app/tools/web_search.py
- [ ] T005 Register `web_search` and root-agent access in src/agent_app/tools/registry.py and src/agent_app/agent/definition.py
- [ ] T006 Add unit coverage for tool schema, successful results, HTTP errors, transport timeouts, malformed JSON, and no-source results in tests/unit/test_web_search.py

**Checkpoint**: A configured model can choose `web_search`, and all search outcomes are normalized safely.

---

## Phase 3: User Story 1 - Research current public information (Priority: P1) 🎯 MVP

**Goal**: Explicit research requests complete a source-backed web search before ReAct makes a final response.

**Independent Test**: A stubbed CLI task containing `查阅` has a successful `web_search` result before model finalization, and the model receives the source observation.

- [ ] T007 [US1] Add explicit-research intent detection and one read-only preflight search to src/agent_app/orchestrator/loop.py
- [ ] T008 [US1] Add source-observation context injection and search trace timing to src/agent_app/orchestrator/loop.py
- [ ] T009 [US1] Add ReAct policy text distinguishing required explicit research from autonomous ordinary tool selection in src/agent_app/agent/definition.py
- [ ] T010 [US1] Add loop tests for ordered preflight search and continued autonomous local-tool decision flow in tests/unit/test_loop.py
- [ ] T011 [US1] Add an end-to-end stubbed research-and-HTML task in tests/integration/test_cli_flow.py

**Checkpoint**: A “查阅 … 并编写 HTML” request has evidence before the model can compose or write its deliverable.

---

## Phase 4: User Story 2 - Receive an actionable unavailable-search result (Priority: P2)

**Goal**: Required research fails safely and visibly when it cannot be performed.

**Independent Test**: Missing configuration and each stubbed provider failure finish with a distinct search stop reason and no final answer.

- [ ] T012 [US2] Map required-search configuration and provider failure categories to explicit task stop reasons in src/agent_app/orchestrator/loop.py and src/agent_app/runtime/task_runtime.py
- [ ] T013 [US2] Persist safe required-search failure traces and state transitions in src/agent_app/orchestrator/loop.py and src/agent_app/state/session_service.py
- [ ] T014 [US2] Add unit tests for missing configuration, timeout, HTTP rejection, invalid response, and no-result termination in tests/unit/test_loop.py
- [ ] T015 [US2] Add integration coverage verifying required search failures are not reported as `model_error` in tests/integration/test_cli_flow.py

**Checkpoint**: Users can distinguish research failure from model failure without any memory-based fallback.

---

## Phase 5: User Story 3 - Configure and observe web research (Priority: P3)

**Goal**: Operators can configure research safely and diagnose each search from persisted traces.

**Independent Test**: A configured stubbed search produces correlated trace metadata without exposing the configured API key.

- [ ] T016 [US3] Add model/tool trace fields for search duration, source count and provider request ID in src/agent_app/orchestrator/loop.py
- [ ] T017 [US3] Add secret-redaction assertions and trace inspection tests in tests/unit/test_tracing.py
- [ ] T018 [US3] Update user configuration and diagnosis guidance in README.md and specs/002-web-search/quickstart.md

**Checkpoint**: Search actions are traceable by task ID, and credentials never enter persisted output.

---

## Phase 6: Polish & Validation

- [ ] T019 Reconcile all tests and public tool inventory documentation in README.md, tests/unit/test_tools.py, and tests/regression/test_regression_harness.py
- [ ] T020 Run the documented unit, integration, regression and coverage checks; record any remaining limitation in specs/002-web-search/quickstart.md

## Dependencies & Execution Order

- Phase 1 → Phase 2 → User Story 1.
- User Stories 2 and 3 depend on the tool and preflight path from User Story 1.
- User Story 1 is the MVP; it independently proves source-backed research.
- User Story 2 completes safe failure semantics; User Story 3 completes operability.

## Parallel Opportunities

- T001 and T002 are parallel.
- T003 and T004 can start independently but T004 must consume the finalized configuration API from T003 before integration.
- Within each user-story phase, tests sharing `loop.py` should be coordinated sequentially with the implementation task.

## Implementation Strategy

1. Build configuration and the bounded, read-only `web_search` tool.
2. Add explicit-research preflight and source propagation; validate the MVP.
3. Add terminal failure mapping and safe observability.
4. Run the full compatibility and coverage gate.
