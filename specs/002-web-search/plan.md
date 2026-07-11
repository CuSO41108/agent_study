# Implementation Plan: Web Research Tool

**Branch**: `codex/web-search` | **Date**: 2026-07-11 | **Spec**: [spec.md](spec.md)

## Summary

Add a Tavily-backed `web_search` tool to the existing single-process ReAct CLI.
It will obtain source-backed public-web results via an explicitly configured
HTTPS API. Explicit research requests are enforced as a pre-final-response
requirement; after the observation is available, the existing coordinator keeps
control of all ordinary tool-selection and task decisions.

## Technical Context

**Language/Version**: Python 3.13  
**Primary Dependencies**: Standard library, `jsonschema`; no new runtime package  
**Storage**: Existing workspace-local SQLite task/event/trace database  
**Testing**: `unittest`, coverage.py  
**Target Platform**: Local Windows/PowerShell CLI  
**Project Type**: Single Python package and CLI  
**Performance Goals**: A failed search respects its configured timeout; only a bounded result summary is appended to model context.  
**Constraints**: Keep API keys only in environment or `.agent_app/.env.local`; never persist or print credentials; preserve synchronous tools and the current OpenAI-compatible model client.  
**Scale/Scope**: One configured Tavily Search provider, one synchronous search per explicit research task before a final answer, at most the configured number of results.

## Constitution Check

- Persistent State First: PASS — the preflight result/failure becomes a durable
  task trace and tool action result.
- Structured Runtime Contracts: PASS — search arguments and normalized results
  have typed schemas; explicit research intent is a policy observation.
- Crash-Consistent Side Effects: PASS — searching is read-only and safe to
  retry; its result is recorded before downstream model work consumes it.
- Observable Execution: PASS — traces differentiate configuration, transport,
  HTTP, schema and empty-source outcomes from model errors.
- Compatibility and Tests: PASS — existing CLI signatures and local tools stay
  unchanged; unit/integration coverage is added.

## Research Decisions

See [research.md](research.md) for the provider contract and policy choices.

## Project Structure

### Documentation (this feature)

```text
specs/002-web-search/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── web-search-tool.md
└── tasks.md
```

### Source Code (repository root)

```text
src/agent_app/
├── agent/definition.py             # research-use policy instruction
├── config.py                       # search configuration loading/validation
├── orchestrator/loop.py            # explicit-research preflight and evidence
├── tools/
│   ├── registry.py                 # tool registration
│   └── web_search.py               # Tavily client and normalized results
└── types.py                         # typed search observation where needed

tests/
├── unit/
│   ├── test_config.py
│   ├── test_web_search.py
│   └── test_loop.py
└── integration/
    └── test_cli_flow.py
```

**Structure Decision**: Extend the existing synchronous tool registry and ReAct
loop. Do not add an orchestration framework, a separate planner/supervisor, or
an asynchronous worker.

## Complexity Tracking

No constitution violations.
