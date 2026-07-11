# Implementation Plan: Controlled Shell Approval

**Date**: 2026-07-11 | **Spec**: [spec.md](spec.md)

## Summary

Upgrade the existing PowerShell tool from a read-only whitelist to a controlled
command capability. Read-only commands remain automatic. Strictly parsed
workspace directory creation, file moves and file copies enter the existing
approval lifecycle and persist their outcome through the task trace.

## Technical Context

**Language**: Python 3.13  
**Dependencies**: Standard library, existing SQLite task persistence  
**Testing**: unittest and coverage.py  
**Platform**: Windows PowerShell local CLI

## Design

- `approval.py` parses one command only; operators and unrecognized commands
  are denied.
- `ShellTool.inspect()` resolves every affected path within the workspace and
  rejects hidden/internal paths, missing sources, missing destination parents,
  and target conflicts.
- The orchestrator uses per-invocation side-effect and idempotency methods so
  read-only shell actions retain existing recovery behavior while controlled
  mutations are not automatically retried.
- CLI approval renders command, risk, operation and affected paths.

## Constitution Check

- Persistent state: PASS — approved shell mutations use ToolAction/trace.
- Structured contracts: PASS — only parsed command forms proceed.
- Crash consistency: PASS — controlled mutations are not retried automatically.
- Observable execution: PASS — approval, command and result are traced.
- Compatibility/tests: PASS — existing read-only shell behavior is regression tested.
