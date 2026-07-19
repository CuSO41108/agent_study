# Implementation Plan: Controlled Shell Approval

**Date**: 2026-07-11 | **Spec**: [spec.md](spec.md)

## Summary

Upgrade the existing PowerShell tool from a read-only whitelist to an
approval-controlled command capability. Recursive and batch deletion remain a
hard deny. Other commands require explicit user approval (or a Session-scoped
approved prefix), and persist their approval and outcome through the task trace.

## Technical Context

**Language**: Python 3.13  
**Dependencies**: Standard library, existing SQLite task persistence  
**Testing**: unittest and coverage.py  
**Platform**: Windows PowerShell local CLI

## Design

- `approval.py` rejects empty commands and forbidden recursive/batch deletion;
  arbitrary commands, operators, and unrecognized commands require user review.
- `ShellTool` starts approved commands from the workspace root. Approval does
  not claim to sandbox arbitrary command side effects to the workspace.
- The orchestrator uses per-invocation side-effect and idempotency methods so
  read-only shell actions retain existing recovery behavior while controlled
  mutations are not automatically retried.
- CLI approval renders command, risk, operation and affected paths.

## Constitution Check

- Persistent state: PASS — approved shell mutations use ToolAction/trace.
- Structured contracts: PASS — forbidden deletion is denied and every other
  Shell command requires explicit approval or a Session-scoped prefix.
- Crash consistency: PASS — controlled mutations are not retried automatically.
- Observable execution: PASS — approval, command and result are traced.
- Compatibility/tests: PASS — existing read-only shell behavior is regression tested.
