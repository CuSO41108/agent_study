# Eval Demo Workflow

This demo is for showing that `agent_study` is a measurable coding agent rather
than a one-off CLI demo.

## Run The Eval Suite

```powershell
python -m evals.runner --dry-run
python -m evals.runner
python -m evals.runner --live-model
```

The runner writes JSONL reports to `evals/results/` and preserves per-case
workspaces under `.eval_runs/<run-id>/`. The default non-dry command reports
`live_model_not_requested` skips; pass `--live-model` to call the configured
model.

## Demo Tasks To Show

- `fix_single_file_001`: one-file code repair with test verification.
- `edit_multi_file_001`: coordinated code and documentation change.
- `add_test_001`: test-only change with implementation protected.
- `forbid_unsafe_shell_001`: unsafe shell call blocked by policy.
- `repair_loop_prompt_001`: failed verification can trigger a bounded repair.

## Evidence To Include

For a real model run, copy the following fields from the JSONL result into a
release note or README section:

- `score.task_pass`
- `score.verify_pass`
- `score.changed_files`
- `score.tool_calls`
- `score.retry_count`
- `score.repair_attempt_count`
- `turn_result.stop_reason`
- preserved workspace path

Do not promote a demo result unless both `verify_pass` and
`changed_files_pass` are true.
