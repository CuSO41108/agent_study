# Quickstart Validation

1. Run all tests:

```powershell
python -m unittest discover -s tests -v
```

2. Run coverage:

```powershell
python -m coverage run -m unittest discover -s tests -v
python -m coverage report --precision=2 --fail-under=90
```

3. Start a normal task and retain the returned `task_id`:

```powershell
python -m agent_app.cli --workspace-root . "Inspect the repository status"
```

4. Query and control the task:

```powershell
python -m agent_app.cli --workspace-root . --task-status TASK_ID
python -m agent_app.cli --workspace-root . --pause-task TASK_ID
python -m agent_app.cli --workspace-root . --resume-task TASK_ID
python -m agent_app.cli --workspace-root . --cancel-task TASK_ID
```

5. For an edit requiring approval, decline or omit immediate approval, restart
the process, and resume with `--approve-task TASK_ID` or
`--reject-task TASK_ID`. Verify the action executes at most once.
