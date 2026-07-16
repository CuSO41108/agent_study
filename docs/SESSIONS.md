# Session progress in the REPL

The REPL stores task state, task plans, session todo items, summaries, and
pending user input in its local SQLite database. You do not need to query the
database directly for the usual progress view.

```text
/sessions           # latest eight sessions
/sessions 12        # choose one to twenty recent sessions
/progress 12        # alias for /sessions 12
```

Each row includes the session id, latest update time, task count, and its active
task when one exists. The details below it show the task goal, unfinished plan,
waiting prompt, session todo list, and compact summary. The current session is
marked with `*`.

This view is read-only: it never resumes, expires, cancels, or edits a task.
