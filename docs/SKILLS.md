# Skills

Skills are read-only, folder-scoped instructions that add a focused workflow to
an agent task. They are not MCP servers, executable plugins, or arbitrary code
downloaded by the model.

## Discovery roots

- Project-shared: `skills/<skill-name>/SKILL.md`
- User-global: `%USERPROFILE%/.agent-study/skills/<skill-name>/SKILL.md`

Both roots are scanned only when needed. A project Skill with the same name
shadows the user-global one. The agent never creates, edits, installs, or
deletes a Skill.

## Minimal Skill

```text
skills/
  review-change/
    SKILL.md
```

```markdown
---
name: review-change
description: Review a code change for correctness, tests, and regressions.
version: 1
invocation: code review, review a diff
---

1. Inspect the relevant diff and tests first.
2. Report findings with file and line evidence.
3. Do not change files unless the user asks for a fix.
```

`name` must be a lower-case slug and exactly match its containing folder.
`description` is required: it is what the model sees in the compact index and
uses to decide whether the Skill fits the task. Optional simple scalar fields
are `version`, `platforms`, `requires_tools`, and `invocation`.

## Progressive loading

1. Every task sees a bounded metadata index: name, description, scope and
   activation conditions.
2. A model calls `skill_load` only when the description fits, or a user selects
   it with `/skill <name>`. Its `SKILL.md` body is then pinned to the task by
   source path and content hash.
3. `skill_read_resource` can read a small support file only if that exact
   relative path is explicitly mentioned in the active `SKILL.md`.

The next turn reloads only active Skills. If a selected Skill moves, changes
scope, or changes content, it is not silently injected; a hash-mismatch trace
is recorded instead.

## REPL commands

- `/` shows the command menu; `prompt_toolkit` supports up/down and Enter.
- `/skills` lists the effective discovery index.
- `/skill <name>` selects a Skill for the next task turn.
- `/skill:<name>` is the completion-friendly shortcut.
- `/handoff [task-id-prefix]` creates a new session from a safe checkpoint.
  It copies only the goal, unfinished plan, compact summary, evidence
  references, and hash-verified active Skill references. Pending approvals and
  raw conversation/tool output are never transferred.

## Learn a new Skill safely

`/learn` is an interactive-only, explicitly user-initiated workflow. It learns
from the current **agent-app** session; it does not read a separate Codex chat
or automatically preserve an entire transcript.

```text
/learn project     # draft a repository Skill under skills/<name>/SKILL.md
/learn user        # draft a private reusable Skill under ~/.agent-study/skills/
/learn drafts      # list unsaved drafts for the current session
/learn save <id>   # create the previewed new Skill
```

The first command calls the model to produce a bounded, portable `SKILL.md`
preview and stores that preview in SQLite. No Skill directory or file is
created at that point. It redacts source credential assignments and rejects a
draft that appears to contain a credential. `/learn save` is the separate
explicit confirmation: it writes a **new** folder and `SKILL.md` in the chosen
scope. Existing Skill names and directories are never overwritten or updated.

Skill update and deletion commands are intentionally absent. A future design
may add a preview-only update workflow, but changing an existing Skill requires
a separate explicit design and confirmation model.
