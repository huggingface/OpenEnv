---
name: terminus-terminal-task
description: Use inside a Terminus environment session when solving one sandboxed terminal task with the terminal tool.
---

# Terminus Terminal Task

Use this skill only inside a Terminus task session.

## Workflow

1. Read the task.
2. Use the `terminal` tool for each terminal action.
3. Pass `command` to inspect and modify the sandbox.
4. Check command output before choosing the next command.
5. When the task is complete, pass `final_answer` exactly once.

## Guardrails

- Do not change hidden checks or task configuration.
- Do not claim completion until the visible task requirements are satisfied.
- Stay focused on the current task and terminal outputs.
- Do not include both `command` and `final_answer` in the same tool call.
- For simple file writes, prefer commands like `printf %s 'text' > path`.
- If a command fails, inspect the error and continue with a smaller diagnostic
  command.
