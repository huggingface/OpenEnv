---
name: work-on-issue
description: Start work on a GitHub issue. Extracts requirements, creates worktree, sets up TDD workflow.
---

# /work-on-issue

Start focused work on a GitHub issue using TDD workflow.

## EXECUTE THESE STEPS NOW

When this skill is invoked, you MUST execute these steps immediately. Do NOT just describe what will happen - actually do it.

### Step 1: Parse Issue Number

Extract the issue number from `$ARGUMENTS`:
- Remove `#` prefix if present
- The issue number is: **$ARGUMENTS**

### Step 2: Spawn Issue Worker Agent

Use the Task tool to spawn the issue-worker agent:

```
Task tool:
  subagent_type: issue-worker
  prompt: "Read GitHub issue #<NUMBER> and extract:
    1. Goal - what the user wants to achieve
    2. Acceptance criteria - specific testable requirements
    3. Edge cases and constraints
    4. Suggested PR split if complex"
```

Wait for the agent to return requirements.

### Step 3: Create Worktree

After receiving requirements, run this command:

```bash
.claude/scripts/worktree-create.sh issue-<NUMBER>-<short-description>
```

Where `<short-description>` is 2-3 words from the goal (e.g., `add-mcp-tools`).

### Step 4: Activate TDD Enforcement

Activate TDD enforcement in the new worktree. This uses `tdd-state.sh`'s
direct-execution mode so it works in a single Bash call:

```bash
cd .worktrees/issue-<NUMBER>-<short-description> && bash .claude/hooks/tdd-state.sh activate <NUMBER>
```

This writes `.tdd-session.json` to the worktree root, which all hooks check.
Without this step, hooks would not block direct edits.

### Step 5: Create Todos

Use TodoWrite to create a todo for EACH acceptance criterion:

```
TodoWrite:
  todos:
    - content: "Test: <acceptance criterion 1>"
      status: pending
      activeForm: "Testing <criterion 1>"
    - content: "Test: <acceptance criterion 2>"
      status: pending
      activeForm: "Testing <criterion 2>"
    ...
```

### Step 6: Begin TDD Cycle

Immediately invoke `/write-tests` for the first todo.

DO NOT stop and wait for user input. Start the TDD cycle now.

---

## When to Use

- Starting work on a GitHub issue
- You want TDD enforcement (opt-in via this skill)
- You want isolated work (no branch switching)

## When NOT to Use

- Quick exploration (just stay in main repo)
- Already in a worktree for this issue
- Issue doesn't exist in GitHub

## Workflow Overview

```
/work-on-issue #42
    ↓
Step 1: Parse "42" from arguments
    ↓
Step 2: Spawn issue-worker → get requirements
    ↓
Step 3: Create worktree issue-42-<name>
    ↓
Step 4: Activate TDD enforcement (.tdd-session.json)
    ↓
Step 5: Create todos from acceptance criteria
    ↓
Step 6: Invoke /write-tests → begin TDD cycle
```

## Important

This skill runs in the MAIN conversation context (not forked) because it needs to:
1. Spawn the issue-worker agent and receive its results
2. Run worktree-create.sh script
3. Create todos that persist in the conversation
4. Invoke /write-tests to continue the workflow

The issue-worker agent runs in a forked context and returns requirements.
