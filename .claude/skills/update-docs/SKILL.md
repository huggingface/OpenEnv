---
name: update-docs
description: Update documentation across the repo after API changes. Finds stale references in docs, examples, docstrings, and fixes them.
---

# /update-docs

Find and fix stale documentation after API changes.

## EXECUTE THESE STEPS NOW

When this skill is invoked, you MUST execute these steps immediately.

### Step 1: Identify Changed Files

Run:

```bash
git diff --name-only main...HEAD -- '*.py'
```

If no changes are found relative to main (e.g., on main or no upstream), fall back to:

```bash
git diff --name-only HEAD~1 -- '*.py'
```

If that also fails, ask the user which files changed.

Collect the list of changed Python files.

### Step 2: Extract API Changes

For each changed .py file, compare old vs new signatures using the same
ref that worked in Step 1:

```bash
# If Step 1 used main...HEAD:
git diff main...HEAD -- <file>

# If Step 1 fell back to HEAD~1:
git diff HEAD~1 -- <file>
```

Look for changes to:
- Function/method signatures (def lines)
- Class names and __init__ signatures
- Module-level constants and type aliases
- Removed or renamed public symbols

Build a list of `(old_signature, new_signature)` pairs.

If no API changes are found (only internal logic changes), report
"No API changes detected — no docs update needed" and stop.

### Step 3: Spawn Docs Updater Agent

Use the Task tool to spawn the docs-updater agent. IMPORTANT: the
`description` field MUST contain "docs-updater" so the SubagentStop
hook fires correctly.

```
Task tool:
  subagent_type: general-purpose
  description: "docs-updater propagation"
  prompt: |
    You are a docs-updater agent. Read .claude/agents/docs-updater.md
    for your full instructions.

    Here are the API changes to propagate:

    Changed files: <list>
    API changes:
    - `old` → `new`
    ...

    Search the entire repo for references to the old APIs and update
    them to match the new signatures. Follow the process in
    .claude/agents/docs-updater.md exactly.
```

### Step 4: Review Results

After the agent returns:
- Review the update report
- Verify no test files were touched
- Verify changes are minimal and correct

Report the summary to the user.

---

## When to Use

- After `/implement` when API signatures changed
- Before `/pre-submit-pr` to ensure docs are fresh
- When refactoring public APIs

## When NOT to Use

- Internal-only changes (no public API affected)
- Test-only changes
- Documentation-only changes (no code changed)

## Workflow Integration

```
/write-tests    →  Red (failing tests)
/implement      →  Green (passing tests)
/update-docs    →  Fix stale docs across repo  ← THIS SKILL
/simplify       →  Refactor (optional)
/pre-submit-pr  →  Validate before PR
```
