---
name: sprint
description: Work on a batch of GitHub issues in parallel using Agent Teams. Creates one worktree per issue with TDD enforcement, coordinates via a lead agent, then produces stacked PRs.
---

# /sprint

Work on multiple GitHub issues in parallel. Each issue gets its own worktree
with TDD enforcement. A lead agent coordinates, and stacked PRs are created
when all work is done.

## EXECUTE THESE STEPS NOW

When this skill is invoked, you MUST execute these steps immediately.

### Step 0: Check Agent Teams Support

Check if Agent Teams are available:

```bash
echo "${CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS:-not_set}"
```

If the value is `not_set` or empty, fall back to **setup-only mode**:
- Parse the comma-separated issue numbers
- Fetch requirements for all issues (parallel issue-worker agents)
- Create worktrees and activate TDD for each
- Report the list of prepared worktrees and tell the user:
  "Agent Teams not enabled. Worktrees are prepared with TDD active.
  cd into each worktree and run `/write-tests` to begin the TDD cycle,
  or set `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` and re-run `/sprint`."
- Do NOT invoke `/work-on-issue` — it would create duplicate worktrees.
  The worktrees and TDD markers are already set up.

### Step 1: Parse Issue Numbers

Extract comma-separated issue numbers from `$ARGUMENTS`:
- Remove `#` prefixes, spaces, and other punctuation
- Example: `67,68,69,70` or `#67 #68 #69`
- Store as a list: `ISSUES=(67 68 69 70)`

### Step 2: Fetch Requirements (Parallel)

Spawn one issue-worker agent per issue, **all in a single message** so they
run in parallel:

```
Task tool (ALL in one message):
  For issue 67:
    subagent_type: issue-worker
    description: "issue-worker #67"
    prompt: "Read GitHub issue #67 and extract:
      1. Goal
      2. Acceptance criteria
      3. Edge cases and constraints
      4. Files likely to be touched"

  For issue 68:
    subagent_type: issue-worker
    description: "issue-worker #68"
    prompt: "Read GitHub issue #68 and extract: ..."

  (etc.)
```

Wait for all agents to return. Collect requirements for each issue.

### Step 3: Create Worktrees and Activate TDD

For each issue, create a worktree and activate TDD:

```bash
.claude/scripts/worktree-create.sh issue-<N>-<short-desc> && \
  cd .worktrees/issue-<N>-<short-desc> && \
  bash .claude/hooks/tdd-state.sh activate <N> && \
  cd -
```

### Step 4: Check for Conflicts

Before launching parallel work, check if any issues touch the same files:
- Compare the "files likely to be touched" from each issue
- If overlap detected, note it — the lead will need to mediate
- If issues are tightly coupled, warn the user and suggest sequential work

### Step 5: Create Agent Team

Create an Agent Team with one teammate per issue.

**Lead** (delegate mode — coordinates only, does not implement):
- Monitors teammate progress
- Mediates if teammates report conflicts on the same files
- After all complete: collects reports and determines PR ordering

**Teammates** (one per issue, spawned as parallel Task agents):

```
Task tool (ALL in one message):
  For issue 67:
    subagent_type: general-purpose
    description: "sprint-teammate #67"
    prompt: |
      You are working on GitHub issue #67.
      Working directory: .worktrees/issue-67-<desc>/
      TDD enforcement is active.

      Requirements:
      <requirements from step 2>

      Your workflow:
      1. cd into your worktree
      2. Create todos from acceptance criteria (TaskCreate)
      3. For each todo, use the Task tool to spawn:
         - subagent_type: tester (to write failing tests)
         - subagent_type: implementer (to make tests pass)
         Then run /update-docs if APIs changed
      4. When all todos complete, report:
         - Files you touched (git diff --name-only)
         - APIs you changed (old → new signatures)
         - Dependencies on other issues (if any)
         - Any conflicts you encountered

  For issue 68:
    subagent_type: general-purpose
    description: "sprint-teammate #68"
    prompt: ...

  (etc.)
```

### Step 6: Collect Results

After all teammates finish, collect from each:
- Files touched
- APIs changed
- Conflict reports
- Test pass/fail status

### Step 7: Create Stacked PRs

Spawn the pr-planner agent to determine dependency ordering:

```
Task tool:
  subagent_type: pr-planner
  description: "pr-planner for sprint"
  prompt: "Given these completed issues and their changes, determine
    the optimal PR ordering. Consider file dependencies, API changes,
    and which PRs can merge independently vs. which must be stacked.

    Issue reports:
    <collected reports>

    Output: ordered list of PRs with base branches"
```

Then create branches and PRs. For stacked PRs, rebase each branch onto the
previous one before creating the PR:

```bash
# First PR targets main directly
gh pr create --base main --head issue-67-branch ...

# Subsequent PRs: rebase onto the previous branch first
git checkout issue-68-branch
git rebase issue-67-branch
git push --force-with-lease
gh pr create --base issue-67-branch --head issue-68-branch ...
```

If rebase conflicts arise, report them to the user rather than auto-resolving.

### Step 8: Summary

Output a summary table:

```
## Sprint Complete

| Issue | PR | Base | Status | Files Changed |
|-------|-----|------|--------|---------------|
| #67 | #XX | main | Created | 3 files |
| #68 | #YY | issue-67-branch | Created | 5 files |
| #69 | #ZZ | issue-68-branch | Created | 2 files |

Stacked PR order: #XX → #YY → #ZZ
```

---

## When to Use

- Multiple related or independent issues to batch together
- You want maximum parallelism
- Issues are small-to-medium sized (each < ~200 lines of change)

## When NOT to Use

- Single issue (use `/work-on-issue` instead)
- Issues that are tightly coupled and must be done sequentially
- Very large issues (each needs its own focused session)

## Requirements

- `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` env var must be set to `1`
- Falls back to setup-only mode (worktree creation) if not available

## Architecture

```
/sprint 67,68,69
    │
    ├─ Fetch requirements (parallel issue-worker agents)
    ├─ Create worktrees + activate TDD
    ├─ Create Agent Team (lead = delegate, teammates = workers)
    │
    ├─ Teammate #67 (worktree, TDD, subagents for test/impl/docs)
    ├─ Teammate #68 (worktree, TDD, subagents for test/impl/docs)
    ├─ Teammate #69 (worktree, TDD, subagents for test/impl/docs)
    │
    ├─ Lead mediates conflicts
    ├─ pr-planner determines ordering
    ├─ Rebase for stacking (conflicts reported to user)
    └─ Stacked PRs created
```
