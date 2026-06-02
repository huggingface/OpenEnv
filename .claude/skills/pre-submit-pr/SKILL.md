---
name: pre-submit-pr
description: Validate changes before submitting a pull request. Run comprehensive checks including lint, tests, alignment review, and RFC analysis. Use before creating a PR, when asked if code is ready for review, or before pushing for PR.
allowed-tools: Read, Grep, Glob, Bash
---

# Pre-Submit PR Check

Comprehensive validation before submitting a pull request. Run this before creating or updating a PR.

## Instructions

1. **Check branch freshness** (BLOCKING):
   - Run `git fetch origin main` to get latest main
   - Run `git rev-list --count HEAD..origin/main` to check commits behind
   - If > 0 commits behind, merge main before proceeding: `git merge origin/main`
   - This prevents "branch out of date" issues on GitHub

2. **Run all automated hooks**:
   - `bash .claude/hooks/lint.sh` - format check (includes ruff format + ruff check)
   - `bash .claude/hooks/test.sh` - run tests
   - `bash .claude/hooks/check-debug.sh` - find debug code

3. **Run alignment review**:
   - Read `.claude/docs/PRINCIPLES.md` and `.claude/docs/INVARIANTS.md`
   - Compare changes against principles and invariants
   - Identify Tier 1 (mechanical) and Tier 2 (alignment) issues

4. **RFC check**:
   - If changes touch `src/openenv/core/`, flag for RFC consideration
   - If any public API signatures change, RFC required
   - Check against existing RFCs in `rfcs/` for conflicts

5. **Documentation freshness check**:
   - Review `.claude/docs/REPO_WALKTHROUGH.md` against the current repo structure
   - If the PR adds new directories, moves files, or changes structure significantly:
     - Update REPO_WALKTHROUGH.md to reflect the changes
     - Include these updates in the PR
   - Check triggers: new directories in `src/`, `envs/`, `.claude/`, or `rfcs/`
   - Check if any public API signatures changed (function/class renames, new params):
     - Search for references in docs/, examples/, README.md, other .py docstrings
     - If stale references found, recommend running `/update-docs` before PR

6. **Summarize PR readiness**:
   - List all blocking issues
   - List all discussion points for reviewers
   - Provide overall verdict

7. **After push/PR creation**, run the post-push check:
   - `bash .claude/hooks/post-push-pr.sh`
   - Verifies: PR is open, no merge conflicts, branch freshness,
     PR description quality, CI check status

## Output Format

```
## Pre-Submit PR Report

### Branch Freshness
| Check | Status | Details |
|-------|--------|---------|
| Up to date with main | YES/NO | [X commits behind, merged if needed] |

### Automated Checks
| Check | Status | Details |
|-------|--------|---------|
| Lint | PASS/FAIL | [summary] |
| Tests | PASS/FAIL | [X passed, Y failed] |
| Debug code | CLEAN/FOUND | [details] |

### Alignment Review

#### Tier 1: Fixes Required (blocking)
- [ ] path/file.py:123 - [issue description]

#### Tier 2: Discussion Points (flag for reviewers)
[ALIGNMENT FLAGS or "None identified"]

### Invariant Check
[List any invariants at risk, or "All invariants maintained"]

### RFC Status
[NOT REQUIRED / RECOMMENDED / REQUIRED: reason]

### Documentation Freshness
[UP TO DATE / UPDATED: list of changes made to REPO_WALKTHROUGH.md]
[STALE DOCS: run /update-docs — list of stale references found]

### Verdict: READY FOR PR / ISSUES TO ADDRESS

### Summary for PR Description
[2-3 sentences summarizing changes for the PR description]
```

## Blocking Issues

The following issues block PR submission:
- Branch out of date with main (must merge first)
- Lint failures
- Test failures
- Debugger statements (breakpoint, pdb)
- Invariant violations
- RFC required but not written

## Non-Blocking (Flag for Reviewers)

These should be noted in PR but don't block:
- Alignment discussion points (Tier 2)
- RFC recommended (optional)
- TODOs in code
- Print statements (unless in core code)
