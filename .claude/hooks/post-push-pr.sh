#!/bin/bash
# Post-push PR validation. Run after `gh pr create` or `git push` to verify
# the PR looks good on GitHub.
#
# Usage: bash .claude/hooks/post-push-pr.sh [PR_NUMBER]
#
# If PR_NUMBER is omitted, uses the PR for the current branch.

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
PR_NUMBER="${1:-}"
FAILED=0

echo ""
echo "==================================================================="
echo "  Post-Push PR Checks"
echo "==================================================================="
echo ""

# Resolve PR number from current branch if not provided
if [[ -z "$PR_NUMBER" ]]; then
    PR_NUMBER=$(gh pr view --json number -q '.number' 2>/dev/null || true)
    if [[ -z "$PR_NUMBER" ]]; then
        echo "ERROR: No PR found for current branch."
        echo "  Create one with: gh pr create"
        exit 1
    fi
fi

echo "Checking PR #$PR_NUMBER..."
echo ""

# Fetch PR details in one call
PR_JSON=$(gh pr view "$PR_NUMBER" --json state,mergeable,baseRefName,headRefName,title,body,statusCheckRollup,commits 2>/dev/null)
if [[ -z "$PR_JSON" ]]; then
    echo "ERROR: Could not fetch PR #$PR_NUMBER"
    exit 1
fi

PR_STATE=$(echo "$PR_JSON" | jq -r '.state')
PR_MERGEABLE=$(echo "$PR_JSON" | jq -r '.mergeable')
PR_BASE=$(echo "$PR_JSON" | jq -r '.baseRefName')
PR_HEAD=$(echo "$PR_JSON" | jq -r '.headRefName')
PR_TITLE=$(echo "$PR_JSON" | jq -r '.title')
PR_BODY=$(echo "$PR_JSON" | jq -r '.body')
COMMIT_COUNT=$(echo "$PR_JSON" | jq '.commits | length')

# 1. PR is open
echo "=== PR State ==="
if [[ "$PR_STATE" == "OPEN" ]]; then
    echo "PASS: PR is open"
else
    echo "FAIL: PR state is '$PR_STATE'"
    FAILED=1
fi

# 2. Mergeable (no conflicts)
echo ""
echo "=== Merge Conflicts ==="
if [[ "$PR_MERGEABLE" == "MERGEABLE" ]]; then
    echo "PASS: No merge conflicts with $PR_BASE"
elif [[ "$PR_MERGEABLE" == "UNKNOWN" ]]; then
    echo "WARN: Mergeability not yet computed (check again shortly)"
else
    echo "FAIL: PR has merge conflicts with $PR_BASE"
    echo "  Rebase onto $PR_BASE to fix:"
    echo "    git fetch origin $PR_BASE"
    echo "    git rebase origin/$PR_BASE"
    echo "    git push --force-with-lease"
    FAILED=1
fi

# 3. Branch freshness (commits behind base)
echo ""
echo "=== Branch Freshness ==="
git fetch origin "$PR_BASE" --quiet 2>/dev/null || true
BEHIND=$(git rev-list --count HEAD.."origin/$PR_BASE" 2>/dev/null || echo "?")
if [[ "$BEHIND" == "0" ]]; then
    echo "PASS: Branch is up to date with $PR_BASE"
elif [[ "$BEHIND" == "?" ]]; then
    echo "WARN: Could not determine freshness"
else
    echo "FAIL: Branch is $BEHIND commit(s) behind $PR_BASE"
    echo "  Rebase to fix:"
    echo "    git rebase origin/$PR_BASE"
    echo "    git push --force-with-lease"
    FAILED=1
fi

# 4. PR description
echo ""
echo "=== PR Description ==="
BODY_LEN=${#PR_BODY}
if [[ "$BODY_LEN" -lt 50 ]]; then
    echo "WARN: PR description is very short ($BODY_LEN chars)"
    echo "  Consider adding a summary, change list, and test plan"
else
    echo "PASS: PR description present ($BODY_LEN chars)"
fi

# Check for test plan
if echo "$PR_BODY" | grep -qi "test plan"; then
    echo "PASS: Test plan section found"
else
    echo "WARN: No 'Test plan' section in PR description"
fi

# 5. CI status
echo ""
echo "=== CI Checks ==="
CHECK_COUNT=$(echo "$PR_JSON" | jq '.statusCheckRollup | length' 2>/dev/null || echo "0")
if [[ "$CHECK_COUNT" -gt 0 ]]; then
    PENDING=$(echo "$PR_JSON" | jq '[.statusCheckRollup[] | select(.status != "COMPLETED")] | length')
    FAILED_CHECKS=$(echo "$PR_JSON" | jq '[.statusCheckRollup[] | select(.conclusion == "FAILURE")] | length')
    PASSED_CHECKS=$(echo "$PR_JSON" | jq '[.statusCheckRollup[] | select(.conclusion == "SUCCESS")] | length')

    echo "$PASSED_CHECKS passed, $FAILED_CHECKS failed, $PENDING pending (of $CHECK_COUNT total)"

    if [[ "$FAILED_CHECKS" -gt 0 ]]; then
        echo ""
        echo "Failed checks:"
        echo "$PR_JSON" | jq -r '.statusCheckRollup[] | select(.conclusion == "FAILURE") | "  - \(.name)"'
        FAILED=1
    fi
    if [[ "$PENDING" -gt 0 ]]; then
        echo ""
        echo "Pending checks (re-run this script after they complete):"
        echo "$PR_JSON" | jq -r '.statusCheckRollup[] | select(.status != "COMPLETED") | "  - \(.name): \(.status)"'
    fi
else
    echo "WARN: No CI checks found (may still be starting)"
fi

# 6. Commit count
echo ""
echo "=== Commits ==="
echo "$COMMIT_COUNT commit(s) in this PR"

# Summary
echo ""
echo "==================================================================="
if [[ $FAILED -eq 1 ]]; then
    echo "  ISSUES FOUND — fix before requesting review"
else
    echo "  ALL CHECKS PASSED — ready for review"
fi
echo "==================================================================="
echo ""
echo "  PR: https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/pull/$PR_NUMBER"
echo ""

exit $FAILED
