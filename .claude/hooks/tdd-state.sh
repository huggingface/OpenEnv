#!/bin/bash
# Shared TDD state helpers.
#
# Can be used two ways:
#   1. Sourced:  source tdd-state.sh && is_tdd_active
#   2. Direct:   bash tdd-state.sh activate 42
#
# TDD is activated by /work-on-issue, which writes .tdd-session.json
# to the worktree root. All hooks check this file instead of the
# .worktrees path, making TDD opt-in.

_tdd_toplevel() {
    git rev-parse --show-toplevel 2>/dev/null
}

is_tdd_active() {
    local toplevel
    toplevel=$(_tdd_toplevel) || return 1
    [[ -f "$toplevel/.tdd-session.json" ]]
}

get_tdd_issue() {
    local toplevel
    toplevel=$(_tdd_toplevel) || return 1
    jq -r '.issue // empty' "$toplevel/.tdd-session.json" 2>/dev/null
}

activate_tdd() {
    local issue="$1"
    if [[ -z "$issue" ]]; then
        echo "Usage: activate_tdd <issue-number>" >&2
        return 1
    fi
    local toplevel
    toplevel=$(_tdd_toplevel) || return 1
    local branch
    branch=$(git branch --show-current 2>/dev/null)

    jq -n \
        --arg issue "$issue" \
        --arg branch "$branch" \
        --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{issue: $issue, branch: $branch, activated_at: $ts}' \
        > "$toplevel/.tdd-session.json"

    echo "TDD enforcement activated for issue #$issue"
}

deactivate_tdd() {
    local toplevel
    toplevel=$(_tdd_toplevel) || return 1
    if [[ -f "$toplevel/.tdd-session.json" ]]; then
        rm "$toplevel/.tdd-session.json"
        echo "TDD enforcement deactivated"
    else
        echo "TDD was not active"
    fi
}

# When executed directly (not sourced), dispatch subcommands
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        activate)   activate_tdd "$2" ;;
        deactivate) deactivate_tdd ;;
        active)     is_tdd_active ;;
        issue)      get_tdd_issue ;;
        *)
            echo "Usage: bash $0 {activate <issue>|deactivate|active|issue}" >&2
            exit 1
            ;;
    esac
fi
