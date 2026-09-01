"""Contract tests for the core validation enums."""

from openenv.validation.types import CheckStatus, Lane, Level, Severity, Verdict


def test_levels_are_ordered_by_cost():
    assert Level.STATIC < Level.RUNTIME < Level.SEMANTIC < Level.STATISTICAL


def test_grader_statuses_and_policy_severities_are_disjoint_vocabularies():
    # Graders emit CheckStatus; only the policy assigns Severity. SKIP/ERROR belong
    # to graders alone; ADVISORY belongs to the policy alone.
    statuses = {s.value for s in CheckStatus}
    severities = {s.value for s in Severity}
    assert "skip" in statuses and "skip" not in severities
    assert "error" in statuses and "error" not in severities
    assert "advisory" in severities and "advisory" not in statuses


def test_verdicts():
    assert {v.value for v in Verdict} == {"pass", "warn", "fail"}


def test_lanes():
    assert {lane.value for lane in Lane} == {"local", "hub"}
