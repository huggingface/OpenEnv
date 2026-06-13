"""Deterministic reward graders for the Email Triage environment.

All graders are pure functions that return scores in ``[0.0, 1.0]``.
No neural models or LLM judges — every score is reproducible from
the action and ground-truth labels alone.

Round 1 base graders (unchanged):
    category_grader, priority_grader, escalation_grader,
    easy_task_grader, medium_task_grader, hard_task_grader, task_grader

Round 2 multi-turn graders (new):
    sla_grader, oversight_grader, efficiency_grader,
    policy_compliance_grader, drift_adaptation_grader
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    from envs.email_triage_env.models import EmailTriageAction, TaskId
except ImportError:
    try:
        from email_triage_env.models import EmailTriageAction, TaskId
    except ImportError:
        from models import EmailTriageAction, TaskId


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Round 1 — Base graders (unchanged)
# ---------------------------------------------------------------------------

def category_grader(action: EmailTriageAction, email: Dict[str, Any]) -> float:
    """Returns 1.0 if predicted category matches true_category,
    0.0 otherwise.
    """
    return 1.0 if action.category == email["true_category"] else 0.0


def priority_grader(action: EmailTriageAction, email: Dict[str, Any]) -> float:
    """Returns:
      1.0 if same bucket (low/med/high)
      0.5 if off by 1 bucket
      0.0 otherwise
    """

    def bucket(p: int) -> int:
        if p <= 2:
            return 0
        if p == 3:
            return 1
        return 2

    true_b = bucket(email["true_priority"])
    act_b = bucket(action.priority)
    if true_b == act_b:
        return 1.0
    if abs(true_b - act_b) == 1:
        return 0.5
    return 0.0


def escalation_grader(action: EmailTriageAction, email: Dict[str, Any]) -> float:
    """Returns 1.0 if escalation decision matches,
    lower for harmful mismatches (spam escalated, urgent ignored).
    """
    if action.should_escalate == email["needs_escalation"]:
        return 1.0

    if email["true_category"] == "spam" and action.should_escalate:
        return 0.0
    if email["true_category"] == "urgent" and not action.should_escalate:
        return 0.0

    return 0.5


# ---------------------------------------------------------------------------
# Round 1 — Task graders (unchanged)
# ---------------------------------------------------------------------------

def easy_task_grader(action: EmailTriageAction, email: Dict[str, Any]) -> float:
    """Easy task: category classification only."""
    return category_grader(action, email)


def medium_task_grader(action: EmailTriageAction, email: Dict[str, Any]) -> float:
    """Medium task: category plus priority bucket quality."""
    score = 0.7 * category_grader(action, email) + 0.3 * priority_grader(action, email)
    return _clamp_01(score)


def hard_task_grader(action: EmailTriageAction, email: Dict[str, Any]) -> float:
    """Hard task: full triage quality with safety-sensitive escalation."""
    score = (
        0.5 * category_grader(action, email)
        + 0.2 * priority_grader(action, email)
        + 0.3 * escalation_grader(action, email)
    )

    # Apply a stronger penalty for clearly harmful mistakes.
    if email["true_category"] == "spam" and action.should_escalate:
        score -= 0.3
    if email["true_category"] == "urgent" and not action.should_escalate:
        score -= 0.3

    return _clamp_01(score)


def task_grader(task_id: TaskId, action: EmailTriageAction, email: Dict[str, Any]) -> float:
    """Dispatch to one of the deterministic task graders."""
    if task_id == "easy":
        return easy_task_grader(action, email)
    if task_id == "medium":
        return medium_task_grader(action, email)
    # hard and adversarial share the same base grader
    return hard_task_grader(action, email)


# ---------------------------------------------------------------------------
# Round 2 — Multi-turn graders (new)
# ---------------------------------------------------------------------------

def sla_grader(
    tickets_resolved: int,
    sla_breaches: int,
) -> float:
    """Episode-level SLA adherence: ``1 - breaches / resolved``.

    Returns 1.0 when no tickets have been resolved yet (episode start).
    """
    if tickets_resolved <= 0:
        return 1.0
    return _clamp_01(1.0 - sla_breaches / tickets_resolved)


def oversight_grader(
    oversight_catches: int,
    total_specialist_errors: int,
) -> float:
    """Fraction of specialist errors the coordinator caught and corrected.

    Returns 1.0 when no specialist errors exist (nothing to catch).
    """
    if total_specialist_errors <= 0:
        return 1.0
    return _clamp_01(oversight_catches / total_specialist_errors)


def efficiency_grader(
    total_steps: int,
    tickets_resolved: int,
) -> float:
    """Coordination efficiency: ideal is 1 step per ticket.

    ``score = 1.0 - (excess_steps / total_steps)`` where
    ``excess_steps = max(0, total_steps - tickets_resolved)``.

    Returns 1.0 when every step resolved exactly one ticket.
    """
    if total_steps <= 0:
        return 1.0
    excess = max(0, total_steps - tickets_resolved)
    return _clamp_01(1.0 - excess / total_steps)


def policy_compliance_grader(
    total_decisions: int,
    policy_violations: int,
) -> float:
    """Fraction of decisions that comply with current active policies.

    Returns 1.0 when no decisions have been made yet.
    """
    if total_decisions <= 0:
        return 1.0
    return _clamp_01(1.0 - policy_violations / total_decisions)


def drift_adaptation_grader(
    steps_since_last_drift: Optional[int],
    compliant_after_drift: bool,
) -> float:
    """Bonus reward for adapting to a policy drift within 2 steps.

    Returns 0.2 if the agent adapted quickly, 0.0 otherwise.
    """
    if steps_since_last_drift is None:
        return 0.0
    if steps_since_last_drift <= 2 and compliant_after_drift:
        return 0.2
    return 0.0


def compute_multi_turn_reward(
    action: EmailTriageAction,
    email: Dict[str, Any],
    task_id: TaskId,
    weights: Dict[str, float],
    episode_stats: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Compute the full multi-turn reward for one step.

    Returns ``(total_reward, component_scores)`` where component_scores
    is a dict of each named reward component for logging.
    """

    # 1. Resolution quality (same base graders)
    quality = (
        0.5 * category_grader(action, email)
        + 0.2 * priority_grader(action, email)
        + 0.3 * escalation_grader(action, email)
    )
    quality = _clamp_01(quality)

    # 2. SLA
    sla = sla_grader(
        episode_stats.get("tickets_resolved", 0),
        episode_stats.get("sla_breaches", 0),
    )

    # 3. Policy compliance
    policy = policy_compliance_grader(
        episode_stats.get("total_decisions", 0),
        episode_stats.get("policy_violations", 0),
    )

    # 4. Oversight
    oversight = oversight_grader(
        episode_stats.get("oversight_catches", 0),
        episode_stats.get("total_specialist_errors", 0),
    )

    # 5. Efficiency
    eff = efficiency_grader(
        episode_stats.get("total_steps", 0),
        episode_stats.get("tickets_resolved", 0),
    )

    # 6. Drift adaptation bonus
    drift_bonus = drift_adaptation_grader(
        episode_stats.get("steps_since_last_drift"),
        episode_stats.get("compliant_after_drift", False),
    )

    # Weighted sum
    w_quality = weights.get("quality", 0.30)
    w_sla = weights.get("sla", 0.20)
    w_policy = weights.get("policy", 0.20)
    w_oversight = weights.get("oversight", 0.15)
    w_efficiency = weights.get("efficiency", 0.15)

    total = (
        w_quality * quality
        + w_sla * sla
        + w_policy * policy
        + w_oversight * oversight
        + w_efficiency * eff
        + drift_bonus
    )

    # Penalties
    if email.get("true_category") == "spam" and action.should_escalate:
        total -= 0.5
    if email.get("true_category") == "urgent" and not action.should_escalate:
        total -= 0.5

    components = {
        "quality": round(quality, 4),
        "sla": round(sla, 4),
        "policy": round(policy, 4),
        "oversight": round(oversight, 4),
        "efficiency": round(eff, 4),
        "drift_bonus": round(drift_bonus, 4),
        "total": round(total, 4),
    }

    return total, components
