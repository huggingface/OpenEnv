"""Mid-episode policy mutation engine for schema drift.

Injects policy changes during multi-turn episodes to test robustness
and adaptation. Supports: policy threshold changes, SLA window changes,
and specialist accuracy degradation triggers.

Targeted bonus: Patronus AI — Consumer Workflows with Schema Drift.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PolicyRule:
    """A single active policy rule in the environment."""

    rule_id: str
    description: str
    active: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule_id,
            "description": self.description,
            "active": self.active,
            "params": dict(self.params),
        }


@dataclass
class DriftEvent:
    """A scheduled drift event that fires at a given episode fraction."""

    trigger_fraction: float  # 0.0–1.0: fraction of queue when drift triggers
    drift_type: str
    description: str
    applied: bool = False


# ---------------------------------------------------------------------------
# Default policies active at the start of every episode
# ---------------------------------------------------------------------------

_DEFAULT_POLICIES: List[Dict[str, Any]] = [
    {
        "rule_id": "escalate_priority_ge_4",
        "description": "Escalate tickets with priority >= 4",
        "params": {"threshold": 4},
    },
    {
        "rule_id": "no_auto_close_urgent",
        "description": "Never close urgent tickets without escalation",
        "params": {},
    },
    {
        "rule_id": "spam_no_escalate",
        "description": "Never escalate spam tickets",
        "params": {},
    },
    {
        "rule_id": "internal_priority_boost",
        "description": "Internal senders get +1 priority consideration",
        "params": {"boost": 1},
    },
    {
        "rule_id": "sla_steps_per_ticket",
        "description": "Each ticket should be resolved within SLA step budget",
        "params": {"steps": 3},
    },
]


# ---------------------------------------------------------------------------
# Possible drift events (pool to sample from)
# ---------------------------------------------------------------------------

_DRIFT_POOL: List[Dict[str, Any]] = [
    {
        "drift_type": "escalation_threshold_lowered",
        "description": "Escalation threshold lowered from priority>=4 to priority>=3",
        "trigger_fraction": 0.35,
    },
    {
        "drift_type": "sla_tightened",
        "description": "SLA budget per ticket reduced from 3 steps to 2 steps",
        "trigger_fraction": 0.50,
    },
    {
        "drift_type": "spam_policy_relaxed",
        "description": "Spam tickets may now be escalated if sender is internal",
        "trigger_fraction": 0.40,
    },
    {
        "drift_type": "urgent_requires_review",
        "description": "All urgent tickets now require compliance review before close",
        "trigger_fraction": 0.30,
    },
    {
        "drift_type": "priority_scale_changed",
        "description": "Priority scale interpretation changed: 1-2=low, 3=medium, 4-5=critical",
        "trigger_fraction": 0.60,
    },
]


class DriftEngine:
    """Manages policy state and mid-episode schema drift."""

    def __init__(self, difficulty: str, seed: int = 0) -> None:
        self._rng = random.Random(seed + 7919)  # offset to decorrelate from data rng
        self._difficulty = difficulty
        self._policies: List[PolicyRule] = self._init_policies()
        self._schedule: List[DriftEvent] = self._build_schedule()
        self._applied_drifts: List[Dict[str, Any]] = []

    # -- initialisation ------------------------------------------------------

    def _init_policies(self) -> List[PolicyRule]:
        return [
            PolicyRule(
                rule_id=p["rule_id"],
                description=p["description"],
                active=True,
                params=dict(p.get("params", {})),
            )
            for p in _DEFAULT_POLICIES
        ]

    def _build_schedule(self) -> List[DriftEvent]:
        if self._difficulty in ("easy", "medium"):
            return []

        pool = list(_DRIFT_POOL)
        self._rng.shuffle(pool)

        n = 2 if self._difficulty == "hard" else min(4, len(pool))
        events: List[DriftEvent] = []
        for entry in pool[:n]:
            events.append(
                DriftEvent(
                    trigger_fraction=entry["trigger_fraction"],
                    drift_type=entry["drift_type"],
                    description=entry["description"],
                )
            )
        return events

    # -- public interface ----------------------------------------------------

    @property
    def active_policies(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._policies if p.active]

    @property
    def all_policies(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._policies]

    @property
    def drift_count(self) -> int:
        return len(self._applied_drifts)

    @property
    def applied_drifts(self) -> List[Dict[str, Any]]:
        return list(self._applied_drifts)

    def check_for_drift(
        self, queue_position: int, queue_size: int
    ) -> Optional[Dict[str, Any]]:
        """Check whether a drift event should fire at the current queue position.

        Returns drift info dict if a drift was applied, ``None`` otherwise.
        """
        if queue_size <= 0:
            return None

        fraction = queue_position / queue_size

        for event in self._schedule:
            if event.applied:
                continue
            if fraction >= event.trigger_fraction:
                return self._apply_drift(event)
        return None

    def check_compliance(
        self, action: Any, email: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Check whether *action* complies with currently active policies.

        Returns ``(is_compliant, list_of_violation_descriptions)``.
        """
        violations: List[str] = []

        for policy in self._policies:
            if not policy.active:
                continue

            if policy.rule_id.startswith("escalate_priority_ge"):
                threshold = policy.params.get("threshold", 4)
                true_pri = email.get("true_priority", 0)
                if true_pri >= threshold and not getattr(action, "should_escalate", False):
                    violations.append(
                        f"{policy.rule_id}: priority {true_pri} >= {threshold} requires escalation"
                    )

            elif policy.rule_id == "spam_no_escalate":
                if (
                    email.get("true_category") == "spam"
                    and getattr(action, "should_escalate", False)
                ):
                    # Check if spam policy was relaxed for internal senders
                    if not (
                        policy.params.get("allow_internal", False)
                        and email.get("is_internal", False)
                    ):
                        violations.append(
                            f"{policy.rule_id}: spam tickets must not be escalated"
                        )

            elif policy.rule_id == "no_auto_close_urgent":
                if (
                    email.get("true_category") == "urgent"
                    and not getattr(action, "should_escalate", False)
                ):
                    violations.append(
                        f"{policy.rule_id}: urgent tickets require escalation"
                    )

        return (len(violations) == 0), violations

    # -- internal ------------------------------------------------------------

    def _apply_drift(self, event: DriftEvent) -> Dict[str, Any]:
        event.applied = True
        result: Dict[str, Any] = {
            "drift_type": event.drift_type,
            "description": event.description,
        }

        if event.drift_type == "escalation_threshold_lowered":
            for p in self._policies:
                if p.rule_id == "escalate_priority_ge_4":
                    p.rule_id = "escalate_priority_ge_3"
                    p.description = "Escalate tickets with priority >= 3"
                    result["old_threshold"] = p.params.get("threshold", 4)
                    p.params["threshold"] = 3
                    result["new_threshold"] = 3
                    break

        elif event.drift_type == "sla_tightened":
            for p in self._policies:
                if p.rule_id == "sla_steps_per_ticket":
                    result["old_steps"] = p.params.get("steps", 3)
                    p.params["steps"] = 2
                    p.description = "Each ticket should be resolved within 2 steps"
                    result["new_steps"] = 2
                    break

        elif event.drift_type == "spam_policy_relaxed":
            for p in self._policies:
                if p.rule_id == "spam_no_escalate":
                    p.params["allow_internal"] = True
                    p.description = "Spam may be escalated if sender is internal"
                    result["change"] = "internal_spam_escalation_allowed"
                    break

        elif event.drift_type == "urgent_requires_review":
            self._policies.append(
                PolicyRule(
                    rule_id="urgent_needs_compliance_review",
                    description="Urgent tickets require compliance review",
                    active=True,
                    params={},
                )
            )
            result["new_policy"] = "urgent_needs_compliance_review"

        elif event.drift_type == "priority_scale_changed":
            result["change"] = "priority_interpretation_updated"
            # This affects how priority_grader evaluates buckets
            # Agents must adapt their priority assignments

        self._applied_drifts.append(result)
        return result
