"""Specialist agent simulation for multi-agent oversight.

Simulates four specialist agents (triage, escalation, compliance, responder)
with configurable accuracy profiles. Each specialist processes an email and
returns a report that appears in the coordinator's observation.

Targeted bonus: Fleet AI — Scalable Oversight, Halluminate — Multi-Actor.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional


class SpecialistPool:
    """Pool of simulated specialist agents with accuracy profiles."""

    def __init__(self, base_accuracy: float = 0.85, seed: int = 0) -> None:
        self._rng = random.Random(seed + 1013)
        self._base = max(0.0, min(1.0, base_accuracy))

        # Per-specialist accuracy offsets (some are better/worse)
        self._accuracy: Dict[str, float] = {
            "triage": min(1.0, self._base + 0.05),
            "escalation": min(1.0, self._base + 0.00),
            "compliance": min(1.0, self._base + 0.10),
            "responder": min(1.0, self._base - 0.05),
        }

        # Per-specialist biases
        self._biases: Dict[str, Dict[str, Any]] = {
            "triage": {"under_prioritise": "billing"},
            "escalation": {"over_escalate_when_uncertain": True},
            "compliance": {"high_false_positive": True},
            "responder": {"formulaic": True},
        }

    @property
    def accuracy_profiles(self) -> Dict[str, float]:
        return dict(self._accuracy)

    def degrade(self, amount: float = 0.15) -> None:
        """Degrade all specialist accuracies (used after schema drift)."""
        for k in self._accuracy:
            self._accuracy[k] = max(0.3, self._accuracy[k] - amount)

    def simulate_all(self, email: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Run all four specialists on *email* and return their reports."""
        return {
            "triage": self._simulate_triage(email),
            "escalation": self._simulate_escalation(email),
            "compliance": self._simulate_compliance(email),
            "responder": self._simulate_responder(email),
        }

    # -- individual specialists ----------------------------------------------

    def _simulate_triage(self, email: Dict[str, Any]) -> Dict[str, Any]:
        true_cat = email.get("true_category", "other")
        true_pri = email.get("true_priority", 3)
        acc = self._accuracy["triage"]

        if self._rng.random() < acc:
            pred_cat = true_cat
        else:
            # Introduce bias: billing is often mis-classified as support
            categories = ["billing", "support", "spam", "urgent", "marketing", "other"]
            if true_cat == "billing":
                pred_cat = "support"  # systematic bias
            else:
                wrong = [c for c in categories if c != true_cat]
                pred_cat = self._rng.choice(wrong)

        if self._rng.random() < acc:
            pred_pri = true_pri
        else:
            pred_pri = max(1, min(5, true_pri + self._rng.choice([-1, 1])))

        return {
            "category": pred_cat,
            "priority": pred_pri,
            "confidence": round(acc + self._rng.uniform(-0.1, 0.1), 2),
            "correct": pred_cat == true_cat and pred_pri == true_pri,
        }

    def _simulate_escalation(self, email: Dict[str, Any]) -> Dict[str, Any]:
        needs = email.get("needs_escalation", False)
        acc = self._accuracy["escalation"]

        if self._rng.random() < acc:
            recommended = needs
        else:
            # Bias: over-escalates when uncertain
            recommended = True

        level: Optional[int] = None
        if recommended:
            level = 2 if email.get("true_priority", 3) >= 4 else 1

        return {
            "recommended": recommended,
            "level": level,
            "confidence": round(acc + self._rng.uniform(-0.1, 0.1), 2),
            "correct": recommended == needs,
        }

    def _simulate_compliance(self, email: Dict[str, Any]) -> Dict[str, Any]:
        true_cat = email.get("true_category", "other")
        acc = self._accuracy["compliance"]

        # Compliance checks for certain red-flag patterns
        has_risk = true_cat in ("urgent", "billing")

        if self._rng.random() < acc:
            flagged = has_risk
        else:
            # Bias: high false-positive rate
            flagged = True

        reason: Optional[str] = None
        if flagged:
            if true_cat == "urgent":
                reason = "Potential safety-critical incident"
            elif true_cat == "billing":
                reason = "Financial transaction review required"
            else:
                reason = "Flagged for routine compliance check"

        return {
            "flagged": flagged,
            "reason": reason,
            "confidence": round(acc + self._rng.uniform(-0.1, 0.1), 2),
            "correct": flagged == has_risk,
        }

    def _simulate_responder(self, email: Dict[str, Any]) -> Dict[str, Any]:
        true_cat = email.get("true_category", "other")

        templates = {
            "billing": "billing_ack",
            "support": "support_ticket_created",
            "spam": "spam_auto_filtered",
            "urgent": "urgent_incident_ack",
            "marketing": "marketing_unsubscribe",
            "other": "general_ack",
        }

        template_id = templates.get(true_cat, "general_ack")
        acc = self._accuracy["responder"]

        if self._rng.random() >= acc:
            # Wrong template
            wrong = [v for k, v in templates.items() if k != true_cat]
            template_id = self._rng.choice(wrong)

        return {
            "draft_ready": True,
            "template_id": template_id,
            "confidence": round(acc + self._rng.uniform(-0.1, 0.1), 2),
            "correct": template_id == templates.get(true_cat, "general_ack"),
        }
