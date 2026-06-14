"""Deterministic scenario generation for multi-ticket episodes.

Generates reproducible queue scenarios from the existing email dataset
using a seed-based random number generator. Each scenario specifies queue
composition, SLA budgets, specialist accuracy, and drift schedule.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List


# Queue sizes and SLA budgets per difficulty tier
_TIER_CONFIG: Dict[str, Dict[str, Any]] = {
    "easy": {
        "queue_min": 1,
        "queue_max": 1,
        "sla_steps_per_ticket": 1,
        "specialist_accuracy": 0.95,
    },
    "medium": {
        "queue_min": 3,
        "queue_max": 5,
        "sla_steps_per_ticket": 3,
        "specialist_accuracy": 0.80,
    },
    "hard": {
        "queue_min": 5,
        "queue_max": 10,
        "sla_steps_per_ticket": 2,
        "specialist_accuracy": 0.75,
    },
    "adversarial": {
        "queue_min": 8,
        "queue_max": 15,
        "sla_steps_per_ticket": 2,
        "specialist_accuracy": 0.65,
    },
}


@dataclass
class TicketSlot:
    """One ticket in the queue with its SLA deadline."""

    email: Dict[str, Any]
    sla_deadline_step: int  # step number by which this ticket must be resolved


@dataclass
class Scenario:
    """A full episode scenario."""

    difficulty: str
    tickets: List[TicketSlot] = field(default_factory=list)
    specialist_accuracy: float = 0.85
    total_sla_budget: int = 1


def generate_scenario(
    emails: List[Dict[str, Any]],
    difficulty: str,
    seed: int = 0,
) -> Scenario:
    """Build a deterministic scenario from *emails* for the given *difficulty*.

    Arguments:
        emails: Full email dataset loaded from ``email_triage_dataset.json``.
        difficulty: One of ``easy``, ``medium``, ``hard``, ``adversarial``.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`Scenario` with ordered ticket queue and SLA deadlines.
    """
    rng = random.Random(seed)
    tier = _TIER_CONFIG.get(difficulty, _TIER_CONFIG["medium"])

    queue_size = rng.randint(tier["queue_min"], tier["queue_max"])
    sla_per = tier["sla_steps_per_ticket"]
    specialist_acc = tier["specialist_accuracy"]

    # Sample tickets matching difficulty, fall back to all if not enough
    diff_key = difficulty if difficulty != "adversarial" else "hard"
    candidates = [e for e in emails if e.get("difficulty") == diff_key]
    if len(candidates) < queue_size:
        candidates = list(emails)

    # Sample with replacement if needed
    selected: List[Dict[str, Any]] = []
    for _ in range(queue_size):
        selected.append(rng.choice(candidates))

    # Build ticket slots with cumulative SLA deadlines
    tickets: List[TicketSlot] = []
    for i, email in enumerate(selected):
        deadline = (i + 1) * sla_per
        tickets.append(TicketSlot(email=email, sla_deadline_step=deadline))

    total_sla = queue_size * sla_per

    return Scenario(
        difficulty=difficulty,
        tickets=tickets,
        specialist_accuracy=specialist_acc,
        total_sla_budget=total_sla,
    )
