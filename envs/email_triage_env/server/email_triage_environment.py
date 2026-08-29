"""Oversight Inbox Arena — multi-turn email triage environment.

Round 1 backward compatibility: ``easy`` mode produces single-step episodes
identical to the original implementation (one email, ``done=True`` after step).

Round 2 upgrades:
- Multi-ticket queue episodes (``medium`` / ``hard`` / ``adversarial``)
- Specialist agent simulation with oversight scoring
- Mid-episode schema drift (policy mutations)
- Composite deterministic reward (quality + SLA + policy + oversight + efficiency)
"""

import json
import os
import random
import uuid
from typing import Any, Dict, List, Optional

try:
    from openenv.core.env_server import Environment
except ImportError:
    try:
        from openenv_core.env_server import Environment
    except ImportError:
        try:
            from core.env_server import Environment
        except ImportError:
            # Last resort: import just the base class, skip http/mcp deps
            try:
                from openenv.core.env_server.interfaces import Environment
            except ImportError:
                from core.env_server.interfaces import Environment

try:
    from envs.email_triage_env.models import (
        Difficulty,
        EmailTriageAction,
        EmailTriageObservation,
        EmailTriageState,
        TaskId,
    )
except ImportError:
    try:
        from email_triage_env.models import (
            Difficulty,
            EmailTriageAction,
            EmailTriageObservation,
            EmailTriageState,
            TaskId,
        )
    except ImportError:
        from models import (
            Difficulty,
            EmailTriageAction,
            EmailTriageObservation,
            EmailTriageState,
            TaskId,
        )

try:
    from envs.email_triage_env.server.graders import (
        category_grader,
        compute_multi_turn_reward,
        escalation_grader,
        priority_grader,
        task_grader,
    )
except ImportError:
    try:
        from email_triage_env.server.graders import (
            category_grader,
            compute_multi_turn_reward,
            escalation_grader,
            priority_grader,
            task_grader,
        )
    except ImportError:
        from server.graders import (
            category_grader,
            compute_multi_turn_reward,
            escalation_grader,
            priority_grader,
            task_grader,
        )

try:
    from envs.email_triage_env.server.scenario_generator import generate_scenario
except ImportError:
    try:
        from email_triage_env.server.scenario_generator import generate_scenario
    except ImportError:
        from server.scenario_generator import generate_scenario

try:
    from envs.email_triage_env.server.stakeholders import SpecialistPool
except ImportError:
    try:
        from email_triage_env.server.stakeholders import SpecialistPool
    except ImportError:
        from server.stakeholders import SpecialistPool

try:
    from envs.email_triage_env.server.schema_drift import DriftEngine
except ImportError:
    try:
        from email_triage_env.server.schema_drift import DriftEngine
    except ImportError:
        from server.schema_drift import DriftEngine


# ---------------------------------------------------------------------------
# Task / difficulty configuration
# ---------------------------------------------------------------------------

# Valid action categories (locked — model cannot invent new ones)
_VALID_CATEGORIES = frozenset({"billing", "support", "spam", "urgent", "marketing", "other"})

TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "easy": {
        "difficulty": "easy",
        "description": "Classify the email category correctly.",
        # Round 1 reward weights (single-step)
        "reward_weights": {"category": 1.0, "priority": 0.1, "escalation": 0.0},
        # Round 2 multi-turn weights (ignored in easy mode)
        "multi_turn_weights": {
            "quality": 1.0,
            "sla": 0.0,
            "policy": 0.0,
            "oversight": 0.0,
            "efficiency": 0.0,
        },
        "max_episode_steps": 1,
    },
    "medium": {
        "difficulty": "medium",
        "description": "Classify category and set the right priority bucket.",
        "reward_weights": {"category": 0.8, "priority": 0.3, "escalation": 0.1},
        "multi_turn_weights": {
            "quality": 0.40,
            "sla": 0.20,
            "policy": 0.15,
            "oversight": 0.15,
            "efficiency": 0.10,
        },
        "max_episode_steps": 20,
    },
    "hard": {
        "difficulty": "hard",
        "description": "Full triage: category, priority, and safe escalation behavior.",
        "reward_weights": {"category": 0.6, "priority": 0.3, "escalation": 0.3},
        "multi_turn_weights": {
            "quality": 0.30,
            "sla": 0.20,
            "policy": 0.20,
            "oversight": 0.15,
            "efficiency": 0.15,
        },
        "max_episode_steps": 40,
    },
    "adversarial": {
        "difficulty": "adversarial",
        "description": (
            "Adversarial triage: contradictory specialist outputs, "
            "heavy schema drift, cascading SLA pressure."
        ),
        "reward_weights": {"category": 0.6, "priority": 0.3, "escalation": 0.3},
        "multi_turn_weights": {
            "quality": 0.25,
            "sla": 0.20,
            "policy": 0.20,
            "oversight": 0.20,
            "efficiency": 0.15,
        },
        "max_episode_steps": 60,
    },
}


class EmailTriageEnvironment(
    Environment[EmailTriageAction, EmailTriageObservation, EmailTriageState]
):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, difficulty: Difficulty = "medium") -> None:
        super().__init__()
        self._difficulty: Difficulty = difficulty if difficulty in TASK_CONFIG else "medium"
        self._task_id: TaskId = self._difficulty
        self._current_email: Dict[str, Any] = {}
        self._emails: List[Dict[str, Any]] = self._load_email_dataset()

        # Multi-turn state
        self._queue: List[Dict[str, Any]] = []
        self._queue_index: int = 0
        self._sla_deadlines: List[int] = []
        self._specialists: Optional[SpecialistPool] = None
        self._drift_engine: Optional[DriftEngine] = None
        self._specialist_reports: Dict[str, Dict[str, Any]] = {}
        self._event_log: List[Dict[str, Any]] = []
        self._total_specialist_errors: int = 0
        self._last_drift_step: Optional[int] = None

        # Anti-reward-hacking state
        self._action_history: List[tuple] = []
        self._repetition_penalties: int = 0
        self._max_episode_steps: int = TASK_CONFIG[self._difficulty].get(
            "max_episode_steps", 50
        )

        self._state = EmailTriageState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            total_reward=0.0,
            difficulty=self._difficulty,
            current_task=self._task_id,
        )

    @staticmethod
    def task_metadata() -> Dict[str, Dict[str, Any]]:
        return TASK_CONFIG

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_email_dataset(self) -> List[Dict[str, Any]]:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "email_triage_dataset.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    def _sample_email(self) -> Dict[str, Any]:
        diff_key = self._difficulty if self._difficulty != "adversarial" else "hard"
        candidates = [e for e in self._emails if e.get("difficulty") == diff_key]
        if not candidates:
            candidates = self._emails
        return random.choice(candidates)

    # ------------------------------------------------------------------
    # Task / difficulty resolution
    # ------------------------------------------------------------------

    def _resolve_task(self, **kwargs: Any) -> TaskId:
        requested_task = kwargs.get("task_id")
        requested_difficulty = kwargs.get("difficulty")

        if requested_task in TASK_CONFIG:
            return requested_task
        if requested_difficulty in TASK_CONFIG:
            return requested_difficulty
        return self._task_id

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EmailTriageObservation:
        if seed is not None:
            random.seed(seed)

        self._task_id = self._resolve_task(**kwargs)
        self._difficulty = TASK_CONFIG[self._task_id]["difficulty"]

        # Generate scenario
        actual_seed = seed if seed is not None else random.randint(0, 2**31)
        scenario = generate_scenario(self._emails, self._difficulty, actual_seed)

        self._queue = [slot.email for slot in scenario.tickets]
        self._sla_deadlines = [slot.sla_deadline_step for slot in scenario.tickets]
        self._queue_index = 0

        # Init specialists and drift engine
        self._specialists = SpecialistPool(
            base_accuracy=scenario.specialist_accuracy, seed=actual_seed
        )
        self._drift_engine = DriftEngine(self._difficulty, seed=actual_seed)

        # Reset tracking
        self._specialist_reports = {}
        self._event_log = []
        self._total_specialist_errors = 0
        self._last_drift_step = None
        self._action_history = []
        self._repetition_penalties = 0
        self._max_episode_steps = TASK_CONFIG[self._task_id].get(
            "max_episode_steps", 50
        )

        # Set current email
        self._current_email = self._queue[0] if self._queue else self._sample_email()

        # Pre-compute specialist reports for first ticket
        first_reports = self._specialists.simulate_all(self._current_email)
        self._specialist_reports[self._current_email.get("id", "0")] = first_reports
        self._total_specialist_errors += self._count_specialist_errors(first_reports)

        # Init state
        queue_size = len(self._queue)
        self._state = EmailTriageState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            total_reward=0.0,
            difficulty=self._difficulty,
            current_task=self._task_id,
            queue_size=queue_size,
            tickets_resolved=0,
            tickets_remaining=queue_size,
            sla_breaches=0,
            policy_violations=0,
            oversight_catches=0,
            drift_count=0,
        )

        info: Dict[str, Any] = {
            "reason": "reset",
            "task_id": self._task_id,
            "task_description": TASK_CONFIG[self._task_id]["description"],
            "queue_size": queue_size,
            "queue_position": 1,
            "tickets_resolved": 0,
            "tickets_remaining": queue_size,
        }

        # Add specialist reports and policies for non-easy modes
        if self._difficulty != "easy":
            info["specialist_reports"] = first_reports
            info["active_policies"] = self._drift_engine.active_policies
            info["policy_drift_occurred"] = False
            info["sla_deadline_step"] = self._sla_deadlines[0] if self._sla_deadlines else 1

        return self._make_observation(reward=0.0, done=False, info=info)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: EmailTriageAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> EmailTriageObservation:
        self._state.step_count += 1

        if not self._current_email:
            self._current_email = self._sample_email()

        # ── Anti-hack: validate and sanitize action ───────────────────
        action = self._validate_action(action)

        # ── Anti-hack: check step timeout ─────────────────────────────
        if self._state.step_count > self._max_episode_steps:
            return self._make_observation(
                reward=-1.0,
                done=True,
                info={
                    "task_id": self._task_id,
                    "timeout": True,
                    "reason": f"Episode terminated: exceeded {self._max_episode_steps} step limit",
                },
            )

        # ── Easy mode: single-step, backward-compatible ───────────────
        if self._difficulty == "easy":
            return self._step_single(action)

        # ── Multi-turn mode ───────────────────────────────────────────
        return self._step_multi_turn(action)

    def _step_single(self, action: EmailTriageAction) -> EmailTriageObservation:
        """Original Round 1 single-step logic — unchanged behaviour."""
        reward = self._compute_reward_v1(action, self._current_email)
        self._state.total_reward += reward

        task_score = task_grader(self._task_id, action, self._current_email)
        info = {
            "task_id": self._task_id,
            "task_description": TASK_CONFIG[self._task_id]["description"],
            "task_score": task_score,
            "true_category": self._current_email["true_category"],
            "true_priority": self._current_email["true_priority"],
            "true_needs_escalation": self._current_email["needs_escalation"],
            "category_score": category_grader(action, self._current_email),
            "priority_score": priority_grader(action, self._current_email),
            "escalation_score": escalation_grader(action, self._current_email),
        }

        return self._make_observation(reward=reward, done=True, info=info)

    def _step_multi_turn(self, action: EmailTriageAction) -> EmailTriageObservation:
        """Round 2 multi-turn queue processing."""
        current_email = self._current_email
        email_id = current_email.get("id", str(self._queue_index))

        # ── 0. Anti-hack: track action history for repetition ────────
        action_sig = (action.category, action.priority, action.should_escalate)
        self._action_history.append(action_sig)
        repetition_penalty = 0.0
        if len(self._action_history) >= 3:
            last_3 = self._action_history[-3:]
            if all(a == last_3[0] for a in last_3):
                repetition_penalty = -0.3
                self._repetition_penalties += 1

        # ── 1. Check schema drift ────────────────────────────────────
        drift_info: Optional[Dict[str, Any]] = None
        if self._drift_engine is not None:
            drift_info = self._drift_engine.check_for_drift(
                self._queue_index, len(self._queue)
            )
            if drift_info is not None:
                self._last_drift_step = self._state.step_count
                self._state.drift_count = self._drift_engine.drift_count
                # Degrade specialists after drift
                if self._specialists is not None:
                    self._specialists.degrade(0.10)

        # ── 2. Check policy compliance ───────────────────────────────
        compliant = True
        violations: List[str] = []
        if self._drift_engine is not None:
            compliant, violations = self._drift_engine.check_compliance(
                action, current_email
            )
        if not compliant:
            self._state.policy_violations += len(violations)

        # ── 3. Check oversight (did coordinator correct specialist?) ─
        reports = self._specialist_reports.get(email_id, {})
        triage_report = reports.get("triage", {})
        specialist_category = triage_report.get("category")
        specialist_correct = triage_report.get("correct", True)
        true_category = current_email.get("true_category", "other")
        agent_category = action.category

        if not specialist_correct and agent_category == true_category:
            self._state.oversight_catches += 1

        # ── 4. Check SLA ─────────────────────────────────────────────
        deadline = (
            self._sla_deadlines[self._queue_index]
            if self._queue_index < len(self._sla_deadlines)
            else self._state.step_count
        )
        if self._state.step_count > deadline:
            self._state.sla_breaches += 1

        # ── 5. Compute reward ────────────────────────────────────────
        steps_since_drift: Optional[int] = None
        if self._last_drift_step is not None:
            steps_since_drift = self._state.step_count - self._last_drift_step

        episode_stats = {
            "tickets_resolved": self._state.tickets_resolved + 1,
            "sla_breaches": self._state.sla_breaches,
            "total_decisions": self._state.step_count,
            "policy_violations": self._state.policy_violations,
            "oversight_catches": self._state.oversight_catches,
            "total_specialist_errors": self._total_specialist_errors,
            "total_steps": self._state.step_count,
            "steps_since_last_drift": steps_since_drift,
            "compliant_after_drift": compliant,
        }

        weights = TASK_CONFIG[self._task_id]["multi_turn_weights"]
        reward, reward_components = compute_multi_turn_reward(
            action, current_email, self._task_id, weights, episode_stats
        )
        # Apply anti-hack repetition penalty
        reward += repetition_penalty
        # Clamp per-step reward to [-2.0, 2.0] to prevent unbounded accumulation
        reward = max(-2.0, min(2.0, reward))
        self._state.total_reward += reward

        # ── 6. Advance queue ─────────────────────────────────────────
        self._queue_index += 1
        self._state.tickets_resolved += 1
        self._state.tickets_remaining = max(0, len(self._queue) - self._queue_index)

        # Log event
        self._event_log.append(
            {
                "step": self._state.step_count,
                "ticket": email_id,
                "action_category": action.category,
                "action_priority": action.priority,
                "action_escalate": action.should_escalate,
                "reward": round(reward, 4),
                "compliant": compliant,
            }
        )

        # ── 7. Determine done ────────────────────────────────────────
        done = self._queue_index >= len(self._queue)

        # ── 8. Prepare next ticket observation ───────────────────────
        if not done:
            self._current_email = self._queue[self._queue_index]
            # Pre-compute specialist reports for next ticket
            next_id = self._current_email.get("id", str(self._queue_index))
            if self._specialists is not None:
                next_reports = self._specialists.simulate_all(self._current_email)
                self._specialist_reports[next_id] = next_reports
                self._total_specialist_errors += self._count_specialist_errors(next_reports)

        # ── 9. Build info ────────────────────────────────────────────
        next_email_id = self._current_email.get("id", str(self._queue_index))
        next_reports_for_obs = self._specialist_reports.get(next_email_id, {})

        info: Dict[str, Any] = {
            "task_id": self._task_id,
            "task_description": TASK_CONFIG[self._task_id]["description"],
            # Base grading scores (backward compatible)
            "task_score": task_grader(self._task_id, action, current_email),
            "true_category": current_email["true_category"],
            "true_priority": current_email["true_priority"],
            "true_needs_escalation": current_email["needs_escalation"],
            "category_score": category_grader(action, current_email),
            "priority_score": priority_grader(action, current_email),
            "escalation_score": escalation_grader(action, current_email),
            # Multi-turn info
            "queue_size": len(self._queue),
            "queue_position": self._queue_index + 1,
            "tickets_resolved": self._state.tickets_resolved,
            "tickets_remaining": self._state.tickets_remaining,
            "sla_breaches": self._state.sla_breaches,
            "policy_violations": self._state.policy_violations,
            "oversight_catches": self._state.oversight_catches,
            "reward_components": reward_components,
            "specialist_reports": next_reports_for_obs,
            "event_log": self._event_log[-5:],  # last 5 events
        }

        # Policy and drift info
        if self._drift_engine is not None:
            info["active_policies"] = self._drift_engine.active_policies
            info["policy_drift_occurred"] = drift_info is not None
            if drift_info is not None:
                info["drift_description"] = drift_info.get("description", "")
            if not done and self._queue_index < len(self._sla_deadlines):
                info["sla_deadline_step"] = self._sla_deadlines[self._queue_index]

        # Compliance info for current action
        info["action_compliant"] = compliant
        if violations:
            info["violations"] = violations

        return self._make_observation(reward=reward, done=done, info=info)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _make_observation(
        self, reward: float, done: bool, info: Dict[str, Any]
    ) -> EmailTriageObservation:
        body = self._current_email.get("body", "")
        snippet = body[:280]

        return EmailTriageObservation(
            email_id=self._current_email.get("id", ""),
            subject=self._current_email.get("subject", ""),
            body_snippet=snippet,
            sender=self._current_email.get("sender", ""),
            sender_domain=self._current_email.get("sender_domain", ""),
            is_internal=self._current_email.get("is_internal", False),
            task_id=self._task_id,
            reward=reward,
            done=done,
            metadata={
                "difficulty": self._current_email.get("difficulty", self._difficulty),
                "task_id": self._task_id,
                "queue_position": self._queue_index + 1,
                "queue_size": len(self._queue),
                "drift_count": self._state.drift_count,
            },
            info=info,
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> EmailTriageState:
        return self._state

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward_v1(
        self, action: EmailTriageAction, email: Dict[str, Any]
    ) -> float:
        """Original Round 1 reward computation — identical to v1 behaviour."""
        cat_score = category_grader(action, email)
        pri_score = priority_grader(action, email)
        esc_score = escalation_grader(action, email)

        weights = TASK_CONFIG[self._task_id]["reward_weights"]
        reward = 0.0
        reward += weights["category"] * cat_score
        reward += weights["priority"] * pri_score
        reward += weights["escalation"] * esc_score

        # Add task-level deterministic score to shape progress.
        reward += 0.25 * task_grader(self._task_id, action, email)

        if email["true_category"] == "spam" and action.should_escalate:
            reward -= 0.5
        if email["true_category"] == "urgent" and not action.should_escalate:
            reward -= 0.5

        return reward

    # ------------------------------------------------------------------
    # Anti-reward-hacking
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_action(action: EmailTriageAction) -> EmailTriageAction:
        """Sanitize action inputs to prevent reward hacking.

        - Clamps priority to valid range [1, 5]
        - Rejects invalid categories (defaults to 'other')
        - Ensures boolean escalation
        """
        # Clamp priority to valid range
        action.priority = max(1, min(5, int(action.priority)))

        # Validate category against locked set
        if action.category not in _VALID_CATEGORIES:
            action.category = "other"

        # Ensure escalation is boolean (not a truthy string)
        action.should_escalate = bool(action.should_escalate)

        return action

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_specialist_errors(reports: Dict[str, Dict[str, Any]]) -> int:
        """Count how many specialist reports are incorrect."""
        return sum(
            1
            for r in reports.values()
            if isinstance(r, dict) and not r.get("correct", True)
        )
