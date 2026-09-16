from __future__ import annotations

from typing import Dict

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
except ImportError:
    from core.client_types import StepResult
    from core.env_client import EnvClient

try:
    from .models import EmailTriageAction, EmailTriageObservation, EmailTriageState
except ImportError:
    from models import EmailTriageAction, EmailTriageObservation, EmailTriageState


class EmailTriageEnv(EnvClient[EmailTriageAction, EmailTriageObservation, EmailTriageState]):
    def _step_payload(self, action: EmailTriageAction) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "category": action.category,
            "priority": action.priority,
            "should_escalate": action.should_escalate,
        }
        # Include Round 2 optional fields when present
        if action.rationale is not None:
            payload["rationale"] = action.rationale
        return payload

    def _parse_result(self, payload: dict) -> StepResult[EmailTriageObservation]:
        obs_p = payload["observation"]

        obs = EmailTriageObservation(
            email_id=obs_p["email_id"],
            subject=obs_p["subject"],
            body_snippet=obs_p["body_snippet"],
            sender=obs_p["sender"],
            sender_domain=obs_p["sender_domain"],
            is_internal=obs_p["is_internal"],
            task_id=obs_p["task_id"],
            reward=obs_p["reward"],
            done=obs_p["done"],
            metadata=obs_p.get("metadata", {}),
            info=obs_p.get("info"),
        )

        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> EmailTriageState:
        return EmailTriageState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            total_reward=payload.get("total_reward", 0.0),
            difficulty=payload.get("difficulty", "medium"),
            current_task=payload.get("current_task", "medium"),
            # Round 2 fields (with defaults for backward compat)
            queue_size=payload.get("queue_size", 0),
            tickets_resolved=payload.get("tickets_resolved", 0),
            tickets_remaining=payload.get("tickets_remaining", 0),
            sla_breaches=payload.get("sla_breaches", 0),
            policy_violations=payload.get("policy_violations", 0),
            oversight_catches=payload.get("oversight_catches", 0),
            drift_count=payload.get("drift_count", 0),
        )
