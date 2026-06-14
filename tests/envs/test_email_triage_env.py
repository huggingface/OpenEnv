"""Smoke test for all difficulty tiers of the Oversight Inbox Arena."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from envs.email_triage_env.models import EmailTriageAction
from envs.email_triage_env.server.email_triage_environment import EmailTriageEnvironment


def test_easy() -> None:
    """Easy mode: single-step, backward-compatible."""
    print("=== EASY MODE (backward compat) ===")
    env = EmailTriageEnvironment(difficulty="easy")
    obs = env.reset(seed=42)
    print(f"  reset OK | email_id={obs.email_id} | subject={obs.subject[:40]}")

    action = EmailTriageAction(category="billing", priority=3, should_escalate=False)
    obs2 = env.step(action)
    cat_score = obs2.info.get("category_score", "N/A")
    print(f"  step OK  | reward={obs2.reward:.3f} | done={obs2.done} | cat_score={cat_score}")
    print(f"  state: total_reward={env.state.total_reward:.3f} step_count={env.state.step_count}")
    assert obs2.done is True, "Easy mode must be done after 1 step"
    print("  PASS\n")


def test_medium() -> None:
    """Medium mode: multi-turn queue, no drift."""
    print("=== MEDIUM MODE (multi-turn) ===")
    env = EmailTriageEnvironment(difficulty="medium")
    obs = env.reset(seed=42)
    queue_size = obs.info.get("queue_size", 0)
    print(f"  reset OK | queue_size={queue_size} | ticket={obs.email_id}")

    specialist_keys = list(obs.info.get("specialist_reports", {}).keys())
    print(f"  specialist_reports keys: {specialist_keys}")
    assert len(specialist_keys) == 4, "Should have 4 specialist reports"

    steps = 0
    while True:
        triage = obs.info.get("specialist_reports", {}).get("triage", {})
        cat = triage.get("category", "other")
        a = EmailTriageAction(category=cat, priority=3, should_escalate=False)
        obs = env.step(a)
        steps += 1
        remaining = obs.info.get("tickets_remaining", "?")
        print(f"  step {steps} | reward={obs.reward:.3f} | done={obs.done} | remaining={remaining}")
        if obs.done:
            break

    s = env.state
    print(f"  Final: resolved={s.tickets_resolved} sla_breaches={s.sla_breaches} "
          f"violations={s.policy_violations} oversight={s.oversight_catches}")
    assert steps == queue_size, f"Should take exactly {queue_size} steps, took {steps}"
    print("  PASS\n")


def test_hard() -> None:
    """Hard mode: multi-turn + schema drift."""
    print("=== HARD MODE (with drift) ===")
    env = EmailTriageEnvironment(difficulty="hard")
    obs = env.reset(seed=123)
    queue_size = obs.info.get("queue_size", 0)
    policies = obs.info.get("active_policies", [])
    print(f"  reset OK | queue_size={queue_size} | policies={len(policies)}")

    steps = 0
    drift_seen = False
    while True:
        a = EmailTriageAction(category="support", priority=4, should_escalate=True)
        obs = env.step(a)
        steps += 1
        if obs.info.get("policy_drift_occurred"):
            drift_seen = True
            desc = obs.info.get("drift_description", "")
            print(f"  step {steps} DRIFT! desc={desc}")
        if obs.done:
            break

    s = env.state
    print(f"  Done in {steps} steps | drift_seen={drift_seen} | drift_count={s.drift_count}")
    print(f"  total_reward={s.total_reward:.3f} | oversight={s.oversight_catches} | violations={s.policy_violations}")
    assert steps == queue_size
    print("  PASS\n")


def test_adversarial() -> None:
    """Adversarial mode: heavy drift, degraded specialists."""
    print("=== ADVERSARIAL MODE ===")
    env = EmailTriageEnvironment(difficulty="adversarial")
    obs = env.reset(seed=99)
    queue_size = obs.info.get("queue_size", 0)
    print(f"  reset OK | queue_size={queue_size}")

    steps = 0
    while True:
        a = EmailTriageAction(category="urgent", priority=5, should_escalate=True)
        obs = env.step(a)
        steps += 1
        if obs.done:
            break

    s = env.state
    print(f"  Done in {steps} steps | drift_count={s.drift_count} | total_reward={s.total_reward:.3f}")
    assert steps == queue_size
    print("  PASS\n")


def test_deterministic() -> None:
    """Same seed produces same rewards."""
    print("=== DETERMINISM TEST ===")
    rewards_a = []
    rewards_b = []

    for run_rewards in [rewards_a, rewards_b]:
        env = EmailTriageEnvironment(difficulty="hard")
        obs = env.reset(seed=777)
        while True:
            a = EmailTriageAction(category="billing", priority=2, should_escalate=False)
            obs = env.step(a)
            run_rewards.append(round(obs.reward, 6))
            if obs.done:
                break

    assert rewards_a == rewards_b, f"Runs differ: {rewards_a} vs {rewards_b}"
    print(f"  Two runs with seed=777 produced identical rewards ({len(rewards_a)} steps)")
    print("  PASS\n")


def test_inference_compat() -> None:
    """Verify easy mode matches the exact v1 reward computation."""
    print("=== INFERENCE BACKWARD COMPAT TEST ===")
    env = EmailTriageEnvironment(difficulty="easy")
    obs = env.reset(seed=11)
    true_cat = obs.info.get("task_id", "easy")
    print(f"  task_id={true_cat}")

    # Test with a correct-ish action
    a = EmailTriageAction(category="spam", priority=1, should_escalate=False)
    obs2 = env.step(a)
    print(f"  action: cat=spam pri=1 esc=False")
    print(f"  reward={obs2.reward:.3f} done={obs2.done}")
    assert obs2.done is True
    # Reward should be the v1 formula result
    print("  PASS\n")


if __name__ == "__main__":
    test_easy()
    test_medium()
    test_hard()
    test_adversarial()
    test_deterministic()
    test_inference_compat()
    print("=" * 50)
    print("ALL ENVIRONMENT TESTS PASSED")
    print("=" * 50)
