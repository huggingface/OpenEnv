#!/usr/bin/env python3
"""Evaluation script for Oversight Inbox Arena.

Runs a fixed set of deterministic scenarios and produces a comparison
table of metrics. Use this to generate the reward curves and metric
tables for your hackathon demo.

Usage:
    python eval_benchmark.py                    # All difficulties
    python eval_benchmark.py --difficulty hard   # Single difficulty
    python eval_benchmark.py --output results.json  # Save JSON
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, os.path.join(ROOT_DIR, "envs"))

from email_triage_env.server.email_triage_environment import EmailTriageEnvironment
from email_triage_env.models import EmailTriageAction


# ---------------------------------------------------------------------------
# Evaluation seeds (fixed, held-out)
# ---------------------------------------------------------------------------

EVAL_SEEDS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
DIFFICULTIES = ["easy", "medium", "hard", "adversarial"]


# ---------------------------------------------------------------------------
# Agent strategies
# ---------------------------------------------------------------------------

def random_agent(obs: Any) -> EmailTriageAction:
    """Baseline: random triage decisions."""
    import random
    cats = ["billing", "support", "spam", "urgent", "marketing", "other"]
    return EmailTriageAction(
        category=random.choice(cats),
        priority=random.randint(1, 5),
        should_escalate=random.choice([True, False]),
    )


def heuristic_agent(obs: Any) -> EmailTriageAction:
    """Rule-based heuristic using specialist reports."""
    info = obs.info or {}
    specialist = info.get("specialist_reports", {})
    triage = specialist.get("triage", {})
    escalation = specialist.get("escalation", {})
    compliance = specialist.get("compliance", {})

    # Trust specialist triage suggestion
    cat = triage.get("category", "other")
    pri = triage.get("priority", 3)

    # Escalation logic
    should_esc = escalation.get("recommended", False)

    # Override: never escalate spam
    if cat == "spam":
        should_esc = False

    # Override: always escalate urgent
    if cat == "urgent":
        should_esc = True

    # Override: if compliance flagged and priority high, escalate
    if compliance.get("flagged", False) and pri >= 4:
        should_esc = True

    return EmailTriageAction(
        category=cat,
        priority=max(1, min(5, pri)),
        should_escalate=should_esc,
    )


def specialist_trust_agent(obs: Any) -> EmailTriageAction:
    """Blindly trusts specialist triage without any coordination."""
    info = obs.info or {}
    triage = info.get("specialist_reports", {}).get("triage", {})

    return EmailTriageAction(
        category=triage.get("category", "other"),
        priority=max(1, min(5, triage.get("priority", 3))),
        should_escalate=info.get("specialist_reports", {}).get(
            "escalation", {}
        ).get("recommended", False),
    )


AGENTS = {
    "random": random_agent,
    "heuristic": heuristic_agent,
    "specialist_trust": specialist_trust_agent,
}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    agent_name: str,
    agent_fn,
    difficulty: str,
    seeds: List[int],
) -> Dict[str, Any]:
    """Run evaluation episodes and collect metrics."""
    results = []

    for seed in seeds:
        env = EmailTriageEnvironment(difficulty=difficulty)
        obs = env.reset(seed=seed)

        while True:
            action = agent_fn(obs)
            obs = env.step(action)
            if obs.done:
                break

        s = env.state
        episode = {
            "seed": seed,
            "total_reward": round(s.total_reward, 4),
            "tickets_resolved": s.tickets_resolved,
            "queue_size": s.queue_size,
            "resolution_rate": round(s.tickets_resolved / max(1, s.queue_size), 4),
            "sla_breaches": s.sla_breaches,
            "sla_breach_rate": round(s.sla_breaches / max(1, s.tickets_resolved), 4),
            "policy_violations": s.policy_violations,
            "violation_rate": round(s.policy_violations / max(1, s.step_count), 4),
            "oversight_catches": s.oversight_catches,
            "drift_count": s.drift_count,
            "steps": s.step_count,
        }
        results.append(episode)

    # Aggregate
    n = len(results)
    agg = {
        "agent": agent_name,
        "difficulty": difficulty,
        "episodes": n,
        "mean_reward": round(sum(r["total_reward"] for r in results) / n, 4),
        "mean_resolution_rate": round(sum(r["resolution_rate"] for r in results) / n, 4),
        "mean_sla_breach_rate": round(sum(r["sla_breach_rate"] for r in results) / n, 4),
        "mean_violation_rate": round(sum(r["violation_rate"] for r in results) / n, 4),
        "mean_oversight_catches": round(sum(r["oversight_catches"] for r in results) / n, 4),
        "total_drift_events": sum(r["drift_count"] for r in results),
        "episodes_detail": results,
    }
    return agg


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_table(all_results: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table."""
    header = f"{'Agent':<20} {'Difficulty':<14} {'Avg Reward':>11} {'Resolution':>11} {'SLA Breach':>11} {'Violations':>11} {'Oversight':>10}"
    print("\n" + "=" * len(header))
    print("  OVERSIGHT INBOX ARENA — EVALUATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in all_results:
        print(
            f"{r['agent']:<20} "
            f"{r['difficulty']:<14} "
            f"{r['mean_reward']:>11.3f} "
            f"{r['mean_resolution_rate']:>10.1%} "
            f"{r['mean_sla_breach_rate']:>10.1%} "
            f"{r['mean_violation_rate']:>10.1%} "
            f"{r['mean_oversight_catches']:>10.1f}"
        )

    print("=" * len(header))


def print_reward_chart(all_results: List[Dict[str, Any]]) -> None:
    """Print a simple ASCII reward chart by difficulty."""
    print("\n📊 Reward by Agent × Difficulty\n")
    max_reward = max(abs(r["mean_reward"]) for r in all_results) or 1.0

    for r in all_results:
        label = f"{r['agent']:>18} | {r['difficulty']:<12}"
        bar_len = int(20 * max(0, r["mean_reward"]) / max_reward) if max_reward > 0 else 0
        bar = "█" * bar_len
        print(f"  {label} {bar} {r['mean_reward']:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Oversight Inbox Arena")
    parser.add_argument("--difficulty", choices=DIFFICULTIES, help="Single difficulty to test")
    parser.add_argument("--agent", choices=list(AGENTS.keys()), help="Single agent to test")
    parser.add_argument("--seeds", type=int, default=10, help="Number of eval seeds")
    parser.add_argument("--output", type=str, help="Save results JSON to file")
    args = parser.parse_args()

    difficulties = [args.difficulty] if args.difficulty else DIFFICULTIES
    agents = {args.agent: AGENTS[args.agent]} if args.agent else AGENTS
    seeds = EVAL_SEEDS[:args.seeds]

    print(f"🔬 Evaluating {len(agents)} agent(s) × {len(difficulties)} difficulty tier(s) × {len(seeds)} seeds")

    all_results: List[Dict[str, Any]] = []

    for agent_name, agent_fn in agents.items():
        for diff in difficulties:
            print(f"  Running {agent_name} on {diff}...", end=" ", flush=True)
            result = evaluate(agent_name, agent_fn, diff, seeds)
            all_results.append(result)
            print(f"avg_reward={result['mean_reward']:.3f}")

    print_table(all_results)
    print_reward_chart(all_results)

    if args.output:
        # Remove episode details for cleaner output
        clean = [{k: v for k, v in r.items() if k != "episodes_detail"} for r in all_results]
        with open(args.output, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
