# Oversight Inbox Arena — Complete Architecture & Pitch Document

**Project:** Oversight Inbox Arena  
**Repo:** [Rhushya/OpenEnv](https://github.com/Rhushya/OpenEnv)  
**Author:** Rhushya KC  
**Event:** Meta PyTorch OpenEnv Hackathon Grand Finale 2026  
**Tech Stack:** Python · TRL · GRPO · Qwen2-0.5B · FastAPI · Gradio · HuggingFace Spaces

---

## Executive Summary

Oversight Inbox Arena is a **multi-agent reinforcement learning environment** built on the OpenEnv framework. It trains a single LLM "coordinator" agent to supervise four specialist AI agents handling a realistic enterprise email inbox — under schema drift, partial observability, and time pressure. The project addresses a fundamental gap in existing RL environments: nearly all prior work trains single-agent, single-step models, while real-world AI deployment requires coordination, error-correction, and policy adaptation across multiple agents.

The environment exposes a 5-function reward system, a 4-tier difficulty curriculum, anti-reward-hacking protections, and a polished Gradio demo UI — all compliant with the OpenEnv Gymnasium API.

---

## 1. What Is This Project?

### The Plain-English Explanation

Imagine you manage a support team of four AI assistants:
- One **classifies** tickets into billing, spam, urgent, support, etc.
- One **decides** whether to escalate to a human
- One **checks** if actions comply with company policy
- One **drafts** response templates

They each make mistakes. They each have biases. And mid-shift, the company updates its escalation policy — without telling all of them.

**Your job (as the coordinator) is:**
- Catch when the triage bot calls a "server outage" email "spam"
- Override the escalation agent when it over-escalates routine billing queries
- Adapt within 2 steps when a policy suddenly changes
- Resolve every ticket before the SLA deadline

That coordination skill is exactly what this environment trains an LLM to perform — via reinforcement learning using GRPO.

### Why This Exists

Most LLM RL environments are single-agent, single-step: one input → one output → one score. The real world requires:

| Problem | Industry Reality | Previous RL Envs |
|---------|-----------------|-----------------|
| Multiple agents | 4 specialists with different biases | 1 agent acts alone |
| Multi-step decisions | 5–15 tickets per episode | Single step only |
| Changing rules | Policy drift mid-episode | Fixed rules forever |
| Oversight | Catch specialist errors | No oversight concept |
| Partial info | See only summaries | Full state visible |

This project fills every one of those gaps in the OpenEnv ecosystem.

---

## 2. Complete Architecture

### Repository Layout

```
Rhushya/OpenEnv (mono-repo)
├── README.md                          # Main project docs (17KB)
├── pre_training.json                  # Dataset seed
├── pyproject.toml                     # Package config
├── EmailTriage_GRPO_Train.ipynb       # Root training notebook
│
└── envs/email_triage_env/
    ├── models.py                      # Pydantic contracts (Action, Observation, State)
    ├── client.py                      # HTTP/WebSocket client (70 lines)
    ├── openenv.yaml                   # OpenEnv environment manifest
    ├── inference.py                   # Inference runner (12KB)
    ├── train_grpo.py                  # GRPO training script (17KB, 320 lines)
    ├── eval_benchmark.py              # 3-agent benchmark (8.7KB, 250 lines)
    ├── Rhushya_OpenEnv_EmailTriage_Training.ipynb  # Primary training notebook
    ├── colab_t4_training.ipynb        # T4-optimized Colab notebook
    ├── FINAL_SHOWCASE_README.md       # Demo-day guide
    │
    └── server/
        ├── app.py                     # FastAPI app (47 lines)
        ├── email_triage_environment.py # Core engine (26KB, 610 lines)
        ├── graders.py                 # 11 deterministic reward graders (8.8KB, 220 lines)
        ├── scenario_generator.py      # Queue builder (3.1KB, 100 lines)
        ├── stakeholders.py            # 4 specialist simulations (5.9KB, 160 lines)
        ├── schema_drift.py            # Policy mutation engine (10KB, 250 lines)
        ├── email_triage_dataset.json  # 120 labeled emails (57KB)
        └── ui.py                      # Gradio demo UI (16KB)
```

**Total codebase: ~3,500 lines of tested Python.**

---

### The Three Layers

#### Layer 1 — Data Layer

- **`email_triage_dataset.json`** (57KB): 120 labeled synthetic emails. Each record contains `id`, `subject`, `body`, `sender`, `sender_domain`, `is_internal`, `true_category`, `true_priority`, `needs_escalation`, `difficulty`. This is the ground truth that all reward graders compare against.
- **`scenario_generator.py`**: Builds deterministic email queues from integer seeds. Given `seed=42, difficulty="hard"`, always produces the same 8-ticket queue. Supports adversarial mixing — includes deliberately confusing tickets that exploit known specialist biases.
- **`pre_training.json`**: Seed data for warm-starting the GRPO dataset builder.

#### Layer 2 — Environment Engine (Server)

This is the core runtime. All five components run inside the FastAPI server and are invoked on every `step()` call.

**`email_triage_environment.py`** (Core State Machine, 610 lines):
- Implements `reset(difficulty, seed)` → loads queue, activates policies, resets state
- Implements `step(action)` → validates action → queries specialists → runs all graders → applies drift → returns observation
- Maintains episode state: `tickets_resolved`, `sla_breaches`, `oversight_catches`, `drift_count`, `policy_violations`
- Contains 5 anti-reward-hacking protections (see Section 4)

**`stakeholders.py`** (4 Specialist Agents, 160 lines):

| Specialist | Function | Accuracy | Known Bias |
|-----------|----------|---------|-----------|
| Triage Agent | Predicts category + priority | 75–95% | Under-prioritizes billing |
| Escalation Agent | Recommend escalate/not | 80–95% | Over-escalates under uncertainty |
| Compliance Agent | Flag policy violations | 85–98% | High false-positive rate |
| Responder Agent | Suggest reply template | 70–90% | Too formulaic, misses nuance |

After each schema drift event, all specialist accuracies degrade by 10%, forcing the coordinator to rely on its own judgment.

**`schema_drift.py`** (Policy Mutation Engine, 250 lines):

At 30–60% through a queue, the drift engine randomly selects and applies a policy mutation:

| Drift Type | Example |
|-----------|---------|
| Escalation threshold lowered | "Escalate if ≥ 4" → "Escalate if ≥ 3" |
| SLA tightened | 3 steps/ticket → 2 steps/ticket |
| Spam policy relaxed | Internal spam can now be escalated |
| New compliance rule added | "All urgent tickets require compliance review" |
| Priority scale changed | Bucket boundaries shift |

The coordinator sees `policy_drift_occurred: true` and `drift_description` in its next observation. If it adapts within 2 steps, it earns a drift adaptation bonus reward.

**`graders.py`** (11 Deterministic Reward Graders, 220 lines):

Every reward is deterministic, verifiable, and computable by anyone given the action and ground truth. No LLM judges, no neural reward models.

| # | Grader | Measures | Range |
|---|--------|---------|-------|
| R1 | `reward_quality` | Category + priority + escalation accuracy vs. ground truth | [0, 1] |
| R2 | `reward_sla` | Tickets resolved before deadline steps | [0, 1] |
| R3 | `reward_compliance` | Actions follow currently active policies | [0, 1] |
| R4 | `reward_oversight` | Coordinator caught and overrode specialist mistakes | [0, 1] |
| R5 | `reward_no_hacking` | No repeated actions, no timeout abuse | [-2, 0] |

Hard-coded safety penalties (applied on top of reward functions):
- Escalating spam: **-0.5** (wastes human reviewer time)
- Ignoring urgent incidents: **-0.5** (safety-critical failure)

**`app.py`** (FastAPI Application, 47 lines):

Exposes three HTTP endpoints used by the training loop and Gradio UI:
- `POST /reset` — start a new episode
- `POST /step` — submit one coordinator action
- `GET /state` — read current episode state

#### Layer 3 — Training & Demo Layer

**`train_grpo.py`** (GRPO Training, 320 lines):

The training script integrates with TRL's `GRPOTrainer` using the 5-function reward interface:

```python
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B",
    reward_funcs=[
        reward_quality,     # R1
        reward_oversight,   # R2
        reward_compliance,  # R3
        reward_sla,         # R4
        reward_no_hacking,  # R5
    ],
    train_dataset=dataset,
    environment_factory=OversightInboxEnv,
)
```

Supports three training modes:
- `--smoke`: single forward pass to verify pipeline
- Standard: full training with optional `--push-to-hub`
- `--curriculum`: 3-phase progression (Easy → Medium → Hard)

**`inference.py`** (Inference Runner, 12KB):

Loads a trained checkpoint from HuggingFace Hub, connects to the env server, and runs the coordinator on a live queue. Backward-compatible with the Round 1 single-step API.

**`ui.py`** (Gradio Demo, 16KB):

Cyber orange hero-styled Gradio interface. Exposes:
- Difficulty selector (Easy / Medium / Hard / Adversarial)
- Queue viewer (shows specialist conflict side-by-side)
- Action submission panel
- Reward breakdown panel (per-component scores)
- Drift warning banner
- Final score + Hub model link

---

### End-to-End Data Flow

```
[email_triage_dataset.json]
         │  120 labeled emails
         ▼
[scenario_generator.py]
   Builds queue by difficulty + seed
         │
         ▼
[email_triage_environment.py]  ◄──── [stakeholders.py]  4 specialist opinions
      Core State Machine        ◄──── [schema_drift.py]  Policy mutations mid-episode
         │
         │  CoordinatorAction: {category, priority, should_escalate, rationale}
         │  (generated by LLM being trained)
         ▼
[graders.py]  5 independent reward signals
         │
         ▼
[train_grpo.py]  GRPO updates model weights
         │
         ▼
[HuggingFace Hub]  Checkpoint stored
         │
         ▼
[inference.py] + [ui.py]  →  Gradio Space (live demo)
```

---

## 3. Difficulty Tiers & Curriculum

| Tier | Queue Size | Specialists | Drift Events | Max Steps | R1 Weight |
|------|-----------|------------|-------------|----------|----------|
| Easy | 1 ticket | No specialists | None | 1 | 1.00 |
| Medium | 3–5 tickets | Active (80% acc.) | None | 20 | 0.40 |
| Hard | 5–10 tickets | Active (75% acc.) | 1–2 | 40 | 0.30 |
| Adversarial | 8–15 tickets | Active (65% acc.) | 3–5 | 60 | 0.25 |

Easy mode is **backward-compatible** with the Round 1 API — existing tests, clients, and scripts all work unchanged.

The curriculum script (`--curriculum`) runs three sequential phases, each loading the previous checkpoint. This ensures the model first learns basic triage format before tackling drift adaptation.

---

## 4. Anti-Reward-Hacking (4-Layer Defense)

| Layer | Mechanism | What It Blocks |
|-------|----------|---------------|
| 1. Pydantic schema | Type enforcement on all action fields | priority=99, invalid categories |
| 2. Action validation | Clamp priority to [1,5], whitelist categories | Edge cases that pass schema but are exploitative |
| 3. Step timeout | Episode ends at max_episode_steps with reward=-1 | Infinite loops, compute abuse |
| 4. Repetition detection | -0.3 penalty if last 3 actions identical | Farming reward by repeating the same action |
| 5. Reward capping | Per-step reward clamped to [-2.0, 2.0] | Unbounded accumulation from exploits |

The category whitelist is a `frozenset` — the model cannot invent new categories to game grader logic.

---

## 5. Baseline Results

| Agent | Difficulty | Avg Reward | Policy Violations | Oversight Catches |
|-------|-----------|-----------|-----------------|-----------------|
| Random | Easy | 0.03 | 0.0% | 0.0 |
| Random | Hard | 5.07 | 4.4% | 0.2 |
| Specialist Trust | Hard | 6.02 | 6.9% | 1.6 |
| Specialist Trust | Adversarial | 8.25 | 15.1% | 1.8 |
| Heuristic | Hard | 6.54 | 0.0% | 1.6 |
| Heuristic | Adversarial | 8.91 | 10.9% | 1.8 |
| **GRPO Trained (target)** | **Hard** | **~8.5+** | **<2%** | **3+** |

The gap between Specialist Trust (6.02) and Heuristic (6.54) on Hard validates the design: applying override rules when specialists are wrong is genuinely learnable and rewarded. GRPO should close the remaining gap to ~8.5+ by learning *when* to override in ambiguous cases, and adapting to drift faster than any heuristic can.

---

## 6. OpenEnv Compliance Checklist

| Requirement | Implementation |
|-------------|---------------|
| `reset()`, `step()`, `state()` API | Exact signatures, no extensions |
| Generic type safety | `Environment[EmailTriageAction, EmailTriageObservation, EmailTriageState]` |
| Pydantic serialization | All wire types are Pydantic models |
| Rewards inside env boundary | All graders compute inside `step()` |
| Client-server separation | `client.py` never imports from `server/` |
| Concurrent sessions | `SUPPORTS_CONCURRENT_SESSIONS = True` |
| Container isolation | Dockerfile based on `openenv-base` |
| Reproducibility | Seed-based determinism, verified across 5 seeds |

---

## 7. The Pitch

### One-Sentence Version

> "Oversight Inbox Arena is the first OpenEnv environment that trains LLMs to coordinate and correct a team of specialist AI agents under changing rules — the exact capability that makes AI systems safe to deploy at scale."

### Two-Minute Pitch Script

**[HOOK — 15 seconds]**

"Every AI deployment you've seen in the real world doesn't use one model. It uses a team of them. And right now, there is no RL environment that trains a model to *lead* that team. Until today."

**[PROBLEM — 30 seconds]**

"Single-agent environments are how the entire field trains LLMs to act. One input, one output, one score. But real workflows — support desks, compliance teams, enterprise ops — require a coordinator that can read specialist recommendations, catch their mistakes, adapt when policies change mid-shift, and keep every ticket resolved before the deadline. No existing OpenEnv environment tests any of this together."

**[WHAT WE BUILT — 30 seconds]**

"Oversight Inbox Arena puts one LLM coordinator in charge of four specialist AI agents — Triage, Escalation, Compliance, Responder — each with known biases and accuracy limitations. The coordinator sees their recommendations, decides when to trust or override them, and adapts when the Schema Drift engine injects a mid-episode policy change. It's scored on 5 independent reward functions: quality, SLA, policy compliance, oversight catches, and anti-cheat. All deterministic, all verifiable, no LLM judges."

**[RESULTS — 20 seconds]**

"Blindly trusting specialists scores 6.02. A smart heuristic scores 6.54. Our GRPO-trained coordinator targets 8.5+ — because it learns the nuanced override strategy no heuristic can hard-code. The gap is real, the signal is clean, and the environment is running live on HuggingFace Spaces right now."

**[CALL TO ACTION — 15 seconds]**

"The model card is on the Hub. The Space is public. The repo has 3,500 lines of tested Python. Try the hardest adversarial queue — pick the action that catches the specialist mistake — and watch the reward breakdown prove it in real time."

---

## 8. Hackathon Theme Alignment

| Hackathon Theme | How This Project Addresses It |
|----------------|------------------------------|
| Multi-Agent Interactions (Primary) | Coordinator manages 4 specialists with different biases under partial observability |
| Professional Tasks | Enterprise inbox operations — a real workflow businesses need AI for |
| Personalized Tasks | Delegation, conflict resolution, prioritization — core assistant capabilities |
| Fleet AI bonus | Coordinator monitors and corrects specialist agents — scalable oversight |
| Patronus AI bonus | Schema drift tests robustness to mid-episode policy mutations |
| Halluminate bonus | Agent interacts with multiple actors across a multi-turn episode |

| Official Guide Requirement | Implementation |
|---------------------------|----------------|
| Step-by-step action (FAQ #1) | Multi-turn queue processing |
| Programmatic verification (FAQ #1) | 11 deterministic graders |
| Adjustable difficulty (FAQ #1) | 4 tiers + curriculum scheduling |
| Multiple reward functions (FAQ #7) | 5 independent TRL reward functions |
| Anti-reward-hacking (FAQ #8, #13) | 4-layer validation + timeout + repetition + capping |
| Curriculum learning (FAQ #14) | Easy → Medium → Hard progressive training |
| Process supervision (FAQ #11) | Per-step reward component breakdown in observation |
| Step timeout (FAQ #21) | `max_episode_steps` enforced per difficulty tier |

---

## 9. Quick-Reference API

```python
# Gymnasium-style usage
from email_triage_env import EmailTriageAction, EmailTriageEnv

env = await EmailTriageEnv.from_docker_image("email-triage-env:latest", port=8010)

# Start a hard episode
result = await env.reset(difficulty="hard", seed=42)

# Submit coordinator decision
action = EmailTriageAction(
    category="billing",
    priority=3,
    should_escalate=False,
    rationale="Specialist over-escalated; this is routine billing, not urgent."
)
result = await env.step(action)
print(f"Reward: {result.observation.reward:.3f}")
print(f"Drift: {result.observation.info['policy_drift_occurred']}")
print(f"Reward breakdown: {result.observation.info['reward_components']}")
```

---

## 10. Deployment Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/reset` | POST | Start a new episode with `difficulty` + `seed` |
| `/step` | POST | Submit one coordinator action |
| `/state` | GET | Read current episode state |
| `/health` | GET | Liveness check for Space deployment |

**HF Space URL:** `https://huggingface.co/spaces/YOUR_USERNAME/oversight-inbox-arena`  
**Model URL:** `https://huggingface.co/YOUR_USERNAME/oversight-arena-grpo-t4`

---

*Document generated from live GitHub scan of [Rhushya/OpenEnv](https://github.com/Rhushya/OpenEnv) · April 2026*
