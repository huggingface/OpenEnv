# Oversight Inbox Arena

**A multi-agent email triage environment for OpenEnv that trains LLMs to coordinate, oversee, and correct specialist AI agents — under policy drift, partial observability, and time pressure.**

Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) | Gymnasium-style API (`reset`, `step`, `state`) | [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause)

**[Live Demo (HF Space)](https://huggingface.co/spaces/Rhushya/email-triage-env-openenv)** | **[Demo Video (Loom)](https://www.loom.com/share/997b46f3c7cf46048ae25d3495b9db91)** | **[Blog Post](https://huggingface.co/spaces/Rhushya/email-triage-env-openenv/blob/main/BLOG.md)** | **[Trained Model](https://huggingface.co/Rhushya/oversight-arena-grpo2)** | **[Training Notebook](https://github.com/Rhushya/OpenEnv/blob/main/envs/email_triage_env/EmailTriage_GRPO_Train%20(3).ipynb)** | **[Source Code](https://github.com/Rhushya/OpenEnv)**

---

## Table of Contents

- [What Is This?](#what-is-this)
- [Why Does This Matter?](#why-does-this-matter)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Specialist Agents (Multi-Agent Design)](#specialist-agents-multi-agent-design)
- [Schema Drift (Novelty)](#schema-drift-novelty)
- [Reward System (Verifiable, Deterministic)](#reward-system-verifiable-deterministic)
- [Anti-Reward-Hacking Protections](#anti-reward-hacking-protections)
- [Curriculum Learning](#curriculum-learning)
- [Action & Observation Space](#action--observation-space)
- [Baseline Results](#baseline-results)
- [Quick Start](#quick-start)
- [Training with GRPO](#training-with-grpo)
- [Evaluation](#evaluation)
- [Deployment (HuggingFace Spaces)](#deployment-huggingface-spaces)
- [For Developers: How to Extend](#for-developers-how-to-extend)
- [OpenEnv Compliance](#openenv-compliance)
- [Hackathon Theme Alignment](#hackathon-theme-alignment)
- [File Inventory](#file-inventory)

---

## What Is This?

Oversight Inbox Arena is a **reinforcement learning environment** built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv). It simulates a realistic enterprise inbox where one LLM coordinator agent must manage a team of four specialist AI agents to triage, escalate, review, and resolve a queue of incoming tickets.

**In plain terms:** Imagine you run a support team with four AI assistants. One classifies tickets, one handles escalations, one checks compliance, one drafts responses. They each make mistakes. Your job (as the coordinator) is to catch those mistakes, adapt when the rules change mid-shift, and make sure every ticket is handled correctly before the deadline.

That's what this environment trains an LLM to do.

---

## Why Does This Matter?

### The Gap in Current RL Environments

Most LLM training environments are **single-agent, single-step**: one input, one output, one score. But real-world AI deployment requires:

- **Multiple AI agents** working together, each with different failure modes
- **Multi-step decisions** where early mistakes cascade into later SLA breaches
- **Changing rules** — compliance policies update, escalation thresholds shift
- **Oversight** — someone must catch when the triage bot mistakes an outage for spam

There was no OpenEnv environment testing all of these together. Now there is.

### What This Adds to OpenEnv

| Capability | Before (existing envs) | After (this env) |
|-----------|----------------------|-----------------|
| Episode length | Single-step (1 action) | Multi-turn queues (5-15 tickets) |
| Agent count | 1 agent acts alone | 1 coordinator + 4 specialists |
| Observability | Full state visible | Partial — coordinator sees summaries only |
| Policy stability | Rules stay fixed | Schema drift — policies mutate mid-episode |
| Reward signal | Single score | 5 independent verifiable reward functions |
| Anti-hacking | None | Action validation, timeout, repetition detection, reward capping |
| Difficulty scaling | Static | 4 tiers with curriculum support |
| Backward compat | N/A | Easy mode = identical single-step behavior |

---

## How It Works

### The High-Level RL Loop

This environment follows the exact RL loop described in the hackathon guide (Section 2):

```
1. Give the model a prompt (inbox queue + specialist reports)
2. Let it generate an action (category, priority, escalation decision)
3. Execute that action in the environment (verify against ground truth)
4. Convert the result into rewards (5 independent signals)
5. Update the model (GRPO shifts probability toward better triage)
```

### Episode Flow

```
[reset(difficulty="hard", seed=42)]
    |
    Queue of 5-10 tickets loaded
    Specialist agents simulate recommendations
    Coordinator sees: ticket + specialist reports + active policies
    |
[step(category="billing", priority=3, should_escalate=False)]
    |
    Environment scores with 5 independent reward functions:
      R1: Quality (correct category/priority/escalation?)
      R2: SLA (resolved before deadline?)
      R3: Policy compliance (followed current rules?)
      R4: Oversight (caught specialist mistakes?)
      R5: Anti-cheat (no repeated/gaming actions?)
    |
    Next ticket loaded, new specialist reports generated
    Maybe a policy changes mid-episode (drift!)
    |
[step(...)]  --> repeat until all tickets resolved or timeout
    |
    Episode ends. Rewards decomposed per-function.
```

### Difficulty Tiers

| Tier | Queue Size | Schema Drift | Specialist Accuracy | Max Steps |
|------|-----------|-------------|-------------------|-----------|
| **Easy** | 1 ticket (Round 1 compatible) | None | 95% | 1 |
| **Medium** | 3-5 tickets | None | 80% | 20 |
| **Hard** | 5-10 tickets | 1-2 mutations | 75% | 40 |
| **Adversarial** | 8-15 tickets | 3-5 mutations | 65% | 60 |

**Easy mode is identical to the original single-step environment.** Old code, old tests, old inference scripts — everything works unchanged.

---

## Architecture

```
envs/email_triage_env/
|
|-- models.py                          # Pydantic contracts (Action, Observation, State)
|-- client.py                          # EnvClient subclass (WebSocket)
|-- openenv.yaml                       # OpenEnv environment manifest
|-- inference.py                       # Baseline inference script (Round 1 compat)
|-- train_grpo.py                      # GRPO training with 5 reward functions + curriculum
|-- eval_benchmark.py                  # Evaluation with 3 baseline agents
|-- test_env.py / test_http.py         # Comprehensive tests
|
|-- server/
|   |-- app.py                         # FastAPI application (create_app)
|   |-- email_triage_environment.py    # Core: reset/step/state + anti-hack protections
|   |-- graders.py                     # 11 deterministic reward graders
|   |-- scenario_generator.py          # Deterministic scenarios from seeds
|   |-- stakeholders.py               # 4 specialist agent simulations
|   |-- schema_drift.py               # Mid-episode policy mutation engine
|   |-- email_triage_dataset.json      # 120 labeled emails
```

### How the Pieces Connect

```
                    +-----------------+
                    |  Coordinator    |  <-- LLM being trained
                    |  (your model)   |
                    +--------+--------+
                             |
                     action: category, priority, escalate
                             |
                             v
+-------------------------------------------------------------------+
|                    Environment Server                              |
|                                                                    |
|  +--------------+   +--------------+   +-----------+  +----------+|
|  | Scenario     |   | Specialist   |   | Drift     |  | Anti-    ||
|  | Generator    |   | Pool (4)     |   | Engine    |  | Hack     ||
|  | (queues)     |   | Triage       |   | Policy    |  | Guard    ||
|  |              |   | Escalation   |   | mutations |  | Validate ||
|  |              |   | Compliance   |   |           |  | Timeout  ||
|  |              |   | Responder    |   |           |  | Clamp    ||
|  +--------------+   +--------------+   +-----------+  +----------+|
|           |                |                |              |       |
|           v                v                v              v       |
|  +--------------------------------------------------------+------+|
|  |  5 Independent Reward Functions (all deterministic)          | |
|  |  R1:quality  R2:SLA  R3:policy  R4:oversight  R5:anti-cheat | |
|  +--------------------------------------------------------------+|
+-------------------------------------------------------------------+
```

---

## Specialist Agents (Multi-Agent Design)

The environment simulates four specialist agents. Each processes the current ticket and produces a recommendation. The coordinator must decide whether to **trust or override** them.

| Specialist | What It Does | Accuracy | Known Bias |
|-----------|-------------|---------|-----------|
| **Triage Agent** | Predicts category and priority | 75-95% | Under-prioritizes billing tickets |
| **Escalation Agent** | Recommends whether to escalate | 80-95% | Over-escalates when uncertain |
| **Compliance Agent** | Flags policy violations | 85-98% | High false-positive rate |
| **Responder Agent** | Suggests response template | 70-90% | Too formulaic, misses nuance |

**Why this matters:** Specialists are useful but imperfect. Blindly trusting them scores 6.02 on hard mode. A coordinator that overrides their mistakes scores 6.54+. A GRPO-trained coordinator should score 8.5+.

**This IS the oversight signal.** When the triage agent says "spam" but the coordinator correctly identifies it as "urgent" — that's an oversight catch, and it gets rewarded.

---

## Schema Drift (Novelty)

In the real world, rules change. Support policies get updated. Escalation thresholds shift. A robust AI coordinator must adapt on the fly.

**How it works:**
1. At episode start, 5 policies are active (e.g., "Escalate if priority >= 4")
2. At configurable points (30-60% through the queue), the drift engine mutates policies
3. The coordinator sees updated policies in its observation
4. If it adapts within 2 steps, it gets a drift adaptation bonus

### Drift Types

| Drift | Example |
|-------|---------|
| Escalation threshold lowered | "Escalate if >= 4" becomes "Escalate if >= 3" |
| SLA tightened | 3 steps/ticket becomes 2 steps/ticket |
| Spam policy relaxed | Internal spam can now be escalated |
| New compliance rule added | "All urgent tickets require compliance review" |
| Priority scale changed | Bucket boundaries shift |

**After each drift, specialist accuracy degrades by 10%** — forcing the coordinator to rely more on its own judgment.

---

## Reward System (Verifiable, Deterministic)

> *"Use multiple independent reward functions, not just one. If you only have a single reward signal, it is easier for the model to hack it."* — Official Hackathon Guide, FAQ #7

**Every reward is deterministic and verifiable.** No LLM judges, no neural reward models, no reward hacking. Given an action and ground truth labels, anyone can recompute the exact same score.

### 5 Independent Reward Functions

These are passed as **separate functions** to TRL's `GRPOTrainer.reward_funcs`:

| # | Function | What It Measures | Range |
|---|----------|-----------------|-------|
| R1 | `reward_quality` | Category + priority + escalation accuracy | [0, 1] |
| R2 | `reward_sla` | Tickets resolved before SLA deadline | [0, 1] |
| R3 | `reward_compliance` | Actions follow current active policies | [0, 1] |
| R4 | `reward_oversight` | Coordinator caught specialist mistakes | [0, 1] |
| R5 | `reward_no_hacking` | No repetition abuse, no timeout exploitation | [-2, 0] |

### Why 5 Functions, Not 1

The official guide (FAQ #7, #8, #13) emphasizes that:
- A single reward signal is easier to hack
- Multiple independent checks reduce gaming risk
- Each function independently tells the model something different

Our 5 functions are **orthogonal** — quality and oversight can improve independently of SLA and compliance.

### Hard-Coded Safety Penalties

| Bad Action | Penalty | Why |
|-----------|---------|-----|
| Escalating spam | -0.5 | Wastes human reviewer time |
| Ignoring urgent incidents | -0.5 | Safety-critical failure |

### Component Weights by Difficulty

| Tier | Quality | SLA | Policy | Oversight | Efficiency |
|------|---------|-----|--------|-----------|-----------|
| Easy | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Medium | 0.40 | 0.20 | 0.15 | 0.15 | 0.10 |
| Hard | 0.30 | 0.20 | 0.20 | 0.15 | 0.15 |
| Adversarial | 0.25 | 0.20 | 0.20 | 0.20 | 0.15 |

---

## Anti-Reward-Hacking Protections

> *"Reward hacking is one of the biggest practical failure modes. The model may learn shortcuts that maximize your reward without solving the real task."* — Official Hackathon Guide, FAQ #8

We implement **four layers** of protection:

### 1. Action Validation (Input Sanitization)
```python
def _validate_action(action):
    action.priority = max(1, min(5, int(action.priority)))  # Clamp to [1,5]
    if action.category not in VALID_CATEGORIES:              # Reject invented categories
        action.category = "other"
    action.should_escalate = bool(action.should_escalate)    # Force boolean
    return action
```
- Pydantic schema rejects completely invalid inputs (priority=99)
- Our validation catches edge cases that pass schema but are exploitative

### 2. Step Timeout (Resource Limit)
```python
if step_count > max_episode_steps:
    return observation(reward=-1.0, done=True, reason="timeout")
```
- Easy: 1 step, Medium: 20, Hard: 40, Adversarial: 60
- Prevents infinite loops and compute abuse

### 3. Repetition Detection (Anti-Gaming)
```python
if last_3_actions_are_identical:
    reward -= 0.3  # per-step penalty
```
- Detects when the model submits the same action repeatedly to farm reward
- Tracked per-episode and available to the anti-cheat reward function

### 4. Reward Capping (Bounded Accumulation)
```python
reward = max(-2.0, min(2.0, reward))  # Per-step clamp
```
- Prevents unbounded reward accumulation from environment exploits
- Each step's reward is clamped to [-2.0, 2.0]

### 5. Locked Category Set
```python
_VALID_CATEGORIES = frozenset({"billing", "support", "spam", "urgent", "marketing", "other"})
```
- Model cannot invent new categories to exploit reward computation
- Categories are validated against a frozen set every step

---

## Curriculum Learning

> *"Start with the easiest version that still proves the concept. RL only works if the probability of getting a good answer is greater than zero."* — Official Hackathon Guide, FAQ #6, #14

Our training script supports explicit curriculum scheduling:

```bash
python train_grpo.py --curriculum --model Qwen/Qwen3-1.7B
```

This runs three training phases:

| Phase | Difficulty | Purpose |
|-------|-----------|---------|
| Phase 1 | Easy (32 prompts) | Learn basic triage format and categories |
| Phase 2 | Medium (64 prompts) | Learn multi-step coordination and specialist usage |
| Phase 3 | Hard (128 prompts) | Learn drift adaptation and oversight under pressure |

Each phase loads the checkpoint from the previous phase, progressively building capability.

**Why curriculum matters:** Without it, the model starts on hard mode where success probability is near zero, gets no reward signal, and learning stalls (FAQ #14).

---

## Action & Observation Space

### Action

```python
class EmailTriageAction(Action):
    category: Literal["billing", "support", "spam", "urgent", "marketing", "other"]
    priority: int  # 1-5 (Pydantic enforced)
    should_escalate: bool
    rationale: Optional[str] = None  # for oversight quality scoring
```

### Observation

```python
class EmailTriageObservation(Observation):
    email_id: str          # Current ticket ID
    subject: str           # Email subject line
    body_snippet: str      # First 280 chars of body
    sender: str            # Sender address
    sender_domain: str     # Domain for internal/external check
    is_internal: bool      # Internal vs external sender
    task_id: TaskId        # Current difficulty tier
    info: Dict[str, Any]   # Rich context (see below)
```

**`info` dict includes (in multi-turn mode):**
- `specialist_reports` — recommendations from all 4 specialists
- `active_policies` — current policy rules (may change after drift)
- `policy_drift_occurred` — whether a policy just changed
- `drift_description` — human-readable description of what changed
- `queue_position`, `tickets_remaining`, `sla_deadline_step`
- `reward_components` — per-step breakdown of all 5 reward signals
- `event_log` — last 5 actions for self-monitoring

### State

```python
class EmailTriageState(State):
    total_reward: float      # Cumulative episode reward
    difficulty: Difficulty
    queue_size: int          # Total tickets in episode
    tickets_resolved: int
    tickets_remaining: int
    sla_breaches: int        # SLA deadline misses
    policy_violations: int   # Policy rule violations
    oversight_catches: int   # Specialist errors caught
    drift_count: int         # Policy mutations occurred
```

---

## Baseline Results

Three baseline agents evaluated across all difficulty tiers (5 deterministic seeds):

```
Agent                Difficulty      Avg Reward  Violations  Oversight
---------------------------------------------------------------------------
random               easy                 0.03       0.0%         0.0
random               hard                 5.07       4.4%         0.2
random               adversarial          7.64      12.2%         0.6
---------------------------------------------------------------------------
specialist_trust     hard                 6.02       6.9%         1.6
specialist_trust     adversarial          8.25      15.1%         1.8
---------------------------------------------------------------------------
heuristic            hard                 6.54       0.0%         1.6
heuristic            adversarial          8.91      10.9%         1.8
---------------------------------------------------------------------------
GRPO trained (est.)  hard                ~8.5+       <2%          3+
```

**Key insight:** Heuristic beats specialist_trust because it applies override rules. This validates the design — **oversight coordination is the learnable skill.**

The gap from heuristic (6.54) to trained (8.5+) is where GRPO adds value: learning *when* to override in ambiguous cases and adapting to drift faster.

---

## Quick Start

### Python (Direct)

```bash
cd OpenEnv
pip install -e .
cd envs/email_triage_env

# Test environment
PYTHONPATH=../../src:../../envs python test_env.py

# Run evaluation
PYTHONPATH=../../src:../../envs python eval_benchmark.py --seeds 5
```

### Docker

```bash
docker build -t email-triage-env -f server/Dockerfile .
docker run -p 8000:8000 email-triage-env
curl http://localhost:8000/health
```

### HTTP API

```bash
# Reset
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "hard", "seed": 42}'

# Step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"category": "billing", "priority": 3, "should_escalate": false}}'

# State
curl http://localhost:8000/state
```

### Python Client

```python
from email_triage_env import EmailTriageAction, EmailTriageEnv

env = await EmailTriageEnv.from_docker_image("email-triage-env:latest", port=8010)
result = await env.reset(difficulty="hard", seed=42)
action = EmailTriageAction(category="billing", priority=3, should_escalate=False)
result = await env.step(action)
print(f"Reward: {result.observation.reward:.3f}")
```

---

## Training with GRPO

> *"GRPO is a more efficient evolution relative to PPO, especially by simplifying away parts like the value model."* — Official Hackathon Guide, FAQ #9

### Smoke Test (Verify Pipeline)

```bash
python train_grpo.py --smoke
```

### Standard Training

```bash
python train_grpo.py --model Qwen/Qwen3-1.7B --max-steps 100 --report-to wandb
```

### Curriculum Training (Recommended)

```bash
python train_grpo.py --curriculum --model Qwen/Qwen3-1.7B --max-steps 50
```

### Low-VRAM (Unsloth + Free Colab T4)

```bash
python train_grpo.py --unsloth --model Qwen/Qwen3-1.7B --max-steps 50
```

### How TRL Integrates

The training script uses TRL's `environment_factory` pattern with **5 independent reward functions**:

```python
trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    reward_funcs=[
        reward_quality,      # R1: triage accuracy
        reward_oversight,    # R2: specialist error correction
        reward_compliance,   # R3: policy adherence
        reward_sla,          # R4: deadline adherence
        reward_no_hacking,   # R5: anti-cheat penalty
    ],
    train_dataset=dataset,
    environment_factory=OversightInboxEnv,
)
```

---

## Evaluation

```bash
# All baselines × all difficulties
python eval_benchmark.py --seeds 10

# Single difficulty
python eval_benchmark.py --difficulty hard --seeds 10

# Save JSON for comparison
python eval_benchmark.py --output results.json
```

Three built-in baseline agents:
- **Random** — random category, priority, escalation
- **Specialist trust** — blindly follows specialist recommendations
- **Heuristic** — follows specialists + safety override rules

---

## Deployment (HuggingFace Spaces)

### Push to Spaces

```bash
# Install CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create Space
huggingface-cli repo create email-triage-env --type space --space-sdk docker

# Push
cd envs/email_triage_env
git init
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/email-triage-env
git add .
git commit -m "Oversight Inbox Arena"
git push space main
```

### Deploy Locally with Uvicorn

```bash
PYTHONPATH=../../src:../../envs uvicorn email_triage_env.server.app:app --host 0.0.0.0 --port 8000
```

### Verify Deployment

```bash
curl http://YOUR_SPACE_URL/health
# {"status": "healthy"}
```

---

## For Developers: How to Extend

### Add a New Specialist Agent

Edit `server/stakeholders.py`:

```python
def _simulate_sentiment(self, email: Dict[str, Any]) -> Dict[str, Any]:
    return {"sentiment": "negative", "confidence": 0.85, "correct": True}
```

### Add a New Drift Type

Edit `server/schema_drift.py`:

```python
{"drift_type": "new_category", "description": "Category 'security' added", "trigger_fraction": 0.45}
```

### Add a New Reward Grader

Edit `server/graders.py`:

```python
def my_grader(action, email, **kwargs) -> float:
    """Must return float in [0, 1]. Must be deterministic."""
    ...
```

### Add a New Difficulty Tier

Edit `TASK_CONFIG` in `server/email_triage_environment.py`:

```python
"nightmare": {
    "difficulty": "nightmare",
    "multi_turn_weights": {"quality": 0.20, "sla": 0.25, ...},
    "max_episode_steps": 80,
}
```

---

## OpenEnv Compliance

| Requirement | How We Comply |
|-------------|--------------|
| Gymnasium API (`reset`, `step`, `state`) | Exact signatures, no extensions |
| Generic type safety | `Environment[EmailTriageAction, EmailTriageObservation, EmailTriageState]` |
| Pydantic serialization | All wire types are Pydantic models |
| Rewards inside environment boundary | All graders compute inside `step()` |
| Client-server separation | `client.py` never imports from `server/` |
| `SUPPORTS_CONCURRENT_SESSIONS = True` | Stateless across sessions |
| Container isolation | Dockerfile based on `openenv-base` |

---

## Hackathon Theme Alignment

| Theme | How This Addresses It |
|-------|----------------------|
| **Multi-Agent Interactions** (Primary) | Coordinator manages 4 specialists with different biases under partial observability |
| **Professional Tasks** | Enterprise inbox operations — a workflow businesses actually need AI for |
| **Personalized Tasks** | Delegation, conflict resolution, prioritization — core assistant capabilities |
| **Fleet AI bonus** | Coordinator monitors and corrects specialist agents — this IS scalable oversight |
| **Patronus AI bonus** | Schema drift tests robustness to policy mutations |
| **Halluminate bonus** | Agent interacts with multiple actors to achieve goals |

### Official Guide Alignment

| Guide Requirement (FAQ #) | Our Implementation |
|--------------------------|-------------------|
| Step-by-step action (#1) | Multi-turn queue processing |
| Programmatic verification (#1) | 11 deterministic graders |
| Adjustable difficulty (#1) | 4 tiers + curriculum |
| Multiple reward functions (#7) | 5 independent TRL reward functions |
| Anti-reward-hacking (#8, #13) | Validation + timeout + repetition + capping |
| Curriculum learning (#14) | Easy → medium → hard progression |
| Process supervision (#11) | Per-step reward components |
| Step timeout (#21) | max_episode_steps per difficulty |
| Reproducibility | Seed-based determinism verified |
| Deploy early (#13) | Docker + FastAPI + HF Spaces guide |

---

## Dataset

- **Path**: `server/email_triage_dataset.json`
- **Size**: 120 labeled synthetic emails
- **Labels**: `id`, `subject`, `body`, `sender`, `sender_domain`, `is_internal`, `true_category`, `true_priority`, `needs_escalation`, `difficulty`
- **Categories**: billing, support, spam, urgent, marketing, other

---

## Running Tests

```bash
# Unit tests (all tiers + determinism + backward compat + anti-hack)
python test_env.py

# HTTP server end-to-end
python test_http.py

# Evaluation benchmark
python eval_benchmark.py --seeds 5
```

---

## File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `models.py` | 55 | Action, Observation, State contracts |
| `client.py` | 70 | WebSocket client |
| `server/email_triage_environment.py` | 610 | Core env + anti-hack protections |
| `server/graders.py` | 220 | 11 deterministic graders |
| `server/scenario_generator.py` | 100 | Seed-based scenarios |
| `server/stakeholders.py` | 160 | 4 specialist simulations |
| `server/schema_drift.py` | 250 | Policy mutation engine |
| `server/app.py` | 47 | FastAPI application |
| `train_grpo.py` | 320 | GRPO + 5 rewards + curriculum |
| `eval_benchmark.py` | 250 | 3-agent baseline evaluation |
| `test_env.py` | 140 | Unit tests |
| `test_http.py` | 75 | HTTP tests |
| `email_triage_dataset.json` | ~1200 | 120 labeled emails |

**Total**: ~3,500 lines of tested Python.

---

## License

BSD 3-Clause License (same as OpenEnv parent repository)

## Author

**Rhushya KC** — Meta PyTorch OpenEnv Hackathon Grand Finale 2026

## Acknowledgments

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) by Meta PyTorch team
- [TRL](https://github.com/huggingface/trl) by Hugging Face
- [Unsloth](https://unsloth.ai) for low-VRAM training
