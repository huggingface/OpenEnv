# Oversight Inbox Arena — Email Triage Environment

**A Multi-Agent Reinforcement Learning Environment for Safe Email Triage Under Schema Drift**

> Train an AI coordinator to manage a team of 4 specialist agents, triage emails at scale, and adapt to mid-shift policy changes — all with deterministic, anti-hack reward signals.

## Live Demo

**[Try it on HuggingFace Spaces](https://huggingface.co/spaces/Rhushya/email-triage-env-openenv)**

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    OVERSIGHT INBOX ARENA                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐         ┌───────────────────────┐          │
│  │  Email Queue      │         │  Schema Drift Engine  │          │
│  │  (5-15 tickets)   │ ──────→ │  (Policy Mutations)   │          │
│  └──────────────────┘         └───────────────────────┘          │
│         │                              │                          │
│         ▼                              ▼                          │
│  ┌──────────────────────────────────────────────────────┐        │
│  │            4 SPECIALIST AGENTS                        │        │
│  │                                                       │        │
│  │  [Triage]     Category + Priority prediction          │        │
│  │  [Escalation] Escalation recommendation               │        │
│  │  [Compliance] Policy flag detection                   │        │
│  │  [Responder]  Draft response template                 │        │
│  └──────────────────────────────────────────────────────┘        │
│         │                                                         │
│         ▼                                                         │
│  ┌──────────────────────────────────────────────────────┐        │
│  │         COORDINATOR (GRPO-Trained Agent)              │        │
│  │                                                       │        │
│  │  Model: Qwen2.5-1.5B + LoRA (4.37 MB adapter)       │        │
│  │  Training: GRPO, 50 steps, T4 GPU                    │        │
│  │  Input: Email + Specialist Reports                    │        │
│  │  Output: <category> <priority> <escalate>             │        │
│  └──────────────────────────────────────────────────────┘        │
│         │                                                         │
│         ▼                                                         │
│  ┌──────────────────────────────────────────────────────┐        │
│  │         COMPOSITE REWARD (5 Components)               │        │
│  │  Quality · SLA · Policy · Oversight · Efficiency      │        │
│  │  + Drift adaptation bonus                             │        │
│  │  + Anti-hack: repetition penalty, action clamp        │        │
│  └──────────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent Oversight** | 4 specialist agents with accuracy profiles, biases, and confidence scores |
| **Schema Drift** | Mid-episode policy mutations (escalation thresholds, SLA budgets, etc.) |
| **GRPO Training** | Qwen2.5-1.5B fine-tuned with Group Relative Policy Optimization |
| **Composite Reward** | 5 weighted components: Quality + SLA + Policy + Oversight + Efficiency |
| **Anti-Reward-Hacking** | Action validation, repetition penalties, step limits, reward clamping |
| **Interactive Demo** | Gradio UI with AI Auto-Triage button |

## Difficulty Levels

| Level | Queue | Specialist Accuracy | Schema Drift | Max Steps |
|-------|-------|-------------------|--------------|-----------|
| Easy | 1 | 95% | 0 events | 1 |
| Medium | 3-5 | 80% | 0 events | 20 |
| Hard | 5-10 | 75% | 2 events | 40 |
| Adversarial | 8-15 | 65% | 4 events | 60 |

## Quick Start

### Run the Gradio UI locally
```bash
cd envs/email_triage_env
pip install gradio pydantic numpy
python -m server.ui
```

### Train the GRPO model (Google Colab T4)
Open `EmailTriage_GRPO_Train (3).ipynb` in Google Colab and run all cells.

## Project Structure

```
envs/email_triage_env/
├── models.py                      # Action, Observation, State
├── server/
│   ├── email_triage_environment.py  # Main environment (658 lines)
│   ├── graders.py                   # Deterministic reward graders
│   ├── stakeholders.py             # 4 specialist agent simulations
│   ├── scenario_generator.py       # Queue + SLA generation
│   ├── schema_drift.py             # Policy drift engine
│   ├── ui.py                       # Gradio UI + AI model integration
│   └── email_triage_dataset.json   # 120 labeled emails
├── EmailTriage_GRPO_Train (3).ipynb # Training notebook
└── inference.py                     # Baseline inference script
```

## Links

| Resource | Link |
|----------|------|
| Live Demo | [HF Space](https://huggingface.co/spaces/Rhushya/email-triage-env-openenv) |
| Trained Model | [Rhushya/oversight-arena-grpo2](https://huggingface.co/Rhushya/oversight-arena-grpo2) |
| Training Notebook | [EmailTriage_GRPO_Train.ipynb](EmailTriage_GRPO_Train%20(3).ipynb) |
