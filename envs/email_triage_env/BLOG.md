# Building the Oversight Inbox Arena: Multi-Agent RL for Safe Email Triage

**Author:** [Rhushya](https://huggingface.co/Rhushya) | **Date:** April 2026 | **Hackathon:** OpenEnv

---

## TL;DR

We built a multi-agent reinforcement learning environment where an AI coordinator learns to manage 4 specialist agents, triage enterprise emails, and adapt to mid-shift policy changes. The coordinator — a Qwen2.5-1.5B model fine-tuned with GRPO — learns to synthesize conflicting specialist signals, detect errors, and comply with shifting policies. Everything runs as an interactive demo on [HuggingFace Spaces](https://huggingface.co/spaces/Rhushya/email-triage-env-openenv).

**[Watch the Demo Video (Loom) Here](https://www.loom.com/share/997b46f3c7cf46048ae25d3495b9db91)**

---

## The Problem: Email Triage is Harder Than Classification

Most people think email triage is a simple classification task: read the email, assign a category, done. In reality, enterprise email operations are *multi-dimensional decision problems*:

1. **Classification** — Is this billing, support, spam, urgent, or something else?
2. **Prioritization** — How urgent is this, on a 1-5 scale?
3. **Escalation** — Should a human reviewer see this before we respond?
4. **Policy Compliance** — Are we following the current company rules?
5. **Time Pressure** — We have SLA deadlines per ticket.

And here's the twist: **the rules change mid-shift**. The escalation threshold drops from priority >= 4 to >= 3. SLA budgets tighten. New compliance requirements appear. This is what we call *schema drift*, and it breaks static rule systems and naive ML classifiers alike.

## Our Solution: Oversight Inbox Arena

We built an **OpenEnv-compatible Gymnasium environment** that models this complexity faithfully. The key innovation is the **multi-agent oversight architecture**:

### The 4 Specialist Agents

Instead of one monolithic model, we simulate 4 specialized AI agents, each analyzing every incoming email independently:

| Specialist | Role | Typical Accuracy | Known Bias |
|-----------|------|-----------------|------------|
| **Triage** | Category + Priority prediction | 65-95% | Under-reports billing as support |
| **Escalation** | Escalation recommendation | 65-95% | Conservative (under-escalates) |
| **Compliance** | Policy violation detection | 65-95% | High false-positive rate |
| **Responder** | Draft response template | 65-95% | N/A |

Each specialist has:
- **Accuracy profiles** that vary by difficulty level
- **Systematic biases** (e.g., Triage tends to misclassify billing as support)
- **Confidence scores** that the coordinator can weigh
- **Accuracy degradation** after schema drift events

### The Coordinator (GRPO-Trained Agent)

The coordinator agent — which is what we train — sees the email *and* all 4 specialist reports. It must learn to:

1. **Trust but verify** — Use specialist recommendations but catch errors
2. **Weigh confidence** — A specialist with 90% confidence is more reliable than one at 60%
3. **Detect conflicts** — When Triage says "support" but Compliance flags the email, something's off
4. **Adapt to drift** — When policies change mid-episode, adjust behavior immediately

### Schema Drift Engine

In `hard` and `adversarial` modes, the environment injects policy mutations mid-episode:

- **Escalation threshold lowered** — Priority >= 4 becomes >= 3
- **SLA budget tightened** — 3 steps/ticket becomes 2 steps/ticket
- **Spam policy relaxed** — Internal spam can now be escalated
- **New compliance requirements** — Urgent tickets need review
- **Priority scale changed** — 1-2=low, 3=medium, 4-5=critical

These drift events test whether the agent can detect the change and adapt, rather than blindly following stale rules.

## The Reward Signal: 5 Components

Our composite reward prevents single-metric gaming:

| Component | What It Measures | Weight (Hard) |
|-----------|-----------------|---------------|
| **Quality** | Category + Priority + Escalation correctness | 30% |
| **SLA** | Tickets resolved within deadline | 20% |
| **Policy** | Compliance with currently active rules | 20% |
| **Oversight** | Correctly overriding specialist errors | 15% |
| **Efficiency** | Steps per ticket (fewer = better) | 15% |

### Anti-Reward-Hacking

RL agents are notorious for gaming reward signals. We built multiple defenses:

- **Action validation** — Categories clamped to valid set, priority to [1,5]
- **Repetition penalty** — -0.3 for 3 identical consecutive actions
- **Step limits** — Max episode steps per difficulty
- **Reward clamping** — Per-step reward capped at [-2.0, 2.0]
- **Escalation penalties** — -0.5 for escalating spam, -0.5 for not escalating urgent

## Training: GRPO on Qwen2.5-1.5B

We used **Group Relative Policy Optimization (GRPO)** from TRL to train the coordinator:

### Why GRPO?

GRPO is perfect for this use case because:
1. **No critic network needed** — Reduces memory by ~50% vs PPO
2. **Group-relative advantages** — Compares generations within a batch rather than against a learned baseline
3. **Works with small models** — We used Qwen2.5-1.5B on a free T4 GPU

### Training Setup

```
Base Model:       Qwen/Qwen2.5-1.5B (4-bit via Unsloth)
LoRA Config:      r=8, alpha=8, targets=q_proj+v_proj
Training Steps:   50
Batch Size:       1 (gradient accumulation: 4)
Generations/Step: 4
GPU:              T4 (free Colab tier)
Time:             ~15 minutes
Adapter Size:     4.37 MB
```

### Reward Functions

We used two reward signals during GRPO training:

1. **Environment Quality Reward** — The actual environment reward from `graders.py`
2. **Format Compliance Reward** — Checks that output follows the XML schema (`<category>`, `<priority>`, `<escalate>`)

### Prompt Format

```
System: You are an email triage agent. Reply ONLY with these 3 XML tags:
<category>CATEGORY</category>
<priority>N</priority>
<escalate>true|false</escalate>

User: Subject: Account balance discrepancy...
```

## Results

### Training Loss and Reward Progression

Training was conducted on a free T4 GPU in Google Colab for 50 GRPO steps:

| Step | Training Loss | Mean Reward | Format Compliance |
|------|--------------|-------------|-------------------|
| 0    | 2.45         | 0.12        | 15%               |
| 10   | 1.82         | 0.38        | 55%               |
| 20   | 1.31         | 0.56        | 78%               |
| 30   | 0.94         | 0.71        | 89%               |
| 40   | 0.72         | 0.82        | 94%               |
| 50   | 0.58         | 0.88        | 97%               |

**Key observations:**
- Loss decreased steadily from 2.45 to 0.58 (76% reduction)
- Mean reward increased from 0.12 to 0.88 (7.3x improvement)
- Format compliance jumped from 15% to 97% -- the model learned the XML schema quickly

### Before vs After Training

| Metric | Random Baseline | Trained GRPO Agent | Improvement |
|--------|----------------|-------------------|-------------|
| Avg Reward / Ticket | 0.28 | 0.88 | 3.1x |
| XML Format Valid | 0% | 97% | -- |
| Category Accuracy | 17% (random) | 78% | 4.6x |
| Escalation Accuracy | 50% (coin flip) | 85% | 1.7x |
| SLA Compliance | 40% | 95% | 2.4x |
| Policy Violations | 4.2 / episode | 0.3 / episode | 14x fewer |

### Live Demo Performance (Hard Mode, 9 tickets)

From the Autopilot run on the live T4 GPU Space:
- **Tickets resolved:** 9/9
- **Total reward:** 7.04
- **Avg reward/ticket:** 0.78
- **SLA breaches:** 0
- **Policy violations:** 1
- **Schema drift events detected:** 2

## Interactive Demo

The live demo at [huggingface.co/spaces/Rhushya/email-triage-env-openenv](https://huggingface.co/spaces/Rhushya/email-triage-env-openenv) lets you:

1. **Pick a difficulty** (easy/medium/hard/adversarial)
2. **Start a queue** of emails
3. **See specialist reports** for each email
4. **Click AI Auto-Triage** to run the trained model
5. **Watch the pipeline** — The UI shows step-by-step what the AI is doing
6. **Submit and see rewards** — Quality, SLA, Policy, Oversight breakdown

## Technical Stack

- **Framework:** [OpenEnv](https://github.com/open-env/OpenEnv) (Meta's open environment framework)
- **Model:** Qwen2.5-1.5B + LoRA via [Unsloth](https://github.com/unslothai/unsloth)
- **Training:** GRPO via [TRL](https://github.com/huggingface/trl)
- **UI:** Gradio 5.x
- **Deployment:** HuggingFace Spaces (Gradio SDK)

## What's Next

- **Scale training** — More steps, larger batch sizes, curriculum learning across difficulties
- **Multi-turn memory** — Let the coordinator remember past triage decisions
- **Real drift detection** — Train a separate drift detector module
- **Human-in-the-loop** — Connect to real email streams with human oversight

## Links

| Resource | URL |
|----------|-----|
| Live Demo | [HF Space](https://huggingface.co/spaces/Rhushya/email-triage-env-openenv) |
| Trained Model | [Rhushya/oversight-arena-grpo2](https://huggingface.co/Rhushya/oversight-arena-grpo2) |
| Source Code | [GitHub: Rhushya/OpenEnv](https://github.com/Rhushya/OpenEnv) |
| Training Notebook | [Google Colab](https://github.com/Rhushya/OpenEnv/blob/main/envs/email_triage_env/EmailTriage_GRPO_Train%20(3).ipynb) |

---

*Built for the OpenEnv Hackathon by [Rhushya](https://huggingface.co/Rhushya)*
