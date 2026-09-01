---
title: Email Triage Environment
emoji: "\U0001F4E7"
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: true
license: mit
tags:
    - openenv
    - rl
    - grpo
    - multi-agent
    - email-triage
    - qwen
    - lora
short_description: "Multi-agent email triage with GRPO AI"
---

# Oversight Inbox Arena

**A Multi-Agent Reinforcement Learning Environment for Safe Email Triage Under Schema Drift**

> Train an AI coordinator to manage a team of specialist agents, triage emails at scale, and adapt to mid-shift policy changes -- all with deterministic, anti-hack reward signals.

**[Demo Video (Loom)](https://www.loom.com/share/997b46f3c7cf46048ae25d3495b9db91)** | **[Blog Post](BLOG.md)** | **[Trained Model](https://huggingface.co/Rhushya/oversight-arena-grpo2)** | **[Source Code](https://github.com/Rhushya/OpenEnv)** | **[Training Notebook](https://github.com/Rhushya/OpenEnv/blob/main/envs/email_triage_env/final_running.ipynb)**

---

## What Is This?

Enterprise email teams handle thousands of messages daily. They must classify, prioritize, and escalate under time pressure and changing policies. This environment models that challenge as a **multi-agent RL problem**:

- **4 specialist AI agents** analyze each email (each with biases and errors)
- **A coordinator agent** (trained with GRPO) synthesizes their conflicting signals
- **Schema drift** changes the rules mid-episode
- **5-component reward** prevents single-metric gaming

## Architecture

```
Email Queue (5-15 tickets)
        |
        v
+-----------------------------+
| 4 SPECIALIST AGENTS         |
|                             |
| [Triage]     cat + pri      |
| [Escalation] esc recommend  |
| [Compliance] policy flags   |
| [Responder]  draft template |
+-----------------------------+
        |
        v
+-----------------------------+
| COORDINATOR (GRPO-Trained)  |
| Qwen2.5-1.5B + LoRA        |
| Input: email + specialists  |
| Output: cat, pri, escalate  |
+-----------------------------+
        |
        v
+-----------------------------+
| COMPOSITE REWARD            |
| Quality + SLA + Policy +    |
| Oversight + Efficiency      |
| + anti-hack defenses        |
+-----------------------------+
```

## How to Use This Demo

1. Select a **difficulty** level (easy / medium / hard / adversarial)
2. Click **Start Queue** to load a batch of emails
3. Read the email on the left, check specialist recommendations on the right
4. Click **AI Auto-Triage (GRPO Model)** -- watch the step-by-step pipeline:
   - Step 1: Read email metadata
   - Step 2: Collect all specialist reports
   - Step 3: Build the model prompt
   - Step 4: Run inference (API / local model / specialist consensus)
   - Step 5: Parse the XML decision
   - Step 6: Show final decision
5. Click **Submit Decision** to see your reward breakdown

## Difficulty Levels

| Level | Queue Size | Specialist Accuracy | Schema Drift | Max Steps | SLA Budget |
|-------|-----------|-------------------|--------------|-----------|------------|
| Easy | 1 ticket | 95% | 0 events | 1 | 1 step |
| Medium | 3-5 tickets | 80% | 0 events | 20 | 3 steps/ticket |
| Hard | 5-10 tickets | 75% | 2 events | 40 | 2 steps/ticket |
| Adversarial | 8-15 tickets | 65% | 4 events | 60 | 2 steps/ticket |

## GRPO Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2.5-1.5B (4-bit via Unsloth) |
| Adapter | LoRA r=8, alpha=8, q_proj + v_proj |
| Algorithm | GRPO (Group Relative Policy Optimization) |
| Steps | 50 |
| GPU | T4 (free Colab tier) |
| Training Time | ~15 minutes |
| Adapter Size | 4.37 MB |
| Reward Signal | Environment quality + XML format compliance |

## Reward Components

| Component | What It Measures | Easy | Medium | Hard | Adversarial |
|-----------|-----------------|------|--------|------|-------------|
| Quality | Category + Priority + Escalation | 100% | 40% | 30% | 25% |
| SLA | Within deadline | 0% | 20% | 20% | 20% |
| Policy | Active rule compliance | 0% | 15% | 20% | 20% |
| Oversight | Correcting specialist errors | 0% | 15% | 15% | 20% |
| Efficiency | Steps per ticket | 0% | 10% | 15% | 15% |

## Anti-Reward-Hacking Defenses

- Action validation (category clamped, priority [1,5])
- Repetition penalty (-0.3 for 3 identical actions)
- Step limits per difficulty
- Reward clamping [-2.0, 2.0]
- Escalation penalties (-0.5 for escalating spam, -0.5 for missing urgent)

## Schema Drift Events (Hard/Adversarial)

- Escalation threshold lowered (priority >= 4 becomes >= 3)
- SLA budget tightened (3 steps becomes 2 steps per ticket)
- Spam policy relaxed for internal senders
- New compliance review requirements for urgent tickets
- Priority scale reinterpretation

## Project Structure

```
envs/email_triage_env/
+-- models.py                    # Action, Observation, State schemas
+-- BLOG.md                      # Detailed writeup (this blog post)
+-- server/
    +-- email_triage_environment.py  # Main environment (658 lines)
    +-- graders.py                   # 5-component deterministic reward
    +-- stakeholders.py             # 4 specialist agent simulations
    +-- scenario_generator.py       # Queue + SLA deadline generation
    +-- schema_drift.py             # Mid-episode policy mutation engine
    +-- ui.py                       # Gradio UI + GRPO model integration
    +-- email_triage_dataset.json   # 120 labeled synthetic emails
```

## Links

| Resource | Link |
|----------|------|
| Live Demo | [HF Space](https://huggingface.co/spaces/Rhushya/email-triage-env-openenv) |
| Trained Model | [Rhushya/oversight-arena-grpo2](https://huggingface.co/Rhushya/oversight-arena-grpo2) |
| Source Code | [GitHub: Rhushya/OpenEnv](https://github.com/Rhushya/OpenEnv) |
| Training Notebook | [Colab Notebook](https://github.com/Rhushya/OpenEnv/blob/main/envs/email_triage_env/EmailTriage_GRPO_Train%20(3).ipynb) |
| Blog Post | [BLOG.md](BLOG.md) |
| Base Model | [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) |

## Tech Stack

- **Model:** Qwen2.5-1.5B + LoRA via Unsloth
- **Training:** GRPO via TRL (Transformer Reinforcement Learning)
- **Environment:** Custom Gymnasium-compatible multi-turn environment
- **UI:** Gradio 5.x with step-by-step AI pipeline visualization
- **Framework:** OpenEnv (Meta's open environment framework)
- **Deployment:** HuggingFace Spaces (Gradio SDK)

---

## FAQ

**Q: Why not just use a classifier instead of RL?**
A: A classifier only predicts category. Our agent must simultaneously classify, prioritize, decide escalation, comply with shifting policies, and manage SLA deadlines. RL lets the agent learn to balance all 5 objectives.

**Q: Why 4 specialist agents instead of 1 model?**
A: Real enterprise systems use multiple specialized models. Each has different failure modes. The coordinator must learn *when* to trust each specialist and *when* to override -- this is the oversight problem.

**Q: What is schema drift and why does it matter?**
A: Schema drift means the rules change mid-episode (e.g., escalation threshold drops from 4 to 3). This tests whether the agent memorized static rules or actually learned the *concept* of policy compliance.

**Q: Why GRPO instead of PPO or DPO?**
A: GRPO doesn't need a critic network (saves ~50% memory), works well with small models, and compares generations within a group rather than against a learned baseline. Perfect for T4 GPU training.

**Q: How do you prevent reward hacking?**
A: Five defenses: action validation, repetition penalty, step limits, reward clamping, and specific escalation penalties. The composite 5-component reward also prevents single-metric gaming.

**Q: Can this work with real emails?**
A: Yes. The environment uses a standardized Action/Observation interface. Replace the synthetic dataset with real labeled emails and the entire pipeline works identically.

**Q: How long does training take?**
A: ~15 minutes on a free T4 GPU in Google Colab. The adapter is only 4.37 MB.

**Q: What makes this different from other email classification projects?**
A: Three things: (1) Multi-agent oversight with 4 specialists, (2) Schema drift that changes rules mid-episode, (3) Anti-reward-hacking defenses. This isn't classification -- it's multi-objective decision-making under uncertainty.

---

*Built for the OpenEnv Hackathon by [Rhushya](https://huggingface.co/Rhushya)*
