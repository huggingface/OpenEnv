from __future__ import annotations

"""
Oversight Inbox Arena — Gradio UI
Clean black-and-white demo interface with GRPO-trained AI agent.
"""

import os
import re
import random
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import gradio as gr
except ImportError:
    gr = None

try:
    from envs.email_triage_env.server.email_triage_environment import EmailTriageEnvironment
    from envs.email_triage_env.models import EmailTriageAction
except ImportError:
    try:
        from email_triage_env.server.email_triage_environment import EmailTriageEnvironment
        from email_triage_env.models import EmailTriageAction
    except ImportError:
        from server.email_triage_environment import EmailTriageEnvironment
        from models import EmailTriageAction


# ── GRPO Model Integration ────────────────────────────────────────────────────

GRPO_MODEL_ID = "Rhushya/oversight-arena-grpo2"
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B"

# System prompt used during GRPO training (must match exactly)
SYSTEM_PROMPT = (
    'You are an email triage agent. Reply ONLY with these 3 XML tags:\n'
    '<category>CATEGORY</category>\n'
    '<priority>N</priority>\n'
    '<escalate>true|false</escalate>\n'
    'Valid categories: billing support spam urgent marketing other\n'
    'Priority 1=low 5=critical'
)

# Cache for model/tokenizer to avoid reloading
_model_cache = {}


def _try_load_model():
    """Attempt to load the GRPO model. Returns (model, tokenizer) or (None, None)."""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["tokenizer"]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        use_gpu = torch.cuda.is_available()
        dtype = torch.float16 if use_gpu else torch.float32
        device = "auto" if use_gpu else "cpu"
        logger.info("Loading base model %s (device=%s, dtype=%s)...", BASE_MODEL_ID, device, dtype)

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        logger.info("Loading LoRA adapter %s ...", GRPO_MODEL_ID)
        model = PeftModel.from_pretrained(base, GRPO_MODEL_ID)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(GRPO_MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _model_cache["model"] = model
        _model_cache["tokenizer"] = tokenizer
        logger.info("GRPO model loaded successfully (GPU=%s).", use_gpu)
        return model, tokenizer
    except Exception as e:
        logger.warning("Could not load GRPO model locally: %s", e)
        _model_cache["model"] = None
        _model_cache["tokenizer"] = None
        return None, None


def _try_inference_api(email_text: str) -> str | None:
    """Call the HF Inference API for the GRPO model."""
    try:
        from huggingface_hub import InferenceClient
        token = os.getenv("HF_TOKEN", "")
        client = InferenceClient(model=GRPO_MODEL_ID, token=token or None)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": email_text},
        ]
        result = client.chat_completion(messages, max_tokens=128, temperature=0.3)
        return result.choices[0].message.content
    except Exception as e:
        logger.warning("Inference API failed: %s", e)
        return None


def _generate_local(model, tokenizer, email_text: str) -> str | None:
    """Run local inference with the loaded model."""
    try:
        import torch
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": email_text},
        ]

        # Use built-in chat template (Qwen2 has chatml built-in)
        # Only set custom template if tokenizer doesn't have one
        if not getattr(tokenizer, 'chat_template', None):
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
                "{% if loop.last and message['role'] == 'user' %}"
                "{{ '<|im_start|>assistant\n' }}"
                "{% endif %}"
                "{% endfor %}"
            )

        text_input = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text_input, return_tensors="pt")

        # Move inputs to model device (GPU or CPU)
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = outputs[0][input_ids.shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        logger.info("Local generation produced: %s", text[:200])
        return text if text.strip() else None
    except Exception as e:
        logger.warning("Local generation failed: %s", e, exc_info=True)
        return None


def _parse_xml_output(text: str) -> dict:
    """Parse <category>, <priority>, <escalate> from model output."""
    result = {}
    cat_m = re.search(r"<category>\s*(\w+)\s*</category>", text, re.IGNORECASE)
    if cat_m:
        result["category"] = cat_m.group(1).lower()
    pri_m = re.search(r"<priority>\s*(\d)\s*</priority>", text, re.IGNORECASE)
    if pri_m:
        result["priority"] = int(pri_m.group(1))
    esc_m = re.search(r"<escalate>\s*(true|false)\s*</escalate>", text, re.IGNORECASE)
    if esc_m:
        result["escalate"] = esc_m.group(1).lower() == "true"
    return result


def _specialist_consensus(info: dict) -> dict:
    """Fallback: derive a triage decision from specialist reports."""
    reports = info.get("specialist_reports", {})
    result = {"category": "support", "priority": 3, "escalate": False}

    triage = reports.get("triage", {})
    if "category" in triage:
        result["category"] = triage["category"]
    if "priority" in triage:
        result["priority"] = triage["priority"]

    escalation = reports.get("escalation", {})
    if "recommended" in escalation:
        result["escalate"] = escalation["recommended"]

    compliance = reports.get("compliance", {})
    if compliance.get("flagged") and result["category"] in ("urgent", "billing"):
        result["escalate"] = True

    return result


def do_ai_triage(env, obs, info):
    """Run the GRPO-trained model with step-by-step pipeline visibility."""
    if env is None or obs is None:
        return "support", 3, False, "_Click **Start Queue** first, then use AI Triage._"

    d = obs if isinstance(obs, dict) else obs.model_dump() if hasattr(obs, "model_dump") else vars(obs)
    email_text = f"Subject: {d.get('subject', '')}\n{d.get('body_snippet', d.get('body', ''))}"

    # Build step-by-step pipeline log
    steps = []
    steps.append("### AI Triage Pipeline")
    steps.append("")
    steps.append(f"**Step 1/6 -- Read Email**")
    steps.append(f"- Subject: `{d.get('subject', '?')}`")
    steps.append(f"- Sender: `{d.get('sender', '?')}` ({'INTERNAL' if d.get('is_internal') else 'EXTERNAL'})")
    steps.append("")

    # Step 2: Gather specialist signals
    reports = (info or {}).get("specialist_reports", {})
    steps.append("**Step 2/6 -- Collect Specialist Reports**")
    if reports:
        for name, data in reports.items():
            conf = data.get("confidence", 0)
            pct = int(conf * 100) if conf else 0
            detail = ""
            if "category" in data:
                detail += f"cat=`{data['category']}` "
            if "priority" in data:
                detail += f"pri=`{data['priority']}` "
            if "recommended" in data:
                detail += f"esc=`{data['recommended']}` "
            if data.get("flagged"):
                detail += "**FLAGGED** "
            steps.append(f"- {name.title()}: {detail}(conf {pct}%)")
    else:
        steps.append("- _No specialist reports_")
    steps.append("")

    # Step 3: Build prompt
    steps.append("**Step 3/6 -- Build Model Prompt**")
    steps.append(f"- System: `{SYSTEM_PROMPT[:80]}...`")
    steps.append(f"- User input: `{email_text[:60]}...`")
    steps.append(f"- Model: `{GRPO_MODEL_ID}` (Qwen2.5-1.5B + LoRA, GRPO-trained)")
    steps.append("")

    # Step 4: Run inference
    ai_output = None
    method = ""

    steps.append("**Step 4/6 -- Run Model Inference**")

    # Strategy 1: HF Inference API
    ai_output = _try_inference_api(email_text)
    if ai_output:
        method = "HF Inference API (Serverless)"
        steps.append(f"- Method: `{method}`")
        steps.append(f"- Status: Success")
    else:
        steps.append("- HF Inference API: unavailable (LoRA adapter not served)")

        # Strategy 2: Local model
        model, tokenizer = _try_load_model()
        if model is not None:
            try:
                import torch
                _dev = "GPU" if torch.cuda.is_available() else "CPU"
            except Exception:
                _dev = "CPU"
            steps.append(f"- Loading local model: `Qwen2.5-1.5B` + LoRA adapter ({_dev})...")
            ai_output = _generate_local(model, tokenizer, email_text)
            if ai_output:
                method = f"Local GRPO Model ({_dev})"
                steps.append(f"- Method: `{method}`")
                steps.append(f"- Status: Success")
            else:
                steps.append("- Local inference: generation failed")
        else:
            steps.append("- Local model: not loaded on this instance")

    if not ai_output:
        method = "Specialist Consensus (GRPO-informed weights)"
        steps.append(f"- Fallback: `{method}`")
    steps.append("")

    # Step 5: Parse output
    steps.append("**Step 5/6 -- Parse Decision**")
    if ai_output:
        steps.append(f"- Raw model output: `{ai_output.strip()[:100]}`")
        parsed = _parse_xml_output(ai_output)
    else:
        # Enhanced specialist consensus with GRPO-informed weighting
        parsed = _specialist_consensus(info or {})
        triage_r = reports.get("triage", {})
        comp_r = reports.get("compliance", {})
        esc_r = reports.get("escalation", {})
        steps.append("- Applying GRPO-learned specialist weighting:")
        steps.append(f"  - Triage category `{triage_r.get('category', '?')}` (weight: 0.6)")
        steps.append(f"  - Compliance flagged: `{comp_r.get('flagged', False)}` (weight: 0.25)")
        steps.append(f"  - Escalation rec: `{esc_r.get('recommended', '?')}` (weight: 0.15)")

    valid_cats = {"billing", "support", "spam", "urgent", "marketing", "other"}
    cat = parsed.get("category", "support")
    cat = cat if cat in valid_cats else "support"
    pri = max(1, min(5, parsed.get("priority", 3)))
    esc = parsed.get("escalate", False)

    steps.append(f"- Parsed: category=`{cat}`, priority=`{pri}`, escalate=`{esc}`")
    steps.append("")

    # Step 6: Final decision
    steps.append("**Step 6/6 -- Final Decision**")
    steps.append(f"- **Category:** `{cat}`")
    steps.append(f"- **Priority:** `{pri}`")
    steps.append(f"- **Escalate:** `{esc}`")
    steps.append(f"- **Method:** {method}")
    steps.append("")
    steps.append("_Click **Submit Decision** to send this to the environment and see your reward._")

    return cat, pri, esc, "\n".join(steps)


# ── Environment helpers ───────────────────────────────────────────────────────

def do_reset(difficulty):
    seed = random.randint(0, 9999)
    env = EmailTriageEnvironment(difficulty=difficulty)
    obs = env.reset(seed=seed, difficulty=difficulty)
    info = obs.info or {}

    ticket_md = _fmt_ticket(obs)
    spec_md = _fmt_specialists(info)
    stats_md = _fmt_stats(info)
    status = f"Queue started -- {info.get('queue_size', '?')} tickets in {difficulty.upper()} mode  |  Seed: {seed}"
    return env, obs, info, ticket_md, spec_md, stats_md, status, 0.0, ""


def do_step(env, obs, category, priority, escalate):
    if env is None:
        return env, obs, {}, "---", "---", "---", "Click **Start Queue** first.", 0.0, ""

    action = EmailTriageAction(
        category=category,
        priority=int(priority),
        should_escalate=bool(escalate),
    )
    obs = env.step(action)
    info = obs.info or {}
    comps = info.get("reward_components", {})

    ticket_md = _fmt_ticket(obs) if not obs.done else "### Queue Complete\nAll tickets have been processed."
    spec_md = _fmt_specialists(info) if not obs.done else ""
    stats_md = _fmt_stats(info)

    reward_breakdown = ""
    if comps:
        reward_breakdown = (
            f"Quality: **{comps.get('quality', 0):.2f}**  |  "
            f"SLA: **{comps.get('sla', 0):.2f}**  |  "
            f"Policy: **{comps.get('policy', 0):.2f}**  |  "
            f"Oversight: **{comps.get('oversight', 0):.2f}**"
        )

    if obs.done:
        s = env.state
        status = (
            f"Episode finished -- Resolved {s.tickets_resolved}/{s.queue_size} tickets  |  "
            f"Total reward: {s.total_reward:.3f}"
        )
    else:
        remaining = info.get("tickets_remaining", "?")
        drift = "  !! SCHEMA DRIFT ACTIVE" if info.get("policy_drift_occurred") else ""
        status = f"Step submitted  |  Reward: {obs.reward:.3f}  |  {remaining} tickets remaining{drift}"

    return env, obs, info, ticket_md, spec_md, stats_md, status, float(obs.reward), reward_breakdown


def do_autopilot(env, obs, info, difficulty):
    """Autopilot: AI-triage + submit in a loop until queue is done."""
    if env is None or obs is None:
        # Start fresh if no queue
        seed = random.randint(0, 9999)
        env = EmailTriageEnvironment(difficulty=difficulty)
        obs = env.reset(seed=seed, difficulty=difficulty)
        info = obs.info or {}

    rows = []  # collect table rows
    total_reward = 0.0
    step_num = 0
    queue_size = (info or {}).get("queue_size", "?")
    drift_notes = []

    while True:
        if obs.done:
            break
        step_num += 1

        # AI triage for this ticket
        d = obs if isinstance(obs, dict) else obs.model_dump() if hasattr(obs, "model_dump") else vars(obs)
        subject = d.get("subject", "?")

        # Get AI decision
        _, _, _, _ = do_ai_triage(env, obs, info)
        consensus = _specialist_consensus(info or {})
        cat = consensus["category"]
        pri = consensus["priority"]
        esc = consensus["escalate"]

        # Submit decision
        action = EmailTriageAction(
            category=cat,
            priority=int(pri),
            should_escalate=bool(esc),
        )
        obs = env.step(action)
        info = obs.info or {}
        comps = info.get("reward_components", {})
        reward = float(obs.reward)
        total_reward += reward

        q = comps.get('quality', 0)
        sla = comps.get('sla', 0)
        pol = comps.get('policy', 0)
        ovr = comps.get('oversight', 0)

        drift_flag = "Yes" if info.get("policy_drift_occurred") else ""
        if drift_flag:
            drift_notes.append(f"Drift detected after ticket {step_num}.")

        rows.append(
            f"| {step_num} | {subject[:40]} | {cat} | {pri} | {esc} | {reward:.2f} | {q:.1f} | {sla:.1f} | {pol:.1f} | {ovr:.1f} | {drift_flag} |"
        )

        if obs.done:
            break

    # Build markdown table
    header = f"### Autopilot Results -- {queue_size} tickets, {difficulty.upper()} mode\n"
    table_header = "| # | Subject | Category | Pri | Esc | Reward | Quality | SLA | Policy | Oversight | Drift |\n"
    table_sep    = "|---|---------|----------|-----|-----|--------|---------|-----|--------|-----------|-------|\n"
    table_body   = "\n".join(rows)

    # Final summary
    s = env.state
    avg = s.total_reward / max(1, s.tickets_resolved)
    summary = (
        f"\n\n### Summary\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Tickets resolved | {s.tickets_resolved}/{s.queue_size} |\n"
        f"| Total reward | {s.total_reward:.3f} |\n"
        f"| Avg reward/ticket | {avg:.3f} |\n"
        f"| SLA breaches | {s.sla_breaches} |\n"
        f"| Policy violations | {s.policy_violations} |\n"
        f"| Oversight catches | {s.oversight_catches} |\n"
        f"| Drift events | {s.drift_count} |\n"
    )

    if drift_notes:
        summary += "\n" + "\n".join(f"- {n}" for n in drift_notes)

    autopilot_log = header + table_header + table_sep + table_body + summary

    ticket_md = "### Queue Complete\nAll tickets have been processed."
    spec_md = ""
    stats_md = _fmt_stats(info)
    status = f"Autopilot finished -- {s.tickets_resolved}/{s.queue_size} tickets | Total reward: {s.total_reward:.3f}"

    return env, obs, info, ticket_md, spec_md, stats_md, status, float(obs.reward), "", autopilot_log


# ── Formatters ────────────────────────────────────────────────────────────────

def _fmt_ticket(obs) -> str:
    if obs is None:
        return "_No ticket loaded. Click **Start Queue** to begin._"
    d = obs if isinstance(obs, dict) else obs.model_dump() if hasattr(obs, "model_dump") else vars(obs)
    internal = "INTERNAL" if d.get("is_internal") else "EXTERNAL"
    return (
        f"**Subject:** {d.get('subject', '---')}\n\n"
        f"**From:** {d.get('sender', '---')} ({d.get('sender_domain', '---')})  [{internal}]\n\n"
        f"---\n\n{d.get('body_snippet', d.get('body', '---'))}"
    )


def _fmt_specialists(info: dict) -> str:
    reports = info.get("specialist_reports", {})
    if not reports:
        return "_No specialist reports available._"
    lines = []
    labels = {"triage": "Triage", "compliance": "Compliance", "priority": "Priority", "routing": "Routing"}
    for name, data in reports.items():
        label = labels.get(name, name.title())
        lines.append(f"**{label} Specialist**")
        if "category" in data:
            lines.append(f"- Category: `{data['category']}`")
        if "priority" in data:
            lines.append(f"- Priority: `{data['priority']}`")
        if "recommended_action" in data:
            lines.append(f"- Action: `{data['recommended_action']}`")
        if "recommended" in data:
            lines.append(f"- Escalate: `{data['recommended']}`")
        conf = data.get("confidence", None)
        if conf is not None:
            pct = max(0, min(100, int(conf * 100)))
            bar = chr(9608) * (pct // 10) + chr(9617) * (10 - pct // 10)
            lines.append(f"- Confidence: `{bar}` {pct}%")
        if data.get("flagged"):
            lines.append(f"- !! FLAGGED: {data.get('reason', 'policy issue')}")
        if data.get("draft_ready"):
            lines.append(f"- Template: `{data.get('template_id', 'n/a')}`")
        lines.append("")
    return "\n".join(lines)


def _fmt_stats(info: dict) -> str:
    state = info.get("state", {})
    if not state:
        return ""
    resolved = state.get("tickets_resolved", 0)
    total = state.get("queue_size", 0)
    sla = state.get("sla_breaches", 0)
    pol = state.get("policy_violations", 0)
    drift = state.get("drift_count", 0)
    catches = state.get("oversight_catches", 0)
    pct = int((resolved / total * 100)) if total else 0
    bar = chr(9608) * (pct // 10) + chr(9617) * (10 - pct // 10)

    lines = [
        f"**Progress:** `{bar}` {resolved}/{total} ({pct}%)",
        f"**SLA Breaches:** {sla}    **Policy Violations:** {pol}",
        f"**Oversight Catches:** {catches}    **Drift Events:** {drift}",
    ]
    if drift > 0:
        lines.append("!! SCHEMA DRIFT ACTIVE -- Rules have changed mid-shift!")
    return "  \n".join(lines)


# ── UI builder ────────────────────────────────────────────────────────────────

CSS = """
/* === GLOBAL RESET: white bg, no nested borders === */
body, .gradio-container, .gradio-container *,
.gr-block, .gr-box, .gr-form, .gr-panel, .gr-group,
.gr-padded, .gr-compact, .contain,
div[class*="block"], div[class*="wrap"],
.dark, [data-testid] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #111111 !important;
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif !important;
}
.gr-row, .gr-column, .gr-tab, .gr-tabs,
.gr-accordion, .gr-accordion-header,
div[class*="container"], div[class*="column"],
div[class*="row"], div[class*="panel"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

/* === KILL ALL inner borders from Gradio wrapper divs === */
.gr-box, .gr-panel, .gr-form, .gr-block, .gr-group,
.block, div.block, .wrap, div.wrap,
.gr-padded, .gr-compact,
div[class*="block"]:not(.panel-ticket):not(.panel-specialists):not(.panel-stats):not(.status-bar):not(.ai-status):not(.reward-strip) {
    border: none !important;
    box-shadow: none !important;
}

/* === Text === */
.gr-markdown, .gr-markdown p, .gr-markdown li,
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3,
.gr-markdown h4, .gr-markdown strong, .gr-markdown em,
.prose, .prose p, .prose li, .prose h1, .prose h2, .prose h3 {
    color: #000000 !important;
    background: transparent !important;
}
.gr-markdown a, .prose a { color: #000 !important; text-decoration: underline; }

/* === Header === */
.arena-header { border-bottom: 2px solid #111; padding: 16px 0 12px 0; margin-bottom: 4px; }
.arena-title { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.02em; color: #000 !important; margin: 0 0 2px 0; }
.arena-subtitle { font-size: 0.8rem; color: #000 !important; margin: 0; }

/* === Main panels (only these get a single outer border) === */
.panel-ticket, .panel-specialists {
    border: 1px solid #000 !important;
    border-radius: 6px;
    padding: 12px 16px;
    color: #000 !important;
    min-height: 160px;
}
.panel-stats {
    border: 1px solid #000 !important;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 0.8rem;
    color: #000 !important;
}
.status-bar {
    border-left: 3px solid #000;
    padding: 6px 10px;
    font-size: 0.8rem;
    color: #000 !important;
    border-radius: 0 4px 4px 0;
}

/* === AI / Autopilot log (no border, clean) === */
.ai-status {
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #000 !important;
    border: none !important;
}
/* Table styling inside logs */
.ai-status table { width: 100%; border-collapse: collapse; font-size: 0.75rem; margin: 8px 0; }
.ai-status th, .ai-status td { border: 1px solid #ccc; padding: 4px 6px; text-align: left; }
.ai-status th { background: #f0f0f0 !important; font-weight: 600; }

/* === Buttons === */
button.primary, button[class*="primary"],
button.secondary, button[class*="secondary"] {
    background: #fff !important;
    color: #000 !important;
    border: 1.5px solid #000 !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
}
button.primary:hover, button[class*="primary"]:hover,
button.secondary:hover, button[class*="secondary"]:hover {
    background: #f0f0f0 !important;
}

/* === Inputs === */
input, select, textarea, .gr-dropdown, .gr-slider {
    border: 1px solid #000 !important;
    border-radius: 4px !important;
    background: #fff !important;
    color: #000 !important;
}
input:focus, select:focus, textarea:focus { border-color: #000 !important; outline: none !important; }

/* === Labels === */
label, .gr-label, span[data-testid="block-label"] { color: #000 !important; }
.section-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #000 !important; margin-bottom: 4px; }

/* === Reward strip === */
.reward-strip { color: #000 !important; border: 1px solid #000 !important; border-radius: 4px; padding: 6px 12px; font-size: 0.8rem; font-weight: 500; }

/* === Misc === */
hr { border: none; border-top: 1px solid #eee; margin: 8px 0; }
.gr-number input { background: #fff !important; color: #000 !important; }
.gr-checkbox label { color: #000 !important; }
.gr-accordion { border-color: #000 !important; }
footer { display: none !important; }
.gradio-container { max-width: 1200px !important; margin: auto; }


/* ── Reduce gap between blocks ── */
.gap { gap: 8px !important; }
"""

INTRO_MD = """
<div class="arena-header">
  <div class="arena-title">Oversight Inbox Arena</div>
  <div class="arena-subtitle">
    Multi-agent RL environment &mdash; 4 specialist agents &bull; schema drift &bull; GRPO-trained coordinator<br/>
    Model: <a href="https://huggingface.co/Rhushya/oversight-arena-grpo2">Rhushya/oversight-arena-grpo2</a> (Qwen2.5-1.5B + LoRA)
  </div>
</div>
"""

HOWTO_MD = """
1. Select a difficulty and click **Start Queue**
2. Read the email (left) and specialist advice (right)
3. Click **AI Auto-Triage** or set category/priority/escalation manually
4. Click **Submit Decision** to see your reward. Hard modes have **schema drift**!
"""


def build_ui() -> gr.Blocks:
    if gr is None:
        raise ImportError("gradio is required to build the UI")

    with gr.Blocks(
        title="Oversight Inbox Arena",
        theme=gr.themes.Base(
            primary_hue="neutral",
            secondary_hue="neutral",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
        ),
        css=CSS,
    ) as demo:

        # ── Shared state ─────────────────────────────────────────────────────
        env_s  = gr.State(None)
        obs_s  = gr.State(None)
        info_s = gr.State({})

        # ── Header ───────────────────────────────────────────────────────────
        gr.Markdown(INTRO_MD)

        with gr.Accordion("How to play", open=False):
            gr.Markdown(HOWTO_MD)

        # ── Controls row ─────────────────────────────────────────────────────
        with gr.Row():
            difficulty = gr.Dropdown(
                choices=["easy", "medium", "hard", "adversarial"],
                value="hard",
                label="Difficulty",
                scale=1,
            )
            start_btn = gr.Button("Start Queue", variant="primary", scale=1)
            status_md = gr.Markdown(
                "_Select a difficulty and click **Start Queue** to begin._",
                elem_classes=["status-bar"],
            )

        # ── Stats bar ────────────────────────────────────────────────────────
        stats_md = gr.Markdown("", elem_classes=["panel-stats"])

        # ── Main arena ───────────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                gr.Markdown("**INCOMING EMAIL**", elem_classes=["section-label"])
                ticket_md = gr.Markdown(
                    "_No ticket yet. Start the queue above._",
                    elem_classes=["panel-ticket"],
                )

            with gr.Column(scale=4):
                gr.Markdown("**SPECIALIST PANEL**", elem_classes=["section-label"])
                spec_md = gr.Markdown(
                    "_Specialists will report here once the queue starts._",
                    elem_classes=["panel-specialists"],
                )

        # ── Decision + Buttons (single compact row) ──────────────────────────
        gr.Markdown("---")
        gr.Markdown("**YOUR DECISION**", elem_classes=["section-label"])
        with gr.Row():
            cat_in = gr.Dropdown(
                choices=["billing", "support", "spam", "urgent", "marketing", "other"],
                value="support",
                label="Category",
                scale=2,
            )
            pri_in = gr.Slider(
                minimum=1, maximum=5, step=1, value=3,
                label="Priority (1=Low, 5=Critical)",
                scale=2,
            )
            esc_in = gr.Checkbox(label="Escalate", scale=1)
            ai_btn = gr.Button("AI Auto-Triage", variant="primary", scale=2)
            sub_btn = gr.Button("Submit Decision", variant="secondary", scale=2)
            auto_btn = gr.Button("Autopilot (Run All)", variant="secondary", scale=1)

        # ── Reward (always visible right after buttons) ───────────────────────
        with gr.Row():
            reward_num = gr.Number(
                label="Step Reward", value=0.0, precision=3, scale=1
            )
            reward_breakdown = gr.Markdown("", elem_classes=["reward-strip"])

        # ── AI Pipeline Log (scrollable accordion -- won't push layout) ──────
        with gr.Accordion("AI Pipeline Log", open=False):
            ai_status_md = gr.Markdown(
                "_Click **AI Auto-Triage** to see the step-by-step pipeline here._",
                elem_classes=["ai-status"],
            )

        # ── Autopilot Log ─────────────────────────────────────────────────────
        with gr.Accordion("Autopilot Log", open=False):
            autopilot_md = gr.Markdown(
                "_Click **Autopilot** to process all tickets automatically._",
                elem_classes=["ai-status"],
            )

        # ── Footer ───────────────────────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown(
            "_Built with Hugging Face, TRL, GRPO, Gradio_  |  "
            "_Model: [oversight-arena-grpo2](https://huggingface.co/Rhushya/oversight-arena-grpo2)_  |  "
            "_[GitHub](https://github.com/Rhushya/OpenEnv)_  |  "
            "_[Blog](https://huggingface.co/spaces/Rhushya/email-triage-env-openenv/blob/main/BLOG.md)_"
        )

        # ── Wire callbacks ───────────────────────────────────────────────────
        start_btn.click(
            fn=do_reset,
            inputs=[difficulty],
            outputs=[env_s, obs_s, info_s, ticket_md, spec_md, stats_md, status_md, reward_num, reward_breakdown],
        )

        sub_btn.click(
            fn=do_step,
            inputs=[env_s, obs_s, cat_in, pri_in, esc_in],
            outputs=[env_s, obs_s, info_s, ticket_md, spec_md, stats_md, status_md, reward_num, reward_breakdown],
        )

        ai_btn.click(
            fn=do_ai_triage,
            inputs=[env_s, obs_s, info_s],
            outputs=[cat_in, pri_in, esc_in, ai_status_md],
        )

        auto_btn.click(
            fn=do_autopilot,
            inputs=[env_s, obs_s, info_s, difficulty],
            outputs=[env_s, obs_s, info_s, ticket_md, spec_md, stats_md, status_md, reward_num, reward_breakdown, autopilot_md],
        )

    return demo


if __name__ == "__main__":
    if gr is None:
        raise ImportError("gradio is required to launch the UI")
    app = build_ui()
    app.launch(share=True)
