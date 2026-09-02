"""Human-facing UI for a Harbor env server.

Two columns: the LLM on the left, the task on the right. Validate, pick, run.

Status text is deliberately terse. The long explanations belong in docs — what a person needs on
screen is whether it will work, what got rewritten, and which sandboxes are usable.

Validation is a gate, not a hint: an LLM endpoint without token-id capture answers every request
normally and returns nothing trainable, so a rollout looks perfect and is worthless.

Rich output (the rollout graph, per-turn tokens) is rendered as HTML rather than Gradio widgets,
because a conversation tree with branches and discarded retries is a shape, and a dataframe cannot
show a shape.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any

import gradio as gr

_UNVALIDATED = "_Enter your LLM URL and press Validate._"

_CSS = """
.hb-wrap { max-width: 1400px; margin: 0 auto; }
.hb-card { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 14px 16px; }
.hb-dim   { opacity: .6; }
.hb-kv    { display: flex; gap: 22px; flex-wrap: wrap; margin: 4px 0 2px; }
.hb-kv b  { font-variant-numeric: tabular-nums; }

/* The two panels read as one undifferentiated wall of controls without a boundary; the border is
   what makes "pick a model" and "pick a task" look like two separate decisions. */
.hb-cell  { border: 1px solid var(--border-color-primary); border-radius: 10px;
            padding: 14px 16px; }
.hb-cell + .hb-cell { margin-left: 12px; }

/* Live conversation. Roles are colour-coded down the left edge so the shape of the loop
   (assistant calls a tool, tool answers, assistant calls again) is readable at a glance. */
/* No max-height here. A fixed-height scroll box nests a second scroller inside the page:
   the wheel gets captured while the pointer is over the conversation, and the page stops
   growing so there is nothing left to scroll to. Let it run at natural height and let the
   page do the scrolling. Length is bounded by the message cap, not by CSS. */
.hb-tx    { margin-top: 10px; }
.hb-msg   { border-left: 3px solid var(--border-color-primary); padding: 6px 0 6px 10px;
            margin: 8px 0; font-size: 13px; line-height: 1.45; }
.hb-msg pre { white-space: pre-wrap; word-break: break-word; margin: 4px 0 0;
              font-size: 12px; opacity: .85; }
.hb-role  { display: inline-block; font-size: 11px; text-transform: uppercase;
            letter-spacing: .04em; opacity: .65; margin-bottom: 2px; }
.hb-assistant { border-left-color: #22c55e; }
.hb-tool      { border-left-color: #38bdf8; }
.hb-user      { border-left-color: #a78bfa; }
.hb-system    { border-left-color: #94a3b8; opacity: .75; }
.hb-tc    { margin-top: 4px; padding: 4px 8px; border-radius: 6px;
            background: var(--background-fill-secondary); }
.hb-tc    { display: block; }
.hb-tc b  { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.hb-arrow { opacity: .5; margin-right: 6px; }
.hb-tr    { margin-top: 4px; padding: 4px 8px; border-radius: 6px; border-left: 2px solid #38bdf8;
            background: var(--background-fill-secondary); }
/* No inner scroller here either, for the same reason as the conversation above, and the previous
   version of this rule was the bug: `overscroll-behavior: contain` does not stop a box from
   swallowing the page scroll, it is what *prevents* the wheel from chaining to the page once the
   box reaches its own end. Tool output is clipped to 500 characters server side, but 500
   characters of shell output is 25 short lines, which overflowed the 220px cap and left the page
   feeling frozen wherever the pointer happened to be. Length is bounded by the clip, not by CSS. */
.hb-tr pre{ margin: 0; font-size: 11.5px; opacity: .8; }

/* A run in flight should look like one. */
.hb-live  { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.hb-pulse { width: 8px; height: 8px; border-radius: 50%; background: #22c55e;
            animation: hb-blink 1.2s ease-in-out infinite; }
@keyframes hb-blink { 0%, 100% { opacity: 1; } 50% { opacity: .25; } }
.hb-drop-msg { opacity: .5; border-left-color: #ef4444; }

/* Verdict. The outcome should be legible from across the room; the numbers behind it should not
   compete with it for attention. */
.hb-verdict { border-left-width: 4px; }
.hb-head  { font-size: 17px; font-weight: 650; margin-bottom: 8px; }
.hb-good  { border-left-color: #22c55e; }
.hb-warn  { border-left-color: #f59e0b; }
.hb-bad   { border-left-color: #ef4444; }
.hb-err   { white-space: pre-wrap; word-break: break-word; font-size: 12px; margin: 10px 0 0;
            padding: 8px 10px; border-radius: 6px; background: var(--background-fill-secondary); }

/* A qualifier on the result: true, load-bearing, and not an error. Bordered rather than coloured
   like a finding, so "this rollout is eval-only" does not read as "this rollout failed". */
.hb-note  { font-size: 12.5px; line-height: 1.5; margin: 10px 0 0; padding: 8px 11px;
            border-radius: 6px; border: 1px solid var(--border-color-primary);
            background: var(--background-fill-secondary); }
.hb-note code { font-size: 11.5px; }

/* Hover explanations. `data-tip` rather than `title=` for the two long ones: the native tooltip
   truncates, takes a second to appear, and cannot wrap a paragraph. Short hints use Gradio's own
   `info=`, which renders under the label and needs no hover at all. */
.hb-i     { display: inline-flex; align-items: center; justify-content: center; cursor: help;
            width: 15px; height: 15px; margin-left: 6px; border-radius: 50%; font-size: 10px;
            font-weight: 700; font-style: normal; vertical-align: 1px;
            border: 1px solid var(--border-color-primary); opacity: .75; position: relative; }
.hb-i:hover { opacity: 1; }
.hb-i::after { content: attr(data-tip); position: absolute; left: 50%; bottom: 130%;
            transform: translateX(-50%); width: max-content; max-width: 320px; padding: 8px 10px;
            border-radius: 6px; border: 1px solid var(--border-color-primary);
            background: var(--background-fill-primary); color: var(--body-text-color);
            font-size: 11.5px; font-weight: 400; line-height: 1.5; text-align: left;
            white-space: pre-line; opacity: 0; visibility: hidden; transition: opacity .12s;
            z-index: 40; box-shadow: 0 4px 14px rgba(0,0,0,.18); }
.hb-i:hover::after { opacity: 1; visibility: visible; }
/* The label row the icon sits on, so the icon lines up with a Gradio label rather than floating. */
.hb-lbl   { display: flex; align-items: center; font-size: 13px; font-weight: 600;
            margin: 2px 0 -6px; }

/* Findings carry severity: a FATAL means unusable, a WARN means read before training on it. */
.hb-find  { font-size: 12.5px; margin: 5px 0; line-height: 1.45; }
.hb-tag   { display: inline-block; min-width: 46px; margin-right: 8px; padding: 1px 6px;
            border-radius: 4px; font-size: 10px; font-weight: 700; letter-spacing: .04em;
            text-align: center; vertical-align: 1px; }
.hb-fatal .hb-tag { background: #ef4444; color: #fff; }
.hb-warn2 .hb-tag { background: #f59e0b; color: #1f2937; }
.hb-info  .hb-tag { background: var(--background-fill-secondary); opacity: .7; }
.hb-info  { opacity: .7; }

/* Turn table: dense, aligned, and the numbers read as numbers. */
.hb-tbl   { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
.hb-tbl th{ text-align: left; font-weight: 600; font-size: 11px; text-transform: uppercase;
            letter-spacing: .04em; opacity: .55; padding: 4px 10px 6px 0;
            border-bottom: 1px solid var(--border-color-primary); }
.hb-tbl td{ padding: 7px 10px 7px 0; border-bottom: 1px solid var(--border-color-primary);
            vertical-align: top; }
.hb-tbl code { font-size: 12px; padding: 1px 6px; border-radius: 4px;
               background: var(--background-fill-secondary); }
.hb-num   { font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap;
            padding-right: 14px !important; }
.hb-prev  { margin-top: 3px; font-size: 12px; }
.hb-drop-row { opacity: .45; }
.hb-drop-tag { background: #ef4444; color: #fff; }
.hb-conf  { display: inline-block; width: 76px; height: 7px; border-radius: 4px;
            background: var(--background-fill-secondary); overflow: hidden; vertical-align: middle; }
.hb-conf span { display: block; height: 100%; }

/* Each conversation folds away; the main one starts open. */
.hb-convo { margin-top: 10px; border-top: 1px solid var(--border-color-primary); padding-top: 8px; }
.hb-convo summary { cursor: pointer; padding: 4px 0; }

/* Setup, before the agent has said anything. */
.hb-steps { margin: 8px 0 0; }
.hb-step  { display: flex; align-items: center; gap: 9px; padding: 3px 0; font-size: 13px; }
.hb-dot   { width: 7px; height: 7px; border-radius: 50%; background: var(--border-color-primary); }
.hb-step.done .hb-dot { background: #22c55e; }
.hb-step.now  .hb-dot { background: #f59e0b; animation: hb-blink 1.2s ease-in-out infinite; }
.hb-step.todo { opacity: .45; }

/* The outcome, at a glance. */
.hb-hero  { display: flex; align-items: center; justify-content: space-between; gap: 20px;
            padding-bottom: 12px; margin-bottom: 4px;
            border-bottom: 1px solid var(--border-color-primary); }
.hb-badge { display: inline-flex; align-items: center; gap: 9px; font-size: 19px;
            font-weight: 700; letter-spacing: -.01em; }
.hb-mark  { display: inline-flex; align-items: center; justify-content: center;
            width: 30px; height: 30px; border-radius: 50%; font-size: 15px; color: #fff; }
.hb-b-good .hb-mark { background: #22c55e; }
.hb-b-warn .hb-mark { background: #f59e0b; }
.hb-b-bad  .hb-mark { background: #ef4444; }
.hb-score   { text-align: right; line-height: 1.05; }
.hb-score-v { font-size: 42px; font-weight: 700; font-variant-numeric: tabular-nums;
              letter-spacing: -.02em; }
.hb-score-c { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; opacity: .55; }
.hb-kv-big span   { font-size: 11px; text-transform: uppercase; letter-spacing: .04em;
                    opacity: .55; }
.hb-kv-big b      { display: block; font-size: 19px; margin-top: 3px; text-transform: none;
                    letter-spacing: normal; opacity: 1; }
.hb-kv-big .hb-key b { color: var(--body-text-color); }
.hb-kv-big .hb-key   { opacity: .85; }

footer { display: none !important; }
"""


def _labelled(label: str, tip: str) -> str:
    """A field label with a hover-explained `i` beside it.

    For the explanations too long to sit under a Gradio label as `info=` text — which is where every
    one-liner belongs instead, since it needs no hover to be seen.
    """
    return (
        f'<div class="hb-lbl">{html.escape(label)}'
        f'<span class="hb-i" data-tip="{html.escape(tip, quote=True)}">i</span></div>'
    )


_KEY_TIP = (
    "Only needed for a hosted endpoint: OpenAI, Anthropic, HF Inference Providers.\n\n"
    "It is sent to the inference endpoint by this server and nothing else. It is NOT the key the "
    "agent receives — that one is a capture session id, minted per rollout, which is how one proxy "
    "serves many rollouts and how an unregistered caller is rejected.\n\n"
    "Leave empty for a local vLLM or SGLang."
)

_LEVEL_TIP = (
    "There are two kinds of rollout, and the endpoint decides which you get.\n\n"
    "TRAIN needs the engine to return token ids and per-token logprobs: vLLM started with "
    "--return-tokens-as-token-ids --logprobs-mode processed_logprobs, or SGLang built from git "
    "main. You get the reward, the trace, and the exact tokens and logprobs to train on.\n\n"
    "EVAL is everything else, including a vLLM started without those flags. You get the reward and "
    "the full trace; there are no token ids, so nothing is trainable. Logprobs alone do not help — "
    "with no ids to pair them with there is nothing to align them to."
)


def _clip(text: Any, limit: int = 400) -> str:
    """Escape and shorten a value for display, keeping the head where the meaning usually is."""
    body = text if isinstance(text, str) else json.dumps(text, default=str)
    body = body.strip()
    return html.escape(body[:limit]) + ("…" if len(body) > limit else "")


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Tool calls on a message, normalised across all four dialects.

    Chat-completions puts them in `tool_calls`; Anthropic puts them in the content block list as
    `tool_use`. Reading only the former shows claude-code as a stream of text with no visible
    actions, which is exactly the case the live view exists to make visible.
    """
    out: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        name = function.get("name") or call.get("name")
        if name:
            out.append(
                {
                    "name": str(name),
                    "arguments": function.get("arguments", call.get("arguments", "")),
                }
            )
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name")
            ):
                out.append(
                    {"name": str(block["name"]), "arguments": block.get("input", "")}
                )
    return out


def _message_text(message: dict[str, Any]) -> str:
    """Readable text of a message, ignoring tool-call and tool-result blocks."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("tool_use", "tool_result"):
                continue
            if block.get("text"):
                parts.append(str(block["text"]))
        return " ".join(parts)
    return ""


def _tool_results(message: dict[str, Any]) -> list[str]:
    """What came back from a tool, in either the chat-completions or the Anthropic shape."""
    if message.get("role") == "tool":
        return [
            _message_text(message) or json.dumps(message.get("content"), default=str)
        ]
    content = message.get("content")
    if not isinstance(content, list):
        return []
    out = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            body = block.get("content")
            if isinstance(body, list):
                body = " ".join(b.get("text", "") for b in body if isinstance(b, dict))
            out.append(str(body if body is not None else ""))
    return out


def _render_calls(calls: list[dict[str, Any]]) -> str:
    return "".join(
        f'<div class="hb-tc"><span class="hb-arrow">▸</span>'
        f"<b>{html.escape(str(c.get('name', 'tool')))}</b>"
        f"<pre>{_clip(c.get('arguments', ''), 600)}</pre></div>"
        for c in calls
    )


def _render_message(message: dict[str, Any], *, label: str = "") -> str:
    """One row of the conversation: who spoke, what they said, what they invoked or returned."""
    role = str(message.get("role", "?"))
    calls = _tool_calls(message)
    results = _tool_results(message)
    text = _message_text(message)

    # A user message carrying only tool results is the tool speaking, not the user; labelling it
    # "user" makes the agent look like it is being prompted between every action.
    shown_role = "tool" if results and role != "assistant" else role
    # For a `role: tool` message the content IS the result, so rendering both duplicates it.
    if shown_role == "tool":
        text = ""
    body = _clip(text, 700 if shown_role in ("user", "system") else 450) if text else ""
    blocks = "".join(
        f'<div class="hb-tr"><pre>{_clip(r, 500)}</pre></div>' for r in results
    )
    if not body and not blocks and not calls:
        return ""
    return (
        f'<div class="hb-msg hb-{html.escape(shown_role)}">'
        f'<span class="hb-role">{html.escape(label or shown_role)}</span>'
        + (f"<div>{body}</div>" if body else "")
        + _render_calls(calls)
        + blocks
        + "</div>"
    )


def _transcript_html(session: Any) -> str:
    """The conversation as it stands right now: what the agent said, called, and got back.

    Counters answer "is it alive"; this answers "is it doing the right thing", which is the question
    worth asking while a rollout is still running. The newest turn's `request_messages` already holds
    the whole conversation the harness assembled, tool results included, so rendering that plus the
    latest response needs no reconstruction from deltas.
    """
    nodes = sorted(session.graph.nodes(), key=lambda n: n.index)
    if not nodes:
        return ""
    latest = nodes[-1]

    rows = [
        row
        for row in (_render_message(m) for m in (latest.request_messages or []))
        if row
    ]

    response = latest.response_message or {}
    tail = _render_message(
        {**response, "role": "assistant"},
        label=f"assistant · turn {latest.index} · generating",
    )
    if tail:
        rows.append(tail)

    # Only the tail is ever new, so cap from the front and say what was dropped.
    shown = rows[-18:]
    elided = (
        f'<div class="hb-dim">… {len(rows) - len(shown)} earlier message(s)</div>'
        if len(rows) > len(shown)
        else ""
    )
    # Count across the conversation, not just the response messages: Anthropic carries tool use in
    # the assistant content blocks the harness replays back, so a response-only tally reads 0.
    calls_so_far = sum(
        len(_tool_calls(m)) for m in (latest.request_messages or [])
    ) + len(_tool_calls(latest.response_message or {}))
    return (
        f'<div class="hb-card hb-tx"><div class="hb-live">'
        f'<span class="hb-pulse"></span><b>Live conversation</b>'
        f'<span class="hb-dim">turn {latest.index} · {calls_so_far} tool call(s) so far · '
        f"{latest.n_tools} tool(s) offered</span></div>{elided}{''.join(shown)}</div>"
    )


# What happens before the agent's first model call, in order. Harbor exposes no progress hook, so
# the stage is inferred from what capture has seen: no session means the trial has not reached the
# agent yet, a session with no turns means the agent is installed and starting up.
_SETUP_STEPS = (
    "creating the sandbox",
    "uploading the task",
    "installing the agent",
    "waiting for the first model call",
)


def _steps_html(stage: int) -> str:
    """The setup sequence, with the current stage marked."""
    rows = []
    for i, label in enumerate(_SETUP_STEPS):
        cls = "done" if i < stage else ("now" if i == stage else "todo")
        rows.append(
            f'<div class="hb-step {cls}"><span class="hb-dot"></span>'
            f"<span>{html.escape(label)}</span></div>"
        )
    return f'<div class="hb-steps">{"".join(rows)}</div>'


def _live_html(
    harness: str,
    sandbox: str,
    phase: str,
    elapsed: float,
    stats: dict[str, Any] | None,
    stage: int = -1,
) -> str:
    """The running header: what is running, how far in, and what it has produced so far."""
    bits = [
        f'<div class="hb-card hb-verdict hb-warn">'
        f'<div class="hb-live"><span class="hb-pulse"></span>'
        f"<b>Running</b> <code>{html.escape(harness)}</code> on "
        f"<code>{html.escape(sandbox)}</code>"
        f'<span class="hb-dim">{html.escape(phase)} · {elapsed:.0f}s</span></div>'
    ]
    # Before the first call there are no numbers worth showing, so show progress instead. A row of
    # zeros for a minute reads as "stuck" when the sandbox is simply still booting.
    if stage >= 0:
        bits.append(_steps_html(stage))
    if stats:
        bits.append(
            '<div class="hb-kv">'
            + "".join(f"<span>{k}<br><b>{v}</b></span>" for k, v in stats.items())
            + "</div>"
        )
    bits.append("</div>")
    return "".join(bits)


# `warn` is already a verdict tone; the finding variant needs its own class name.
_FINDING_CLASS = {"FATAL": "fatal", "WARN": "warn2", "INFO": "info"}


def _findings_html(findings: list[str]) -> str:
    """Findings, grouped by how much they should worry you.

    They were previously all rendered the same dim grey and truncated to 220 characters, which put
    "the intercept saw no model calls" and "3 roots across 7 turns" at equal weight. A FATAL means
    the rollout is unusable; a WARN means read it before training on it.
    """
    if not findings:
        return ""
    buckets: dict[str, list[str]] = {"FATAL": [], "WARN": [], "INFO": []}
    for raw in findings:
        level = (
            "FATAL"
            if raw.startswith("[FATAL")
            else "WARN"
            if raw.startswith("[WARN")
            else "INFO"
        )
        buckets[level].append(
            raw.split("]", 1)[-1].strip() if raw.startswith("[") else raw
        )

    out = []
    for level, items in buckets.items():
        for item in items:
            out.append(
                f'<div class="hb-find hb-{_FINDING_CLASS[level]}">'
                f'<span class="hb-tag">{level}</span>{html.escape(item[:400])}</div>'
            )
    return "".join(out)


def _result_html(r: dict[str, Any]) -> str:
    """The verdict, the numbers behind it, and anything that qualifies it.

    The outcome is the one thing every reader wants first, so the reward is set at display size and
    the supporting counts are deliberately quieter. Getting that hierarchy wrong is how a failed
    rollout reads as a successful one at a glance.
    """
    reward = r.get("reward")
    if not r.get("ok"):
        tone, mark, label = "bad", "✕", "Failed"
        value, caption = "—", str(r.get("exception_type") or "error")
    elif reward is None:
        # Not a zero. The verifier never ran, so this says nothing about the model.
        tone, mark, label = "warn", "!", "Not graded"
        value, caption = "—", "the verifier never ran"
    elif reward > 0:
        tone, mark, label = "good", "✓", "Solved"
        value, caption = f"{reward:.2f}", "reward"
    else:
        tone, mark, label = "warn", "○", "Not solved"
        value, caption = f"{reward:.2f}", "reward"

    turns = r.get("turns") or []
    generated = sum(len(t.get("completion_token_ids") or []) for t in turns)
    dropped = sum(
        len(t.get("completion_token_ids") or []) for t in turns if t.get("discarded")
    )
    tools = sum(len(t.get("tool_calls") or []) for t in turns)
    atif = r.get("atif", "none")

    # `key` marks the figures that decide whether this rollout is usable, as opposed to describing it.
    # The initial prompt: task instruction plus the harness's system prompt and tool manifest.
    # Constant across turns, so it is a property of the rollout rather than a per-row column.
    context = len((turns[0].get("prompt_token_ids") or [])) if turns else 0

    is_eval = r.get("rollout_type", "train") == "eval"
    kv = [
        # "trainable tokens: 0" on an eval rollout reads as a capture failure. It is not one, so the
        # slot says what kind of rollout this is instead of reporting a zero that means nothing here.
        ("rollout", f"EVAL · {r.get('capture_level', '?')}", True)
        if is_eval
        else ("trainable tokens", f"{r.get('n_trainable_tokens', 0):,}", True),
        ("context", f"{context:,}", False),
        ("trace check", atif, atif != "match"),
        ("model calls", r.get("n_turns", 0), False),
        ("tool calls", tools, False),
        ("conversations", r.get("n_roots", 0), False),
        (
            "generated",
            f"{generated:,}" + (f" · {dropped:,} discarded" if dropped else ""),
            False,
        ),
        ("wall", f"{r.get('wall_s', 0):.0f}s", False),
    ]

    out = [
        f'<div class="hb-card hb-verdict hb-{tone}">',
        '<div class="hb-hero">',
        f'<div class="hb-badge hb-b-{tone}"><span class="hb-mark">{mark}</span>'
        f"<span>{html.escape(label)}</span></div>",
        f'<div class="hb-score"><div class="hb-score-v">{html.escape(value)}</div>'
        f'<div class="hb-score-c">{html.escape(caption)}</div></div>',
        "</div>",
        '<div class="hb-kv hb-kv-big">'
        + "".join(
            f'<span class="{"hb-key" if key else ""}">{k}<br><b>{v}</b></span>'
            for k, v, key in kv
        )
        + "</div>",
    ]

    if is_eval:
        out.append(
            '<div class="hb-note">This is an <b>eval rollout</b>. The endpoint returned '
            f"{'logprobs but no token ids' if r.get('capture_level') == 'logprobs' else 'no token ids and no logprobs'}, "
            "so you get the reward and the full trace below, but nothing trainable — there is no "
            "<code>contract.json</code> and no per-token logprobs. Point the server at vLLM "
            "(<code>--return-tokens-as-token-ids --logprobs-mode processed_logprobs</code>) or "
            "SGLang built from main for trainable rollouts.</div>"
        )
    for fix in r.get("param_fixes") or []:
        out.append(
            f'<div class="hb-note hb-dim">upstream compatibility: {html.escape(fix)} — '
            "the request differs from what the harness asked for.</div>"
        )

    rewards = r.get("rewards") or {}
    if len(rewards) > 1:
        chosen = r.get("reward_key", "")
        parts = [
            f"<b>{html.escape(k)}</b> {v:.3f}" + (" ←" if k == chosen else "")
            for k, v in sorted(rewards.items())
        ]
        out.append(f'<div class="hb-kv hb-dim">{" &nbsp; ".join(parts)}</div>')

    for step in r.get("step_results") or []:
        vals = ", ".join(f"{k}={v:.2f}" for k, v in (step.get("rewards") or {}).items())
        out.append(
            f'<div class="hb-dim">step <b>{html.escape(step.get("name", ""))}</b> {vals}</div>'
        )

    if r.get("error"):
        out.append(f'<pre class="hb-err">{html.escape(str(r["error"])[:1200])}</pre>')
    if r.get("agent_log_tail"):
        out.append(
            '<details class="hb-dim"><summary>agent log</summary>'
            f"<pre>{html.escape(str(r['agent_log_tail'])[:4000])}</pre></details>"
        )

    out.append(_findings_html(r.get("findings") or []))
    out.append(
        '<div class="hb-dim" style="margin-top:10px">Capture quality and reward are '
        "independent: a perfectly captured rollout can still score 0 because the model was "
        "wrong, and reward <b>—</b> means the verifier never ran at all.</div></div>"
    )
    return "".join(out)


def _conversation_html(r: dict[str, Any]) -> str:
    """The whole conversation as it was actually sent: system prompt, tools, results, replies.

    Rebuilt from the result rather than the live session, so it survives the run. Several are
    possible: each root is a separate conversation, and an auxiliary one (a next-speaker check, a
    summariser) is labelled as such so it is not mistaken for the agent working on the task.
    """
    conversations = r.get("conversations") or []
    if not conversations:
        return ""

    agents = [c for c in conversations if c.get("role", "agent") == "agent"]
    blocks = []
    seen_agents = 0
    for i, convo in enumerate(conversations):
        role = convo.get("role", "agent")
        if role == "agent":
            seen_agents += 1
            # Numbered when there is more than one, so two blocks are never both "main".
            badge = (
                "main conversation"
                if len(agents) == 1
                else f"conversation {seen_agents} of {len(agents)}"
            )
        else:
            badge = {
                "auxiliary": "auxiliary call",
                "discarded": "discarded branch",
            }.get(role, role)
        rows = [
            row
            for row in (_render_message(m) for m in convo.get("messages") or [])
            if row
        ]
        if not rows:
            continue
        blocks.append(
            f'<details class="hb-convo" {"open" if role == "agent" and i == 0 else ""}>'
            f"<summary><b>{html.escape(badge)}</b> "
            f'<span class="hb-dim">{convo.get("n_turns", 0)} model call(s), '
            f"{len(rows)} message(s)</span></summary>{''.join(rows)}</details>"
        )
    if not blocks:
        return ""
    return (
        f'<div class="hb-card hb-tx"><b>Conversation</b> '
        f'<span class="hb-dim">everything the model saw and produced</span>'
        f"{''.join(blocks)}</div>"
    )


def _confidence(mean_logp: float) -> str:
    """A bar for mean logprob. Closer to 0 is more confident; -1.0 is the practical floor here."""
    pct = max(0.0, min(1.0, 1.0 + mean_logp))  # -0 -> 1.0, -1 -> 0.0
    hue = 8 + int(112 * pct)  # red through amber to green
    return (
        f'<span class="hb-conf" title="mean logprob {mean_logp:.3f}">'
        f'<span style="width:{pct * 100:.0f}%;background:hsl({hue} 75% 45%)"></span></span>'
    )


def _turns_html(r: dict[str, Any]) -> str:
    """Turn by turn: what it did, how much it wrote, how sure it was.

    Replaces a table whose most prominent column was "tools", meaning the number of tools *offered*
    to the model. That number is a property of the harness, identical on every row, and told nobody
    anything. What varies per turn, and is worth reading, is the action taken, the tokens spent on
    it, and the model's confidence while producing them.
    """
    turns = r.get("turns") or []
    if not turns:
        return '<div class="hb-dim">No model calls were captured.</div>'

    used: dict[str, int] = {}
    for t in turns:
        for call in t.get("tool_calls") or []:
            name = str(call.get("name", "?"))
            used[name] = used.get(name, 0) + 1

    rows = []
    for t in turns:
        lp = t.get("per_token_logps") or []
        mean = sum(lp) / len(lp) if lp else 0.0
        gen = len(t.get("completion_token_ids") or [])
        calls = t.get("tool_calls") or []
        if calls:
            action = " ".join(
                f"<code>{html.escape(str(c.get('name', 'tool')))}</code>" for c in calls
            )
        elif t.get("finish_reason") == "stop":
            action = '<span class="hb-dim">final answer</span>'
        else:
            action = '<span class="hb-dim">text only</span>'
        note = (
            ' <span class="hb-tag hb-drop-tag">discarded</span>'
            if t.get("discarded")
            else ""
        )
        preview = _clip(t.get("text") or "", 160)
        rows.append(
            f'<tr class="{"hb-drop-row" if t.get("discarded") else ""}">'
            f'<td class="hb-num">{t.get("turn")}</td>'
            f"<td>{action}{note}"
            + (f'<div class="hb-dim hb-prev">{preview}</div>' if preview else "")
            + f'</td><td class="hb-num">{gen:,}</td>'
            f"<td>{_confidence(mean) if lp else ''}</td>"
            f'<td class="hb-dim">{html.escape(str(t.get("finish_reason") or ""))}</td></tr>'
        )

    histogram = ""
    if used:
        top = sorted(used.items(), key=lambda kv: -kv[1])
        histogram = (
            '<div class="hb-dim" style="margin-top:10px">tools used: '
            + " &nbsp; ".join(f"<code>{html.escape(k)}</code>×{v}" for k, v in top)
            + "</div>"
        )

    return (
        '<div class="hb-card"><b>Turn by turn</b>'
        '<table class="hb-tbl"><thead><tr><th>#</th><th>action</th><th>tokens</th>'
        "<th>confidence</th><th>stopped because</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + histogram
        + '<div class="hb-dim" style="margin-top:6px">Confidence is the mean logprob of the '
        "sampled tokens: full bar means the model was near-certain, short means it was "
        "guessing. Discarded turns were generated and billed but lead nowhere, so they are "
        "excluded from training paths.</div></div>"
    )


def _write_contract(r: dict[str, Any]) -> str | None:
    """Write `contract.json`: exactly what a trainer consumes, nothing else.

    Per turn, `(prompt_token_ids, completion_token_ids, per_token_logps)` plus the reward. The
    logprobs are the load-bearing part and the reason this is a separate file: they are the
    behaviour policy's, recorded at sampling time, and cannot be recovered afterwards by re-running
    the prompt. Discarded turns are kept but flagged, because they were generated and billed and a
    trainer must be able to see them in order to exclude them deliberately.

    Returns `None` for an eval rollout. Writing a file whose every `prompt_token_ids` is `[]` would
    hand someone a download named `contract.json` containing no contract, and a file on disk is far
    more convincing than an empty list in a JSON blob.
    """
    import tempfile
    from pathlib import Path as _Path

    turns = r.get("turns") or []
    if not turns or r.get("rollout_type", "train") == "eval":
        return None
    contract = {
        "task_id": r.get("task_id", ""),
        "task_name": r.get("task_name", ""),
        "dataset": r.get("dataset", ""),
        "harness": r.get("harness", ""),
        "sandbox": r.get("sandbox", ""),
        "trial_name": r.get("trial_name", ""),
        "reward": r.get("reward"),
        "rewards": r.get("rewards") or {},
        "reward_key": r.get("reward_key", ""),
        "n_trainable_tokens": r.get("n_trainable_tokens", 0),
        "turns": [
            {
                "turn": t.get("turn"),
                "prompt_token_ids": t.get("prompt_token_ids") or [],
                "completion_token_ids": t.get("completion_token_ids") or [],
                "per_token_logps": t.get("per_token_logps") or [],
                "finish_reason": t.get("finish_reason"),
                "discarded": bool(t.get("discarded")),
            }
            for t in turns
        ],
    }
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(r.get("task_name") or "rollout"))
    target = (
        _Path(tempfile.mkdtemp(prefix="harbor-contract-")) / f"{name}.contract.json"
    )
    target.write_text(json.dumps(contract, indent=2))
    return str(target)


def _summary_json(r: dict[str, Any]) -> str:
    """The result with the token arrays summarised, which is the part anyone actually reads.

    The full document stays available below; printing 8000 integers first buries the fields that
    carry meaning.
    """
    compact = {k: v for k, v in r.items() if k not in ("turns", "conversations")}
    compact["turns"] = [
        {
            "turn": t.get("turn"),
            "action": [c.get("name") for c in (t.get("tool_calls") or [])] or "text",
            "prompt_token_ids": f"<{len(t.get('prompt_token_ids') or [])} ids>",
            "completion_token_ids": f"<{len(t.get('completion_token_ids') or [])} ids>",
            "per_token_logps": f"<{len(t.get('per_token_logps') or [])} floats>",
            "finish_reason": t.get("finish_reason"),
            "discarded": t.get("discarded"),
            "text": (t.get("text") or "")[:200],
        }
        for t in (r.get("turns") or [])[:200]
    ]
    compact["conversations"] = [
        {
            "role": c.get("role"),
            "n_turns": c.get("n_turns"),
            "messages": f"<{len(c.get('messages') or [])} messages>",
        }
        for c in (r.get("conversations") or [])
    ]
    return json.dumps(compact, indent=2)[:200_000]


def _read(path: Any, limit: int = 20000) -> str:
    try:
        text = path.read_text(errors="replace")
    except Exception:  # noqa: BLE001
        return ""
    return text if len(text) <= limit else text[:limit] + "\n…truncated…"


def harbor_gradio_builder(
    *,
    datasets: list[str] | None = None,
    title: str | None = None,
) -> gr.Blocks:
    """Build the Harbor UI.

    Args:
        datasets (`list[str]`, *optional*):
            Dataset specs served by this server; each becomes a selectable split.

    Returns:
        `gr.Blocks`: The interface.
    """
    from .tasks import HarborTaskProvider, resolve_task_dirs

    datasets = list(datasets or [])

    def on_validate(url: str, model: str, api_key: str):
        from openenv.core.harness.capture.validate_llm import list_models, validate_llm

        from .capabilities import capabilities
        from .seams import agent_facing_model
        from .serving import HarborService

        url = (url or "").strip().rstrip("/")
        api_key = (api_key or "").strip() or None
        if not url:
            return (
                _UNVALIDATED,
                gr.update(),
                gr.update(),
                {},
                gr.update(interactive=False),
            )

        if not model:
            served = list_models(url, api_key=api_key)
            if len(served) != 1:
                hint = (
                    f"`{', '.join(served[:12])}`"
                    if served
                    else "nothing reachable — check the URL, and the API key if it needs one"
                )
                return (
                    f"**Pick a model** — this endpoint serves {hint}.",
                    gr.update(),
                    gr.update(),
                    {},
                    gr.update(interactive=False),
                )
            model = served[0]

        report = validate_llm(url, model, api_key=api_key)
        if not report.reachable:
            why = "; ".join(report.findings) or "unreachable"
            return (
                f"**Not usable** — {why}\n\n"
                "Needs vLLM with `--return-tokens-as-token-ids --logprobs-mode "
                "processed_logprobs`, SGLang built from git main, or any reachable OpenAI-spec "
                "endpoint (with an API key) for eval rollouts.",
                gr.update(),
                gr.update(),
                {},
                gr.update(interactive=False),
            )

        caps = capabilities(
            datasets=datasets,
            llm={
                "url": url,
                "model": model,
                "ok": report.ok,
                "capture_level": report.capture_level,
                "reachable": True,
                "authenticated": bool(api_key),
            },
        )
        sandboxes = caps.available_sandboxes
        by_dialect: dict[str, list[str]] = {}
        for h in caps.harnesses:
            if h.status == "validated":
                by_dialect.setdefault(h.dialect, []).append(h.name)
        choices = [
            (f"{n}  ({d})", n)
            for d, names in sorted(by_dialect.items())
            for n in sorted(names)
        ]
        values = [v for _, v in choices]

        leaf = agent_facing_model(model)
        if report.trainable:
            lines = [f"**Ready — TRAIN** · `{model}` · token ids + logprobs ✓"]
        else:
            detail = (
                "logprobs, no token ids"
                if report.capture_level == "logprobs"
                else "no token ids, no logprobs"
            )
            lines = [
                f"**Ready — EVAL ONLY** · `{model}` · {detail}",
                "Rollouts carry the reward and the full trace, but nothing trainable.",
            ]
        if leaf != model:
            lines.append(f"Sent to agents as `{leaf}`, rewritten back on the way out.")
        for fix in report.param_fixes:
            lines.append(f"<span style='opacity:.7'>upstream compat: {fix}</span>")
        # The one thing a user cannot discover by reading the endpoint's own docs: whether a model
        # will actually sustain an agent loop here. Shown at Validate rather than after a rollout,
        # because a rollout costs a sandbox and several minutes to learn the same thing.
        for finding in report.findings:
            if "behaviour_changed" in finding or "tool_call" in finding:
                detail = finding.split(": ", 2)[-1]
                lines.append(f"⚠️ {detail}")

        # Run uses the endpoint typed above. The engine is a per-rollout argument, so a browser can
        # point this server at any reachable OpenAI-spec endpoint without restarting it — which is
        # the whole point of validating a URL here. Say which one will be used, because a server may
        # also have been booted with a default and the two can differ.
        service = HarborService.current()
        if (
            service is not None
            and service.llm_url
            and service.llm_url.rstrip("/") != url
        ):
            lines.append(
                f"Rollouts will use **this** endpoint, not the server's default "
                f"(`{service.llm_url}`)."
            )
        lines.append(
            f"Sandboxes: {', '.join(f'`{s}`' for s in sandboxes) or '**none usable**'}"
        )
        blocked = [s.name for s in caps.sandboxes if not s.available]
        if blocked:
            lines.append(
                f"<span style='opacity:.6'>unavailable: {', '.join(blocked)}</span>"
            )

        return (
            "  \n".join(lines),
            gr.update(
                choices=choices,
                value="opencode"
                if "opencode" in values
                else (values[0] if values else None),
            ),
            gr.update(choices=sandboxes, value=sandboxes[0] if sandboxes else None),
            # `ok` gates the Run button and now means "reachable", not "trainable": an eval endpoint
            # is a perfectly good thing to press Run against.
            {
                "url": url,
                "model": model,
                "ok": True,
                "capture_level": report.capture_level,
                "trainable": report.trainable,
                # Carried so Run can reach a token-gated endpoint. Without it, validating a hosted
                # provider succeeded and pressing Run then failed to authenticate against the same
                # URL. `gr.State` is held server-side and this is never rendered back into the page,
                # which is the same rule the API key box itself follows.
                "api_key": api_key or "",
            },
            gr.update(
                interactive=bool(sandboxes),
                value="Run rollout" if report.trainable else "Run eval rollout",
            ),
        )

    def on_dataset(spec: str):
        if not spec:
            return gr.update(), ""
        try:
            n = len(resolve_task_dirs(spec))
        except Exception as exc:  # noqa: BLE001
            return gr.update(value=0), f"Cannot load `{spec}` — {exc}"
        return gr.update(value=0), f"**{n}** tasks · 0–{n - 1}"

    def on_task(spec: str, index: int):
        if not spec:
            return "", "", "", "", ""
        try:
            task_dir = HarborTaskProvider([spec]).task_dir(spec, int(index))
        except Exception as exc:  # noqa: BLE001
            return f"_{exc}_", "", "", "", ""
        env_dir, tests_dir = task_dir / "environment", task_dir / "tests"
        return (
            f"`{task_dir.name}`",
            _read(task_dir / "instruction.md"),
            _read(env_dir / "Dockerfile"),
            _read(task_dir / "task.toml"),
            _read(tests_dir / "test.sh"),
        )

    def on_run(engine: dict, spec: str, index: int, harness: str, sandbox: str):
        """Stream progress while the rollout runs, then the result and its graph."""
        import asyncio
        import queue
        import threading
        import time
        from pathlib import Path

        from .rollout import run_rollout as _run
        from .serving import HarborService

        if not engine.get("ok"):
            yield _UNVALIDATED, "", "", "{}", None, gr.update(interactive=True)
            return
        service = HarborService.current()
        if service is None:
            yield (
                "<b>Server not initialised</b> — no capture proxy running.",
                "",
                "",
                "{}",
                None,
                gr.update(interactive=True),
            )
            return
        try:
            task_dir = HarborTaskProvider([spec]).task_dir(spec, int(index))
        except Exception as exc:  # noqa: BLE001
            yield (
                f"<b>Bad task</b> — {html.escape(str(exc))}",
                "",
                "",
                "{}",
                None,
                gr.update(interactive=True),
            )
            return

        done: queue.Queue = queue.Queue(maxsize=1)

        async def _run_with_engine():
            """Resolve the engine the user validated, then run against it.

            The engine is per rollout, so the URL in the box is the one used. Resolving it through the
            capture server's pool means the tier comes from a real probe of that endpoint rather than
            from whatever the server happened to boot with — and the probe is cached, so pressing Run
            repeatedly costs nothing after the first time.
            """
            from openenv.core.harness.capture.sessions import Upstream

            pool = service.capture.app.state.upstreams
            typed_url = str((engine or {}).get("url") or "").strip()
            if typed_url:
                upstream = Upstream(
                    llm_url=typed_url,
                    model=str((engine or {}).get("model") or ""),
                    api_key=str((engine or {}).get("api_key") or "") or None,
                )
                client, level = await pool.resolve(upstream)
                served = client.served_model or upstream.model
            else:
                # Nothing validated in the box: fall back to the server's default, which is what a
                # server booted with --llm-url provides. With neither, the rollout reports the
                # missing engine rather than silently producing an untrainable result.
                upstream, (client, level) = None, pool.default
                level = getattr(service, "capture_level", "text")
                served = service.model
            return await _run(
                task_dir=task_dir,
                harness=harness,
                sandbox=sandbox,
                registry=service.capture.registry,
                intercept_url=service.public_url,
                model=served,
                trials_dir=Path("/tmp/openenv-harbor-trials"),
                dataset=spec,
                capture_level=level,
                upstream=upstream,
                inference=client,
            )

        def worker() -> None:
            try:
                res = asyncio.run(_run_with_engine())
                done.put(("ok", res.model_dump()))
            except Exception as exc:  # noqa: BLE001 - show it, never take the server down
                done.put(("err", f"{type(exc).__name__}: {exc}"))

        before = set(service.capture.registry.list_ids())
        thread = threading.Thread(target=worker, daemon=True)
        started = time.monotonic()
        thread.start()
        session_id = None

        while thread.is_alive():
            if session_id is None:
                session_id = next(
                    iter(set(service.capture.registry.list_ids()) - before), None
                )
            stats, phase, stage = None, "starting up", 0
            if session_id:
                session = service.capture.registry.get(session_id)
                if session is not None:
                    st = session.graph.stats()
                    # n_trainable_tokens only exists after export; mid-run we can count only what
                    # has been sampled, before masking and discards.
                    sampled = sum(
                        len(n.sampled_ids or []) for n in session.graph.nodes()
                    )
                    turns = st.get("n_turns", 0)
                    stats = {
                        "calls": turns,
                        "roots": st.get("n_roots", 0),
                        "sampled tokens": sampled,
                        "discarded": st.get("n_discarded", 0),
                    }
                    if turns:
                        stage = -1  # past setup; the numbers mean something now
                        phase = "agent working"
                        # Only meaningful once a call has landed. Before that `idle_seconds` counts
                        # from session creation, which renders as a stall during a normal boot.
                        stats["since last call"] = f"{session.idle_seconds:.0f}s"
                    else:
                        # The session exists, so the trial reached the agent: the sandbox is up and
                        # the task is uploaded. What is left is the agent's own startup.
                        stage = 2
                        phase = "sandbox ready, starting the agent"
            # The transcript rides in the graph slot: it is empty until the run finishes anyway,
            # and the two answer the same question at different times.
            transcript = ""
            if session_id:
                live = service.capture.registry.get(session_id)
                if live is not None:
                    transcript = _transcript_html(live)
            yield (
                _live_html(
                    harness, sandbox, phase, time.monotonic() - started, stats, stage
                ),
                transcript,
                "",
                "{}",
                None,
                gr.update(interactive=False),
            )
            time.sleep(2.0)

        kind, payload = done.get()
        if kind == "err":
            yield (
                f'<div class="hb-card hb-verdict hb-bad"><div class="hb-head">Run failed</div>'
                f'<pre class="hb-err">{html.escape(payload)}</pre></div>',
                "",
                "",
                "{}",
                None,
                gr.update(interactive=True),
            )
            return
        yield (
            _result_html(payload),
            _conversation_html(payload),
            _turns_html(payload),
            _summary_json(payload),
            _write_contract(payload),
            gr.update(interactive=True),
        )

    with gr.Blocks(title=title or "Harbor") as app:
        gr.HTML(f"<style>{_CSS}</style>")
        state = gr.State({})

        with gr.Column(elem_classes="hb-wrap"):
            gr.Markdown(
                "## Harbor\nRun a coding agent on a Harbor task. Against vLLM or SGLang you get "
                "every token and logprob it produced, ready to train on; against any other "
                "OpenAI-spec endpoint you get the reward and the full trace."
            )

            with gr.Row(equal_height=False):
                # left — the model
                with gr.Column(scale=1, elem_classes="hb-cell"):
                    gr.Markdown("### LLM")
                    # Deliberately empty. Prefilling meant the box already held whatever URL the
                    # server was started with, so Validate confirmed a value nobody chose and a
                    # stale endpoint could be used without anyone noticing it was stale.
                    url_in = gr.Textbox(
                        label="LLM URL",
                        placeholder="https://…  any OpenAI-spec endpoint",
                        info="vLLM, SGLang, OpenAI, Anthropic, HF Inference Providers. "
                        "Accepts a bare root or one ending in /v1.",
                    )
                    gr.HTML(_labelled("API key (optional)", _KEY_TIP))
                    key_in = gr.Textbox(
                        label="",
                        type="password",
                        placeholder="only for a hosted provider",
                        show_label=False,
                    )
                    model_in = gr.Textbox(
                        label="Model (optional)",
                        placeholder="read from the endpoint",
                        info="Required when the endpoint serves more than one model.",
                    )
                    validate_btn = gr.Button("Validate", variant="secondary")
                    gr.HTML(_labelled("Capture level", _LEVEL_TIP))
                    engine_md = gr.Markdown(_UNVALIDATED)
                    gr.Markdown("### Agent")
                    harness_in = gr.Dropdown(
                        label="Agent",
                        choices=[],
                        info="The coding agent to run. Its dialect is shown in brackets; the "
                        "proxy translates all four to one upstream call.",
                    )
                    sandbox_in = gr.Dropdown(
                        label="Sandbox",
                        choices=[],
                        info="Where the agent executes. Harbor's backends, not OpenEnv's "
                        "container providers — only those with working credentials are listed.",
                    )

                # right — the task. Everything but the picker is folded away: the instruction alone
                # runs to a screenful, and it pushed the Run button below the fold.
                with gr.Column(scale=2, elem_classes="hb-cell"):
                    gr.Markdown("### Task")
                    with gr.Row():
                        ds_in = gr.Dropdown(
                            label="Dataset",
                            choices=datasets,
                            value=datasets[0] if datasets else None,
                            scale=3,
                        )
                        idx_in = gr.Number(
                            label="Index", value=0, precision=0, minimum=0, scale=1
                        )
                    count_md = gr.Markdown()
                    task_md = gr.Markdown()
                    with gr.Accordion("Task details", open=False):
                        with gr.Accordion("Instruction", open=True):
                            instruction_box = gr.Code(
                                label="", language="markdown", lines=16
                            )
                        with gr.Accordion("Dockerfile", open=False):
                            dockerfile_box = gr.Code(
                                label="", language="dockerfile", lines=12
                            )
                        with gr.Accordion("task.toml", open=False):
                            toml_box = gr.Code(label="", language="python", lines=12)
                        with gr.Accordion("Grader", open=False):
                            tests_box = gr.Code(label="", language="shell", lines=12)

            # Full width, under both columns: the action belongs to the pair, not to either one.
            run_btn = gr.Button(
                "Run rollout", variant="primary", interactive=False, scale=1
            )

            gr.Markdown("---")
            result_html = gr.HTML()
            # Live transcript while running, then the full conversation once finished.
            convo_html = gr.HTML()
            # Per-turn analysis plus the token-flow graph.
            analysis_html = gr.HTML()
            # The training contract as a file: token ids and the behaviour-policy logprobs, which
            # are the part that cannot be reconstructed after the fact.
            contract_file = gr.File(
                label="contract.json — token ids, logprobs and reward "
                "(train rollouts only)",
                interactive=False,
                visible=True,
            )
            with gr.Accordion("Result JSON", open=False):
                raw_json = gr.Code(language="json", lines=22)

        validate_btn.click(
            on_validate,
            [url_in, model_in, key_in],
            [engine_md, harness_in, sandbox_in, state, run_btn],
        )
        ds_in.change(on_dataset, [ds_in], [idx_in, count_md])
        ds_in.change(
            on_task,
            [ds_in, idx_in],
            [task_md, instruction_box, dockerfile_box, toml_box, tests_box],
        )
        idx_in.change(
            on_task,
            [ds_in, idx_in],
            [task_md, instruction_box, dockerfile_box, toml_box, tests_box],
        )
        run_btn.click(
            on_run,
            [state, ds_in, idx_in, harness_in, sandbox_in],
            [result_html, convo_html, analysis_html, raw_json, contract_file, run_btn],
        )

        if datasets:
            app.load(on_dataset, [ds_in], [idx_in, count_md])
            app.load(
                on_task,
                [ds_in, idx_in],
                [task_md, instruction_box, dockerfile_box, toml_box, tests_box],
            )
    return app
