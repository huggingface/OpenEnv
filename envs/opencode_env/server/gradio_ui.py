# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal Gradio UI for opencode_env.

Mounts under the standard OpenEnv ``/web`` path via the
``gradio_builder=`` callback documented at
https://meta-pytorch.org/OpenEnv/customizing-web-ui.html.

One page with:
  - endpoint selector (``vllm`` / ``openai`` / ``hf_router``) — the catalog
    resolves the actual base_url / api_key / model from env vars.
  - instruction + setup (bash, one cmd per line) + verify (bash, one cmd
    per line) textareas — the same Task shape the MCP tool accepts.
  - Tunables (mode, disable_thinking, max_tokens_cap, top_logprobs,
    agent_timeout_s, template).
  - Preset buttons for the ready-made example tasks.
  - Run button → result panel with reward, setup/verify per-command
    results, file outputs, logprob stats, agent + proxy log tails,
    and the raw RolloutResult JSON.
"""

from __future__ import annotations

import json
from typing import Any

import gradio as gr

try:
    from .catalog import ENDPOINT_KINDS, catalog_summary, resolve_endpoint
    from .opencode_environment import OpenCodeEnvironment
except ImportError:  # pragma: no cover
    from server.catalog import ENDPOINT_KINDS, catalog_summary, resolve_endpoint  # type: ignore
    from server.opencode_environment import OpenCodeEnvironment  # type: ignore


# ────────────────────────────────────────────────────────────────────────────
# Preset task examples — each fills (instruction, setup, verify).
# ────────────────────────────────────────────────────────────────────────────

PRESETS: dict[str, dict[str, str]] = {
    "binary_search": {
        "instruction": (
            "Create a single Python file named `binary_search.py` in the "
            "current working directory. Use the relative path `binary_search.py`. "
            "Expose exactly one function:\n"
            "    def binary_search(arr: list[int], target: int) -> int\n"
            "Return the index of `target` in the sorted list `arr`, or -1 if "
            "absent. Use the binary-search algorithm; do not call list.index."
        ),
        "setup": "",
        "verify": (
            "test -f /home/user/workdir/binary_search.py\n"
            "python -c \"import sys; sys.path.insert(0, '/home/user/workdir'); "
            "import binary_search; "
            "assert binary_search.binary_search([1,2,3,4,5], 3) == 2; "
            "assert binary_search.binary_search([1,2,3], 99) == -1; "
            "assert binary_search.binary_search([], 1) == -1; "
            "print('OK')\""
        ),
    },
    "fizzbuzz": {
        "instruction": (
            "Create `fizzbuzz.py` in the current directory exposing "
            "`def fizzbuzz(n: int) -> list[str]` that returns the FizzBuzz "
            "sequence for the integers 1..n. 'Fizz' for multiples of 3, 'Buzz' "
            "for 5, 'FizzBuzz' for both, otherwise the number as a string."
        ),
        "setup": "",
        "verify": (
            "test -f /home/user/workdir/fizzbuzz.py\n"
            "python -c \"import sys; sys.path.insert(0, '/home/user/workdir'); "
            "import fizzbuzz; "
            "assert fizzbuzz.fizzbuzz(5) == ['1','2','Fizz','4','Buzz']; "
            "assert fizzbuzz.fizzbuzz(15)[-1] == 'FizzBuzz'; "
            "print('OK')\""
        ),
    },
    "pandas_csv": {
        "instruction": (
            "Read `/home/user/data/numbers.csv` (a CSV with a single column "
            "`x` of integers) using pandas. Compute the mean of the `x` "
            "column and write it as a single float to `/home/user/workdir/mean.txt` "
            "(no extra characters, no newline)."
        ),
        "setup": (
            "pip install --quiet pandas\n"
            "mkdir -p /home/user/data\n"
            "printf 'x\\n1\\n2\\n3\\n4\\n5\\n6\\n7\\n8\\n9\\n10\\n' > /home/user/data/numbers.csv"
        ),
        "verify": (
            "test -f /home/user/workdir/mean.txt\n"
            "python -c \"v=float(open('/home/user/workdir/mean.txt').read().strip()); "
            "assert abs(v-5.5) < 1e-6, v; print('mean=', v)\""
        ),
    },
}


# ────────────────────────────────────────────────────────────────────────────
# Result rendering helpers
# ────────────────────────────────────────────────────────────────────────────


def _split_commands(text: str) -> list[str]:
    return [line for line in (text or "").splitlines() if line.strip()]


def _badge_for_reward(reward: float | None) -> str:
    if reward is None:
        return "**reward**: _n/a_"
    if reward >= 0.999:
        emoji = "[PASS]"
    elif reward > 0.0:
        emoji = "[PARTIAL]"
    else:
        emoji = "[FAIL]"
    return f"### {emoji}  reward = `{reward:.2f}`"


def _summary_md(result: dict[str, Any]) -> str:
    parts = [_badge_for_reward(result.get("reward"))]
    parts.append(
        f"**sandbox**: `{result.get('sandbox_id') or 'n/a'}`  ·  "
        f"**wall**: `{result.get('wall_s', 0):.1f}s`  ·  "
        f"**agent_exit**: `{result.get('agent_exit_code')}`  ·  "
        f"**mode**: `{result.get('mode', 'n/a')}`"
    )
    if result.get("error"):
        parts.append(f"**error**: `{result['error']}`")
    return "\n\n".join(parts)


def _command_rows(items: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for it in items or []:
        cmd = it.get("cmd", "")
        rows.append(
            [
                cmd if len(cmd) <= 80 else cmd[:77] + "...",
                str(it.get("exit_code", "")),
                f"{it.get('duration_s', 0):.2f}s",
                (it.get("stderr") or "").splitlines()[-1][:80] if it.get("exit_code") else "",
            ]
        )
    return rows


def _logprobs_md(turns: list[dict[str, Any]]) -> str:
    if not turns:
        return "_No proxy turns captured._\n\nThis is normal in `black_box` mode. In `transparent_proxy` mode, an empty list usually means the agent never made an LLM call (check the agent log)."
    n = len(turns)
    productive = sum(1 for t in turns if t.get("completion_tokens"))
    total_toks = sum(len(t.get("completion_tokens") or []) for t in turns)
    all_lps = [
        float(x)
        for t in turns
        for x in (t.get("per_token_logps") or [])
        if x is not None
    ]
    mean_lp = (sum(all_lps) / len(all_lps)) if all_lps else None
    lines = [
        f"**turns**: `{n}`  ·  **productive**: `{productive}`  ·  "
        f"**total_completion_tokens**: `{total_toks}`",
    ]
    if mean_lp is not None:
        lines.append(f"**mean_logprob**: `{mean_lp:+.4f}`")
    finishes: dict[str, int] = {}
    for t in turns:
        f = t.get("finish_reason") or "unknown"
        finishes[f] = finishes.get(f, 0) + 1
    if finishes:
        lines.append(
            "**finish_reasons**: " + "  ".join(f"`{k}={v}`" for k, v in finishes.items())
        )
    productive_rows = [t for t in turns if t.get("completion_tokens")]
    if productive_rows:
        first = productive_rows[0]
        toks = first["completion_tokens"][:10]
        lps = first.get("per_token_logps") or []
        lines.append(
            f"\n**first productive turn (first 10 tokens)**\n\n"
            f"```\n"
            + "\n".join(
                f"  {tok!r:<14}  {lp:+.3f}" if i < len(lps) else f"  {tok!r:<14}  -"
                for i, (tok, lp) in enumerate(zip(toks, lps + [None] * len(toks)))
            )
            + "\n```"
        )
    return "\n\n".join(lines)


def _live_status_md(
    endpoint_kind: str,
    model: str,
    mode: str,
    elapsed_s: float,
    lines: list[tuple[float, str]],
) -> str:
    """Render a live phase log (latest at the bottom) with elapsed timestamps."""
    head = (
        f"### running…  `elapsed={elapsed_s:.1f}s`\n\n"
        f"_endpoint=`{endpoint_kind}`  model=`{model}`  mode=`{mode}`_\n\n"
    )
    if not lines:
        body = "_(waiting for first phase update…)_"
    else:
        # Show the most recent ~12 lines so the panel doesn't grow unbounded.
        rows = ["| t (s) | phase |", "|---|---|"]
        for ts, msg in lines[-12:]:
            rows.append(f"| `{ts:>6.1f}` | {msg.replace(chr(10), ' ')[:200]} |")
        body = "\n".join(rows)
    return head + body


def _files_md(files: dict[str, str]) -> str:
    if not files:
        return "_No files in the workdir._"
    chunks = []
    for path, content in files.items():
        chunks.append(f"**`{path}`**\n```python\n{content[:4000]}\n```")
    return "\n\n".join(chunks)


def _catalog_banner() -> str:
    rows = ["**Endpoint catalog (env vars + defaults)**", ""]
    rows.append("| kind | base_url | model | env vars | configured |")
    rows.append("|---|---|---|---|---|")
    for entry in catalog_summary():
        envs = (
            f"`{entry['base_url_env']}`<br/>`{entry['api_key_env']}`<br/>"
            f"`{entry['model_env']}`"
        )
        ok = "yes" if entry["configured"] else "**no**"
        rows.append(
            f"| `{entry['kind']}` | `{entry['default_base_url'] or '-'}` | "
            f"`{entry['default_model'] or '-'}` | {envs} | {ok} |"
        )
    return "\n".join(rows)


# ────────────────────────────────────────────────────────────────────────────
# Builder
# ────────────────────────────────────────────────────────────────────────────


def opencode_gradio_builder(
    web_manager,        # noqa: ARG001 (unused: we instantiate the env directly)
    action_fields,      # noqa: ARG001
    metadata,           # noqa: ARG001
    is_chat_env,        # noqa: ARG001
    title,
    quick_start_md,     # noqa: ARG001
) -> gr.Blocks:
    """Build the opencode_env console.

    Compatible with ``create_app(..., gradio_builder=...)``. We ignore
    ``web_manager`` and instantiate :class:`OpenCodeEnvironment` ourselves
    inside the run handler — opencode_env's run_rollout doesn't need any
    per-session state beyond the env's own bookkeeping, and instantiating
    is cheap (no sandbox is created until the tool fires).
    """

    def run(
        endpoint: str,
        model: str,
        base_url: str,
        api_key: str,
        instruction: str,
        setup_text: str,
        verify_text: str,
        mode: str,
        disable_thinking: str,
        template: str,
        max_tokens_cap: int,
        top_logprobs: int,
        agent_timeout_s: float,
    ):
        """Generator handler — yields incremental UI updates.

        Each ``yield`` is a tuple matching ``outputs=[...]``:
        (summary_md, setup_table, verify_table, files_md, logprobs_md,
        logs_md, raw_json). Early yields keep summary_md as a live phase
        log while the rollout runs; the final yield populates everything.
        """
        import queue
        import threading
        import time

        # Resolve endpoint up front — if this fails, we can return one
        # immediate result with no streaming needed.
        try:
            resolved = resolve_endpoint(
                endpoint, base_url=base_url, api_key=api_key, model=model
            )
        except ValueError as exc:
            err = f"endpoint resolution failed: {exc}"
            yield (f"### error\n\n```\n{err}\n```", [], [], "", "", "", {"error": err})
            return

        # Translate "auto" / "on" / "off" into bool / None.
        if disable_thinking == "on":
            dt: bool | None = True
        elif disable_thinking == "off":
            dt = False
        else:
            dt = None

        env = OpenCodeEnvironment()

        # The worker fires _run_rollout_impl in a background thread and
        # streams progress messages into a queue; this generator polls the
        # queue every 0.5s and yields a refreshed status_md to the UI.
        status_q: queue.Queue = queue.Queue()
        result_holder: dict = {}

        def _cb(msg: str) -> None:
            status_q.put(("status", msg, time.time()))

        def _worker():
            try:
                payload = env._run_rollout_impl(
                    base_url=resolved.base_url,
                    api_key=resolved.api_key,
                    model=resolved.model,
                    instruction=instruction,
                    setup=_split_commands(setup_text),
                    verify=_split_commands(verify_text),
                    task_id="ui",
                    mode=mode,
                    disable_thinking=(
                        dt if dt is not None else resolved.disable_thinking_default
                    ),
                    max_tokens_cap=int(max_tokens_cap),
                    top_logprobs=int(top_logprobs),
                    agent_timeout_s=float(agent_timeout_s),
                    template=template,
                    progress_cb=_cb,
                )
                result_holder["payload"] = payload
            except Exception as exc:  # noqa: BLE001
                result_holder["error"] = f"{type(exc).__name__}: {exc}"
                status_q.put(("error", result_holder["error"], time.time()))
            finally:
                status_q.put(("done", None, time.time()))

        worker = threading.Thread(target=_worker, daemon=True)
        t_start = time.time()
        worker.start()

        # First yield: announce we've started. Empty result panels.
        yield (
            f"### running…\n\n_endpoint=`{resolved.kind}`  model=`{resolved.model}`  mode=`{mode}`_",
            [], [], "", "", "", {},
        )

        status_lines: list[tuple[float, str]] = []
        finished = False
        while not finished:
            try:
                kind, msg, ts = status_q.get(timeout=0.5)
                if kind == "status":
                    status_lines.append((ts - t_start, msg))
                elif kind == "error":
                    status_lines.append((ts - t_start, f"ERROR: {msg}"))
                elif kind == "done":
                    finished = True
            except queue.Empty:
                pass

            # Render the live status pane.
            elapsed = time.time() - t_start
            md = _live_status_md(resolved.kind, resolved.model, mode, elapsed, status_lines)
            yield (md, [], [], "", "", "", {})

        # Drain any final messages still in the queue.
        while not status_q.empty():
            try:
                kind, msg, ts = status_q.get_nowait()
                if kind == "status":
                    status_lines.append((ts - t_start, msg))
            except queue.Empty:
                break

        if "payload" not in result_holder:
            err = result_holder.get("error", "unknown error")
            yield (
                f"### error\n\n```\n{err}\n```",
                [], [], "", "",
                _live_status_md(resolved.kind, resolved.model, mode,
                                time.time() - t_start, status_lines),
                {"error": err},
            )
            return

        result = json.loads(result_holder["payload"])
        yield (
            _summary_md(result),
            _command_rows(result.get("setup_results") or []),
            _command_rows(result.get("verify_results") or []),
            _files_md(result.get("files") or {}),
            _logprobs_md(result.get("proxy_turns") or []),
            (
                f"### live phase log\n\n"
                + _live_status_md(resolved.kind, resolved.model, mode,
                                  time.time() - t_start, status_lines)
                + f"\n\n### agent log (tail)\n```\n{result.get('agent_log_tail', '')[:4000]}\n```\n\n"
                f"### proxy log (tail)\n```\n{result.get('proxy_log_tail', '')[:4000]}\n```"
            ),
            result,
        )

    def apply_preset(name: str) -> tuple[str, str, str]:
        p = PRESETS.get(name) or {"instruction": "", "setup": "", "verify": ""}
        return p["instruction"], p["setup"], p["verify"]

    with gr.Blocks(title=title or "opencode_env") as app:
        gr.Markdown(f"# {title or 'opencode_env'}")
        gr.Markdown(
            "Run one OpenCode rollout in an E2B sandbox against your chosen "
            "LLM endpoint. Pick an endpoint, write the task as `(instruction, "
            "setup, verify)`, and inspect the reward + per-token logprobs."
        )

        gr.Markdown(_catalog_banner())

        with gr.Row():
            endpoint = gr.Dropdown(
                choices=list(ENDPOINT_KINDS),
                value="openai",
                label="Endpoint",
                scale=1,
            )
            model = gr.Textbox(
                label="Model (blank → catalog default)", placeholder="gpt-4o-mini",
                scale=2,
            )
        with gr.Row():
            base_url = gr.Textbox(
                label="Base URL (blank → env / catalog default)",
                placeholder="https://api.openai.com/v1", scale=2,
            )
            api_key = gr.Textbox(
                label="API key (blank → server env var)",
                placeholder="(server env)", type="password", scale=1,
            )

        instruction = gr.Textbox(
            label="Instruction (the prompt opencode runs)",
            lines=4,
            value=PRESETS["binary_search"]["instruction"],
        )

        with gr.Row():
            setup_text = gr.Textbox(
                label="Setup (one bash command per line — runs BEFORE the agent)",
                lines=5,
                value=PRESETS["binary_search"]["setup"],
            )
            verify_text = gr.Textbox(
                label="Verify (one bash command per line — runs AFTER the agent)",
                lines=5,
                value=PRESETS["binary_search"]["verify"],
            )

        with gr.Row():
            preset_bs = gr.Button("preset · binary_search", size="sm")
            preset_fb = gr.Button("preset · fizzbuzz", size="sm")
            preset_pd = gr.Button("preset · pandas_csv", size="sm")

        with gr.Accordion("Tunables", open=False):
            with gr.Row():
                mode = gr.Dropdown(
                    choices=["transparent_proxy", "black_box"],
                    value="transparent_proxy",
                    label="mode",
                )
                disable_thinking = gr.Dropdown(
                    choices=["auto", "on", "off"],
                    value="auto",
                    label="disable_thinking",
                )
                template = gr.Textbox(
                    label="E2B template (e.g. opencode-rl)",
                    placeholder="(blank → cold install per rollout)",
                )
            with gr.Row():
                max_tokens_cap = gr.Number(value=4096, label="max_tokens_cap", step=1)
                top_logprobs = gr.Number(value=5, label="top_logprobs", step=1)
                agent_timeout_s = gr.Number(value=600, label="agent_timeout_s", step=1)

        run_btn = gr.Button("Run rollout", variant="primary")

        gr.Markdown("---")
        summary_md = gr.Markdown("_Submit a rollout above to see results._")

        with gr.Tabs():
            with gr.Tab("Setup"):
                setup_table = gr.Dataframe(
                    headers=["cmd", "exit", "duration", "stderr"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("Verify"):
                verify_table = gr.Dataframe(
                    headers=["cmd", "exit", "duration", "stderr"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("Files"):
                files_md = gr.Markdown("")
            with gr.Tab("Logprobs"):
                logprobs_md = gr.Markdown("")
            with gr.Tab("Logs"):
                logs_md = gr.Markdown("")
            with gr.Tab("Raw JSON"):
                raw_json = gr.JSON(value={})

        # Wire it up.
        for btn, name in [
            (preset_bs, "binary_search"),
            (preset_fb, "fizzbuzz"),
            (preset_pd, "pandas_csv"),
        ]:
            btn.click(
                fn=lambda n=name: apply_preset(n),
                outputs=[instruction, setup_text, verify_text],
            )

        run_btn.click(
            fn=run,
            inputs=[
                endpoint, model, base_url, api_key,
                instruction, setup_text, verify_text,
                mode, disable_thinking, template,
                max_tokens_cap, top_logprobs, agent_timeout_s,
            ],
            outputs=[
                summary_md, setup_table, verify_table,
                files_md, logprobs_md, logs_md, raw_json,
            ],
        )

    return app
