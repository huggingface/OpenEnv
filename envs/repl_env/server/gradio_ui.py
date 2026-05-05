# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom Gradio tab for the REPL environment."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import gradio as gr
from openenv.core.env_server.types import EnvironmentMetadata


def _code_block(title: str, content: str) -> str:
    if not content:
        return ""
    return f"**{title}:**\n```text\n{content}\n```"


def _format_repl_response(data: Dict[str, Any]) -> str:
    """Render REPL observations in a compact Markdown view."""
    observation = data.get("observation", {})
    result = observation.get("result", {})
    sections: List[str] = ["# REPL Session"]

    context_preview = observation.get("context_preview")
    if context_preview:
        sections.append(_code_block("Context Preview", context_preview))

    task_prompt = observation.get("task_prompt")
    if task_prompt:
        sections.append(_code_block("Task Prompt", task_prompt))

    available_variables = observation.get("available_variables") or []
    if available_variables:
        sections.append(
            "**Available Variables:** " + ", ".join(f"`{name}`" for name in available_variables)
        )

    if result.get("locals_snapshot"):
        sections.append(
            "**Locals Snapshot:**\n```json\n"
            + json.dumps(result["locals_snapshot"], indent=2, sort_keys=True)
            + "\n```"
        )

    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    sections.append(_code_block("Stdout", stdout))
    sections.append(_code_block("Stderr", stderr))

    reward = data.get("reward")
    done = data.get("done")
    sections.append(f"**Reward:** `{reward}`")
    sections.append(f"**Done:** `{done}`")

    return "\n\n".join(section for section in sections if section)


def build_repl_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """Build the REPL-specific Gradio tab."""
    del action_fields, is_chat_env, metadata, quick_start_md

    async def reset_repl(
        context: str,
        task_prompt: str,
        hf_token: str,
        llm_model: str,
    ):
        reset_kwargs: Dict[str, Any] = {}
        if context.strip():
            reset_kwargs["context"] = context
        if task_prompt.strip():
            reset_kwargs["task_prompt"] = task_prompt
        if hf_token.strip():
            reset_kwargs["hf_token"] = hf_token
        if llm_model.strip():
            reset_kwargs["llm_model"] = llm_model

        try:
            data = await web_manager.reset_environment(reset_kwargs)
            state = web_manager.get_state()
            return (
                _format_repl_response(data),
                json.dumps(data, indent=2, sort_keys=True),
                json.dumps(state, indent=2, sort_keys=True),
                "REPL reset complete.",
            )
        except Exception as exc:
            return ("", "", "", f"Error: {exc}")

    async def run_code(code: str):
        if not code.strip():
            return ("", "", "", "Enter Python code to run.")

        try:
            data = await web_manager.step_environment({"code": code})
            state = web_manager.get_state()
            return (
                _format_repl_response(data),
                json.dumps(data, indent=2, sort_keys=True),
                json.dumps(state, indent=2, sort_keys=True),
                "Code executed.",
            )
        except Exception as exc:
            return ("", "", "", f"Error: {exc}")

    def get_state_sync():
        try:
            return json.dumps(web_manager.get_state(), indent=2, sort_keys=True)
        except Exception as exc:
            return f"Error: {exc}"

    with gr.Blocks(title=f"{title} - REPL") as blocks:
        gr.Markdown(
            "# REPL Control Panel\n\n"
            "Load a problem into the REPL, execute Python, and inspect state without "
            "leaving the Space."
        )
        with gr.Row():
            with gr.Column(scale=2):
                context = gr.Textbox(
                    label="Context",
                    placeholder="Problem context or source text...",
                    lines=8,
                )
                task_prompt = gr.Textbox(
                    label="Task Prompt",
                    placeholder="What should the agent solve?",
                    lines=3,
                )
                with gr.Accordion("Optional Model Settings", open=False):
                    hf_token = gr.Textbox(
                        label="Hugging Face Token",
                        placeholder="Used only for this reset; not persisted in state",
                        type="password",
                    )
                    llm_model = gr.Textbox(
                        label="LLM Model",
                        placeholder="Optional override for llm_query / rlm_query",
                    )
                code = gr.Textbox(
                    label="Python Code",
                    placeholder="count = len(context.split())",
                    lines=10,
                )
                with gr.Row():
                    reset_btn = gr.Button("Reset", variant="secondary")
                    run_btn = gr.Button("Run", variant="primary")
                    state_btn = gr.Button("Get state", variant="secondary")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=3):
                session_view = gr.Markdown(
                    value="# REPL Session\n\nReset the environment to start."
                )
                raw_json = gr.Code(
                    label="Raw JSON response",
                    language="json",
                    interactive=False,
                )
                state_json = gr.Code(
                    label="Session state",
                    language="json",
                    interactive=False,
                )

        reset_btn.click(
            fn=reset_repl,
            inputs=[context, task_prompt, hf_token, llm_model],
            outputs=[session_view, raw_json, state_json, status],
        )
        run_btn.click(
            fn=run_code,
            inputs=[code],
            outputs=[session_view, raw_json, state_json, status],
        )
        code.submit(
            fn=run_code,
            inputs=[code],
            outputs=[session_view, raw_json, state_json, status],
        )
        state_btn.click(fn=get_state_sync, outputs=[state_json])

    return blocks
