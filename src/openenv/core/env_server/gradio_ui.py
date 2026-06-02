# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Gradio-based web UI for OpenEnv environments.

Replaces the legacy HTML/JavaScript interface when ENABLE_WEB_INTERFACE is set.
Mount at /web via gr.mount_gradio_app() from create_web_interface_app().
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import gradio as gr

from .types import EnvironmentMetadata


def _escape_md(text: str) -> str:
    """Escape Markdown special characters in user-controlled content."""
    return re.sub(r"([\\`*_\{\}\[\]()#+\-.!|~>])", r"\\\1", str(text))


def _format_observation(data: Dict[str, Any]) -> str:
    """Format reset/step response for Markdown display."""
    lines: List[str] = []
    obs = data.get("observation", {})
    if isinstance(obs, dict):
        if obs.get("prompt"):
            lines.append(f"**Prompt:**\n\n{_escape_md(obs['prompt'])}\n")
        messages = obs.get("messages", [])
        if messages:
            lines.append("**Messages:**\n")
            for msg in messages:
                sender = _escape_md(str(msg.get("sender_id", "?")))
                content = _escape_md(str(msg.get("content", "")))
                cat = _escape_md(str(msg.get("category", "")))
                lines.append(f"- `[{cat}]` Player {sender}: {content}")
            lines.append("")
    reward = data.get("reward")
    done = data.get("done")
    if reward is not None:
        lines.append(f"**Reward:** `{reward}`")
    if done is not None:
        lines.append(f"**Done:** `{done}`")
    return "\n".join(lines) if lines else "*No observation data*"


def _readme_section(metadata: Optional[EnvironmentMetadata]) -> str:
    """README content for the left panel."""
    if not metadata or not metadata.readme_content:
        return "*No README available.*"
    return metadata.readme_content


def get_gradio_display_title(
    metadata: Optional[EnvironmentMetadata],
    fallback: str = "OpenEnv Environment",
) -> str:
    """Return the title used for the Gradio app (browser tab and Blocks)."""
    name = metadata.name if metadata else fallback
    return f"OpenEnv Agentic Environment: {name}"


def build_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str = "OpenEnv Environment",
    quick_start_md: Optional[str] = None,
) -> gr.Blocks:
    """
    Build a Gradio Blocks app for the OpenEnv web interface.

    Args:
        web_manager: WebInterfaceManager (reset/step_environment, get_state).
        action_fields: Field dicts from _extract_action_fields(action_cls).
        metadata: Environment metadata for README/name.
        is_chat_env: If True, single message textbox; else form from action_fields.
        title: App title (overridden by metadata.name when present; see get_gradio_display_title).
        quick_start_md: Optional Quick Start markdown (class names already replaced).

    Returns:
        gr.Blocks to mount with gr.mount_gradio_app(app, blocks, path="/web").
    """
    readme_content = _readme_section(metadata)
    display_title = get_gradio_display_title(metadata, fallback=title)

    async def reset_env():
        try:
            data = await web_manager.reset_environment()
            obs_md = _format_observation(data)
            return (
                obs_md,
                json.dumps(data, indent=2),
                "Environment reset successfully.",
            )
        except Exception as e:
            return ("", "", f"Error: {e}")

    def _step_with_action(action_data: Dict[str, Any]):
        async def _run():
            try:
                data = await web_manager.step_environment(action_data)
                obs_md = _format_observation(data)
                return (
                    obs_md,
                    json.dumps(data, indent=2),
                    "Step complete.",
                )
            except Exception as e:
                return ("", "", f"Error: {e}")

        return _run

    async def step_chat(message: str):
        if not (message or str(message).strip()):
            return ("", "", "Please enter an action message.")
        action = {"message": str(message).strip()}
        return await _step_with_action(action)()

    def get_state_sync():
        try:
            data = web_manager.get_state()
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"Error: {e}"

    with gr.Blocks(title=display_title) as demo:
        with gr.Row():
            with gr.Column(scale=1, elem_classes="col-left"):
                if quick_start_md:
                    with gr.Accordion("Quick Start", open=True):
                        gr.Markdown(quick_start_md)
                with gr.Accordion("README", open=False):
                    gr.Markdown(readme_content)

            with gr.Column(scale=2, elem_classes="col-right"):
                obs_display = gr.Markdown(
                    value=("# Playground\n\nClick **Reset** to start a new episode."),
                )
                with gr.Group():
                    if is_chat_env:
                        action_input = gr.Textbox(
                            label="Action message",
                            placeholder="e.g. Enter your message...",
                        )
                        step_inputs = [action_input]
                        step_fn = step_chat
                    else:
                        step_inputs = []
                        for field in action_fields:
                            name = field["name"]
                            field_type = field.get("type", "text")
                            label = name.replace("_", " ").title()
                            placeholder = field.get("placeholder", "")
                            if field_type == "checkbox":
                                inp = gr.Checkbox(label=label)
                            elif field_type == "number":
                                inp = gr.Number(label=label)
                            elif field_type == "select":
                                choices = field.get("choices") or []
                                inp = gr.Dropdown(
                                    choices=choices,
                                    label=label,
                                    allow_custom_value=False,
                                )
                            elif field_type in ("textarea", "tensor"):
                                inp = gr.Textbox(
                                    label=label,
                                    placeholder=placeholder,
                                    lines=3,
                                )
                            else:
                                inp = gr.Textbox(
                                    label=label,
                                    placeholder=placeholder,
                                )
                            step_inputs.append(inp)

                        async def step_form(*values):
                            if not action_fields:
                                return await _step_with_action({})()
                            action_data = {}
                            for i, field in enumerate(action_fields):
                                if i >= len(values):
                                    break
                                name = field["name"]
                                val = values[i]
                                if field.get("type") == "checkbox":
                                    action_data[name] = bool(val)
                                elif val is not None and val != "":
                                    action_data[name] = val
                            return await _step_with_action(action_data)()

                        step_fn = step_form

                    with gr.Row():
                        step_btn = gr.Button("Step", variant="primary")
                        reset_btn = gr.Button("Reset", variant="secondary")
                        state_btn = gr.Button("Get state", variant="secondary")
                    with gr.Row():
                        status = gr.Textbox(
                            label="Status",
                            interactive=False,
                        )
                    raw_json = gr.Code(
                        label="Raw JSON response",
                        language="json",
                        interactive=False,
                    )

        reset_btn.click(
            fn=reset_env,
            outputs=[obs_display, raw_json, status],
        )
        step_btn.click(
            fn=step_fn,
            inputs=step_inputs,
            outputs=[obs_display, raw_json, status],
        )
        if is_chat_env:
            action_input.submit(
                fn=step_fn,
                inputs=step_inputs,
                outputs=[obs_display, raw_json, status],
            )
        state_btn.click(
            fn=get_state_sync,
            outputs=[raw_json],
        )

    return demo
