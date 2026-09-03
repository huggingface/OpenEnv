# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Custom Gradio UI for the LaTeX OCR environment.

Interactive playground:
  1. "Get next task"  -> pulls a task (image) from the env; shows progress.
  2. type LaTeX        -> the user's transcription attempt.
  3. "Score"           -> grades it server-side and reveals the ground truth.

Drives the *same* environment instance the API uses, via ``web_manager``, so in
stream mode the cursor and progress here match the real episode flow.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

import gradio as gr


def _decode_image(b64: str):
    if not b64:
        return None
    try:
        from PIL import Image

        return Image.open(io.BytesIO(base64.b64decode(b64)))
    except Exception:
        return None


def _obs(result: Dict[str, Any]) -> Dict[str, Any]:
    """reset/step return {'observation': {...}, 'reward':..., 'done':...}."""
    if isinstance(result, dict) and "observation" in result:
        return result["observation"] or {}
    return result or {}


def latex_ocr_ui_builder(
    web_manager,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[Any],
    is_chat_env: bool,
    title: str,
    quick_start_md: Optional[str],
) -> gr.Blocks:
    def _splits() -> List[str]:
        try:
            return web_manager.env.list_splits()
        except Exception:
            return ["train"]

    def _progress_md(o: Dict[str, Any]) -> str:
        total = o.get("total", -1)
        if total and total > 0:
            return (
                f"**Task `{o.get('task_id', '')}`** · position "
                f"**{o.get('index')}/{total}** "
                f"({o.get('pct_done', 0):.4%}) · remaining **{o.get('remaining')}**"
            )
        return f"**Task `{o.get('task_id', '')}`** (index {o.get('index')})"

    with gr.Blocks(title=f"{title} — Playground") as demo:
        gr.Markdown(f"# {title} — try it")
        gr.Markdown(
            "Transcribe the image into **LaTeX**, then score it. Reward = "
            "`0.8·(1−CER) + 0.2·exact_match` against the hidden ground truth."
        )

        with gr.Row():
            with gr.Column(scale=1):
                split_dd = gr.Dropdown(
                    choices=_splits(),
                    value=(_splits() or ["train"])[0],
                    label="Split",
                    interactive=True,
                )
                get_btn = gr.Button("🎲 Get next task", variant="primary", size="lg")
                progress_md = gr.Markdown("*No task yet — click “Get next task”.*")
                image = gr.Image(label="Image to transcribe", type="pil", height=200)

            with gr.Column(scale=1):
                pred_box = gr.Textbox(
                    label="Your LaTeX prediction",
                    placeholder="e.g.  x ^ { 2 } + 1",
                    lines=4,
                )
                score_btn = gr.Button("✅ Score", variant="primary", size="lg")
                score_md = gr.Markdown("")
                truth_box = gr.Textbox(
                    label="Ground truth (revealed after scoring)",
                    lines=4,
                    interactive=False,
                )

        async def on_get_task(split: str):
            result = await web_manager.reset_environment({"split": split})
            o = _obs(result)
            return (
                _decode_image(o.get("image_base64", "")),
                _progress_md(o),
                "",  # clear prediction
                "",  # clear score
                "",  # clear truth
            )

        async def on_score(prediction: str):
            try:
                result = await web_manager.step_environment({"latex": prediction or ""})
            except Exception as e:
                return f"⚠️ {e}. Click **Get next task** first.", ""
            o = _obs(result)
            # The server serializes reward at the top level; fall back to the
            # observation for older cores.
            reward = result.get("reward") if isinstance(result, dict) else None
            if reward is None:
                reward = o.get("reward")
            reward = reward if reward is not None else 0.0
            mark = "🎯 **exact match**" if o.get("exact_match") else "≈ partial"
            md = (
                f"### Reward: `{reward:.4f}`  {mark}\n"
                f"- character error rate: `{o.get('char_error_rate', 1.0):.4f}`\n"
                f"- exact match: `{o.get('exact_match', False)}`"
            )
            return md, o.get("target_latex", "")

        get_btn.click(
            on_get_task,
            inputs=[split_dd],
            outputs=[image, progress_md, pred_box, score_md, truth_box],
        )
        score_btn.click(on_score, inputs=[pred_box], outputs=[score_md, truth_box])

    return demo
