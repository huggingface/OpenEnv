# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unified terminal-style theme for OpenEnv Gradio UI (light/dark)."""

from __future__ import annotations

import gradio as gr

_MONO_FONTS = (
    "JetBrains Mono",
    "Fira Code",
    "Cascadia Code",
    "Consolas",
    "ui-monospace",
    "monospace",
)

_CORE_FONT = (
    "Lato",
    "Inter",
    "Arial",
    "Helvetica",
    "sans-serif",
)

_ZERO_RADIUS = gr.themes.Size(
    xxs="0px",
    xs="0px",
    sm="0px",
    md="0px",
    lg="0px",
    xl="0px",
    xxl="0px",
)

_GREEN_HUE = gr.themes.Color(
    c50="#e6f4ea",
    c100="#ceead6",
    c200="#a8dab5",
    c300="#6fcc8b",
    c400="#3fb950",
    c500="#238636",
    c600="#1a7f37",
    c700="#116329",
    c800="#0a4620",
    c900="#033a16",
    c950="#04200d",
)

_NEUTRAL_HUE = gr.themes.Color(
    c50="#f6f8fa",
    c100="#eaeef2",
    c200="#d0d7de",
    c300="#afb8c1",
    c400="#8c959f",
    c500="#6e7781",
    c600="#57606a",
    c700="#424a53",
    c800="#32383f",
    c900="#24292f",
    c950="#1b1f24",
)

OPENENV_GRADIO_THEME = gr.themes.Base(
    primary_hue=_GREEN_HUE,
    secondary_hue=_NEUTRAL_HUE,
    neutral_hue=_NEUTRAL_HUE,
    font=_CORE_FONT,
    font_mono=_MONO_FONTS,
    radius_size=_ZERO_RADIUS,
).set(
    body_background_fill="#ffffff",
    background_fill_primary="#ffffff",
    background_fill_secondary="#f6f8fa",
    block_background_fill="#ffffff",
    block_border_color="#ffffff",
    block_label_text_color="#57606a",
    block_title_text_color="#24292f",
    border_color_primary="#d0d7de",
    input_background_fill="#ffffff",
    input_border_color="#d0d7de",
    button_primary_background_fill="#1a7f37",
    button_primary_background_fill_hover="#116329",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#f6f8fa",
    button_secondary_background_fill_hover="#eaeef2",
    button_secondary_text_color="#24292f",
    button_secondary_border_color="#d0d7de",
    body_background_fill_dark="#0d1117",
    background_fill_primary_dark="#0d1117",
    background_fill_secondary_dark="#0d1117",
    block_background_fill_dark="#0d1117",
    block_border_color_dark="#0d1117",
    block_label_text_color_dark="#8b949e",
    block_title_text_color_dark="#c9d1d9",
    border_color_primary_dark="#30363d",
    input_background_fill_dark="#0d1117",
    input_border_color_dark="#30363d",
    button_primary_background_fill_dark="#30363d",
    button_primary_background_fill_hover_dark="#484f58",
    button_primary_text_color_dark="#c9d1d9",
    button_secondary_background_fill_dark="#21262d",
    button_secondary_background_fill_hover_dark="#30363d",
    button_secondary_text_color_dark="#c9d1d9",
    button_secondary_border_color_dark="#30363d",
)

OPENENV_GRADIO_CSS = """
* { border-radius: 0 !important; }
.col-left { padding: 16px !important; }
.col-right { padding: 16px !important; }
.prose, .markdown-text, .md,
.prose > *, .markdown-text > * {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.dark .col-left {
    border-left-color: rgba(139, 148, 158, 0.4) !important;
}
.dark .col-right {
    border-left-color: rgba(201, 209, 217, 0.3) !important;
}
"""
