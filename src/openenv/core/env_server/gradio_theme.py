# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unified black-on-white theme for OpenEnv Gradio UI."""

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
    background_fill_secondary="#ffffff",
    block_background_fill="#ffffff",
    block_border_color="#111111",
    block_label_text_color="#111111",
    block_title_text_color="#111111",
    border_color_primary="#111111",
    input_background_fill="#ffffff",
    input_border_color="#111111",
    button_primary_background_fill="#ffffff",
    button_primary_background_fill_hover="#f2f2f2",
    button_primary_text_color="#111111",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#f2f2f2",
    button_secondary_text_color="#111111",
    button_secondary_border_color="#111111",
    body_background_fill_dark="#ffffff",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary_dark="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color_dark="#111111",
    block_label_text_color_dark="#111111",
    block_title_text_color_dark="#111111",
    border_color_primary_dark="#111111",
    input_background_fill_dark="#ffffff",
    input_border_color_dark="#111111",
    button_primary_background_fill_dark="#ffffff",
    button_primary_background_fill_hover_dark="#f2f2f2",
    button_primary_text_color_dark="#111111",
    button_secondary_background_fill_dark="#ffffff",
    button_secondary_background_fill_hover_dark="#f2f2f2",
    button_secondary_text_color_dark="#111111",
    button_secondary_border_color_dark="#111111",
)

OPENENV_GRADIO_CSS = """
* { border-radius: 0 !important; }
.col-left { padding: 16px !important; }
.col-right { padding: 16px !important; }
.gradio-container,
.gradio-container .gr-block,
.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .gr-panel,
.gradio-container .gr-group,
.gradio-container .gr-row,
.gradio-container .gr-column,
.gradio-container .gr-tab,
.gradio-container .gr-tabs,
.gradio-container .gr-accordion,
.gradio-container .gr-accordion-header {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #111111 !important;
}
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #111111 !important;
    border-color: #111111 !important;
}
.prose, .markdown-text, .md,
.prose > *, .markdown-text > * {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.dark .col-left {
    border-left-color: #111111 !important;
}
.dark .col-right {
    border-left-color: #111111 !important;
}
"""
