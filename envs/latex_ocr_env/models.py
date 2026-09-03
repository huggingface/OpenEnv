# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the LaTeX OCR Environment.

The environment presents a rendered math/text image (from a Hugging Face
dataset) and the agent must return the LaTeX source that produced it. Reward
is computed server-side against the hidden ground-truth LaTeX.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class LatexOCRAction(Action):
    """Agent's transcription attempt for the current image.

    A single ``latex`` string. Episodes are single-step (bandit): one action
    per reset terminates the episode.
    """

    latex: str = Field(
        default="",
        description="Predicted LaTeX source for the current image.",
    )


class LatexOCRObservation(Observation):
    """Observation for the LaTeX OCR environment.

    On ``reset`` the observation carries the image the agent must transcribe;
    ``target_latex`` is intentionally empty so the agent cannot cheat. On the
    terminal ``step`` observation the ground truth and scoring details are
    revealed for logging/training.
    """

    # --- Present on reset (the task prompt) ---
    image_base64: str = Field(
        default="",
        description="Base64-encoded image (PNG) the agent must transcribe.",
    )
    image_format: str = Field(
        default="png", description="Encoding format of image_base64."
    )
    prompt: str = Field(
        default="",
        description="Instruction shown to the agent describing the OCR task.",
    )
    split: str = Field(default="", description="Dataset split of the current task.")
    index: int = Field(
        default=-1, description="Row index (materialize) or cursor position (stream)."
    )
    task_id: str = Field(default="", description="Stable task identifier.")

    # --- Streaming-mode progress (present when mode='stream') ---
    total: int = Field(default=-1, description="Total rows in the split (denominator).")
    remaining: int = Field(default=-1, description="Rows left in this stream.")
    pct_done: float = Field(
        default=0.0, description="Fraction of the split consumed so far."
    )
    exhausted: bool = Field(
        default=False, description="True once the stream is fully consumed."
    )

    # --- Present after step (grading result) ---
    predicted_latex: str = Field(
        default="", description="The LaTeX the agent submitted."
    )
    target_latex: str = Field(
        default="",
        description="Ground-truth LaTeX. Empty until the episode terminates.",
    )
    exact_match: bool = Field(
        default=False,
        description="Whether the normalized prediction equals the target.",
    )
    char_error_rate: float = Field(
        default=1.0,
        description="Normalized Levenshtein distance between prediction and target.",
    )
