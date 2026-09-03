# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LaTeX OCR Environment.

A dataset-backed, single-step (bandit) RL environment over a Hugging Face
dataset of (image, LaTeX) pairs. Reward is computed against the hidden
ground-truth LaTeX via :class:`LatexOCRRubric`.

Two access modes (env var ``LATEX_OCR_MODE``):

- ``materialize`` (default) — the split is loaded and indexed. Supports the full
  Task API with random access: ``reset(split, index)`` selects any row. Best
  when the dataset fits on disk and is re-read across epochs.

- ``stream`` — a sequential **cursor** over a streamed split (no full download).
  ``reset()`` pulls the *next* sample; every observation carries progress
  (``index``, ``total``, ``remaining``, ``pct_done``). No-repeat within a pass;
  ``reset(index=i)`` random access is unsupported by design. Best for very large
  / TB-scale datasets.

The dataset (materialize) and the split row-count (stream) are cached at module
scope because the server instantiates a fresh env per Task API call.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import random
from functools import lru_cache
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from PIL import Image

try:
    from ..models import LatexOCRAction, LatexOCRObservation
except ImportError:  # standalone (uvicorn server.app:app)
    from models import LatexOCRAction, LatexOCRObservation

from .rubric import LatexOCRRubric

logger = logging.getLogger(__name__)

DEFAULT_DATASET = os.environ.get("LATEX_OCR_DATASET", "unsloth/LaTeX_OCR")
IMAGE_COLUMN = os.environ.get("LATEX_OCR_IMAGE_COLUMN", "image")
TEXT_COLUMN = os.environ.get("LATEX_OCR_TEXT_COLUMN", "text")
DEFAULT_MODE = os.environ.get("LATEX_OCR_MODE", "materialize")  # materialize | stream
SEED = int(os.environ.get("LATEX_OCR_SEED", "0"))
SHUFFLE_BUFFER = int(os.environ.get("LATEX_OCR_SHUFFLE_BUFFER", "0"))
CONFIG = os.environ.get("LATEX_OCR_CONFIG") or None
DEFAULT_PROMPT = os.environ.get(
    "LATEX_OCR_PROMPT",
    "Transcribe the mathematical expression in this image into LaTeX. "
    "Return only the LaTeX source, with no surrounding text or code fences.",
)


def _configured_splits() -> list[str]:
    raw = os.environ.get("LATEX_OCR_SPLITS", "train,test")
    return [s.strip() for s in raw.split(",") if s.strip()]


def _max_rows() -> int | None:
    """Per-split row cap (LATEX_OCR_MAX_ROWS), honored in both modes. None = no cap."""
    v = os.environ.get("LATEX_OCR_MAX_ROWS")
    return int(v) if v else None


# Safety cap on get_task_range() stub generation in stream mode: the metadata
# row-count can be enormous, so an unbounded range must not build a giant list.
_STREAM_RANGE_CAP = int(os.environ.get("LATEX_OCR_STREAM_RANGE_CAP", "100000"))


@lru_cache(maxsize=None)
def _load_split(dataset_name: str, split: str):
    """Load + cache a full split (materialize mode) for the process lifetime."""
    from datasets import load_dataset

    logger.info("Materializing %s split=%s ...", dataset_name, split)
    ds = load_dataset(
        dataset_name, name=CONFIG, split=split, token=os.environ.get("HF_TOKEN")
    )
    max_rows = os.environ.get("LATEX_OCR_MAX_ROWS")
    if max_rows:
        ds = ds.select(range(min(int(max_rows), len(ds))))
    return ds


@lru_cache(maxsize=None)
def _split_total(dataset_name: str, split: str) -> int:
    """Row count from dataset metadata — cheap, no data download (stream mode)."""
    from datasets import load_dataset_builder

    builder = load_dataset_builder(
        dataset_name, name=CONFIG, token=os.environ.get("HF_TOKEN")
    )
    info = builder.info.splits.get(split)
    if info is None or info.num_examples is None:
        return -1
    return int(info.num_examples)


def _encode_image(image: Any) -> str:
    # Normalize to PNG so the hard-coded image_format="png" is always correct,
    # even when a swapped-in dataset stores raw (e.g. JPEG) bytes in its image column.
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(bytes(image)))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class LatexOCREnvironment(Environment):
    """Single-step LaTeX OCR environment (materialize or streaming cursor)."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET,
        exact_weight: float | None = None,
        mode: str = DEFAULT_MODE,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.mode = mode
        # Exact-match share of the reward; env-var tunable (default 0.4 -> 0.6/0.4).
        if exact_weight is None:
            exact_weight = float(os.environ.get("LATEX_OCR_EXACT_WEIGHT", "0.4"))
        # Length guard against whitespace-padding reward hacks; env-var tunable.
        overlong_ratio = float(os.environ.get("LATEX_OCR_OVERLONG_RATIO", "4.0"))
        overlong_floor = int(os.environ.get("LATEX_OCR_OVERLONG_FLOOR", "80"))
        self._rubric = LatexOCRRubric(
            exact_weight=exact_weight,
            overlong_ratio=overlong_ratio,
            overlong_floor=overlong_floor,
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._target: Optional[str] = None
        self._current_split: str = ""
        self._current_index: int = -1
        self._current_task_id: str = ""
        self._done = False
        # streaming cursor state (per session/instance)
        self._stream = None
        self._stream_split: str = ""
        self._cursor = 0
        self._exhausted = False

    # ------------------------------------------------------------------ #
    # Task API                                                            #
    # ------------------------------------------------------------------ #
    def list_splits(self) -> list[str]:
        return _configured_splits()

    def _validate_split(self, split: str) -> None:
        splits = self.list_splits()
        if split not in splits:
            raise ValueError(f"unknown split {split!r}; configured splits: {splits}")

    def num_tasks(self, split: str) -> int:
        self._validate_split(split)
        # Honest denominator in both modes; stream reads metadata only.
        if self.mode == "stream":
            total = _split_total(self.dataset_name, split)
            cap = _max_rows()
            if cap is None:
                return total
            # A missing `num_examples` gives total == -1, but the cursor still stops at the cap,
            # so the cap IS the number of tasks this split will serve. Returning -1 there reports
            # "unknown" for a quantity we know exactly.
            return cap if total < 0 else min(total, cap)
        return len(_load_split(self.dataset_name, split))

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        self._validate_split(split)
        if self.mode == "stream":
            raise NotImplementedError(
                "list_tasks is unavailable in stream mode; use num_tasks and "
                "get_task_range for bounded discovery"
            )
        n = self.num_tasks(split)
        return [{"id": f"{split}-{i}", "index": i} for i in range(n)]

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        n = self.num_tasks(split)
        if index < 0 or (n >= 0 and index >= n):
            raise IndexError(f"index {index} out of range for split {split} (n={n})")
        task = {"id": f"{split}-{index}", "index": index, "split": split}
        if self.mode == "stream":
            # Position metadata only; stream serves rows sequentially via reset().
            task["sequential"] = True
        return task

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        n = self.num_tasks(split)
        if n >= 0:
            start, stop, _ = slice(start, stop).indices(n)
        else:
            if (start is not None and start < 0) or (stop is not None and stop < 0):
                raise ValueError(
                    "negative task range bounds require a known split size"
                )
            if stop is None:
                raise ValueError("stop is required when the split size is unknown")
            start = 0 if start is None else start

        # In stream mode `n` comes from metadata and can be enormous. Reject
        # oversized requests rather than silently returning an incomplete slice.
        if self.mode == "stream" and stop - start > _STREAM_RANGE_CAP:
            raise ValueError(
                f"stream task ranges may contain at most {_STREAM_RANGE_CAP} tasks"
            )
        tasks = [
            {"id": f"{split}-{i}", "index": i, "split": split}
            for i in range(start, stop)
        ]
        if self.mode == "stream":
            for task in tasks:
                task["sequential"] = True
        return tasks

    # ------------------------------------------------------------------ #
    # Episode                                                             #
    # ------------------------------------------------------------------ #
    def reset(
        self,
        split: str = "train",
        index: int | None = None,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> LatexOCRObservation:
        self._validate_split(split)
        self._done = False
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        if self.mode == "stream":
            if index is not None:
                raise ValueError(
                    "Random-access reset(index=...) is not supported in stream mode; "
                    "use materialize mode (LATEX_OCR_MODE=materialize) for indexed access."
                )
            return self._reset_stream(split)
        return self._reset_materialize(split, index, seed)

    def _reset_materialize(
        self, split: str, index: int | None, seed: int | None
    ) -> LatexOCRObservation:
        ds = _load_split(self.dataset_name, split)
        n = len(ds)
        if n == 0:
            raise IndexError(
                f"split {split!r} is empty (n=0); check the dataset or LATEX_OCR_MAX_ROWS"
            )
        if index is None:
            index = random.Random(seed).randrange(n)
        if index < 0 or index >= n:
            raise IndexError(f"index {index} out of range for split {split} (n={n})")
        row = ds[int(index)]
        self._target = str(row[TEXT_COLUMN])
        self._current_split, self._current_index = split, int(index)
        self._current_task_id = f"{split}-{index}"
        return LatexOCRObservation(
            done=False,
            image_base64=_encode_image(row[IMAGE_COLUMN]),
            image_format="png",
            prompt=DEFAULT_PROMPT,
            split=split,
            index=int(index),
            task_id=f"{split}-{index}",
        )

    def _ensure_stream(self, split: str) -> None:
        if (
            self._stream is not None
            and self._stream_split == split
            and not self._exhausted
        ):
            return
        from datasets import load_dataset

        ds = load_dataset(
            self.dataset_name,
            name=CONFIG,
            split=split,
            streaming=True,
            token=os.environ.get("HF_TOKEN"),
        )
        if SHUFFLE_BUFFER > 0:
            ds = ds.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)
        self._stream = iter(ds)
        self._stream_split = split
        self._cursor = 0
        self._exhausted = False

    def _stream_exhausted(self, split: str, total: int) -> LatexOCRObservation:
        # Mark the episode finished and clear the target so a following step() fails
        # fast instead of grading against the previous (stale) task.
        self._exhausted = True
        self._done = True
        self._target = None
        return LatexOCRObservation(
            done=True,
            split=split,
            index=self._cursor,
            total=total,
            remaining=0,
            pct_done=1.0,
            exhausted=True,
            task_id=f"{split}-stream-end",
        )

    def _reset_stream(self, split: str) -> LatexOCRObservation:
        self._ensure_stream(split)
        total = self.num_tasks(split)  # honors LATEX_OCR_MAX_ROWS
        cap = _max_rows()
        if cap is not None and self._cursor >= cap:
            return self._stream_exhausted(split, total)
        try:
            row = next(self._stream)
        except StopIteration:
            return self._stream_exhausted(split, total)
        self._cursor += 1
        self._target = str(row[TEXT_COLUMN])
        self._current_split, self._current_index = split, self._cursor
        self._current_task_id = f"{split}-stream-{self._cursor}"
        remaining = (total - self._cursor) if total > 0 else -1
        return LatexOCRObservation(
            done=False,
            image_base64=_encode_image(row[IMAGE_COLUMN]),
            image_format="png",
            prompt=DEFAULT_PROMPT,
            split=split,
            index=self._cursor,
            task_id=f"{split}-stream-{self._cursor}",
            total=total,
            remaining=remaining,
            pct_done=round(self._cursor / total, 6) if total > 0 else 0.0,
            exhausted=False,
        )

    def step(self, action: LatexOCRAction, **kwargs: Any) -> LatexOCRObservation:
        if self._target is None:
            raise RuntimeError("step() called before reset()")
        if self._done:
            raise RuntimeError("Episode already terminated; call reset() first")
        prediction = action.latex if isinstance(action, LatexOCRAction) else ""
        result = self._rubric.grade(prediction, self._target)
        self._state.step_count += 1
        self._done = True

        # `num_tasks`, not `_split_total`: the latter ignores LATEX_OCR_MAX_ROWS, so with a cap
        # configured `total` jumped from the capped value reset() reported to the full dataset size
        # the moment a prediction was graded.
        total = self.num_tasks(self._current_split) if self.mode == "stream" else -1
        return LatexOCRObservation(
            done=True,
            reward=result.reward,
            split=self._current_split,
            index=self._current_index,
            task_id=self._current_task_id,
            predicted_latex=prediction,
            target_latex=self._target,
            exact_match=result.exact_match,
            char_error_rate=result.char_error_rate,
            total=total,
            remaining=(total - self._cursor)
            if (self.mode == "stream" and total > 0)
            else -1,
            pct_done=round(self._cursor / total, 6)
            if (self.mode == "stream" and total > 0)
            else 0.0,
            metadata={
                "exact_match": result.exact_match,
                "char_error_rate": result.char_error_rate,
                "mode": self.mode,
            },
        )

    @property
    def state(self) -> State:
        return self._state
