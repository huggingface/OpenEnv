# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reward computation for the LaTeX OCR environment.

A smooth, dense reward: normalized character edit distance with a small
exact-match bonus, which is the standard shape for OCR-style RL where an
exact-string reward alone is too sparse (and discontinuous auxiliary terms make
training spiky):

    reward = [(1 - exact_weight) * (1 - CER) + exact_weight * exact_match] * length_factor

where ``CER`` is the normalized character error rate (Levenshtein distance over
the longer whitespace-stripped string). A perfect transcription scores 1.0; a
partial one is capped at ``1 - exact_weight``. Whitespace-insensitive, since
LaTeX spacing is cosmetic.

**Length guard.** Because the score is whitespace-insensitive, a policy could
otherwise emit the correct answer followed by unlimited whitespace and still
score 1.0 — a reward hack that shows up in RL as completions drifting to the
generation-length cap ("correct LaTeX + padding"). ``length_factor`` closes it:
the reward decays once the *raw* prediction grows past a multiple of the target
length (padding is measured before it is stripped away). Normal LaTeX spacing
stays well within the allowance, so legitimate answers are unaffected.

The rubric also cleans the raw completion itself (strips markdown code fences
and surrounding math delimiters) so it can score the model's raw output
directly and measure its true length. Cleaning is idempotent, so callers that
pre-clean their text still work unchanged.

Implemented with a pure-Python Levenshtein so the environment carries no extra
runtime dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_WHITESPACE = re.compile(r"\s+")
_FENCE = re.compile(r"```[a-zA-Z]*\s*(.*?)\s*```", re.DOTALL)

# Length-guard defaults. A correct transcription is about as long as the target;
# allow up to ``ratio`` times the target length (with a small floor so very short
# targets get breathing room) before the reward starts to decay.
_DEFAULT_OVERLONG_RATIO = 4.0
_DEFAULT_OVERLONG_FLOOR = 80


def normalize_latex(text: str) -> str:
    """Collapse whitespace runs to single spaces and strip."""
    return _WHITESPACE.sub(" ", text or "").strip()


def _strip_all_whitespace(text: str) -> str:
    return _WHITESPACE.sub("", text or "")


def clean_latex(text: str) -> str:
    """Strip markdown code fences and surrounding math delimiters from a completion.

    Idempotent: already-clean LaTeX is returned unchanged. Applied inside the rubric so it
    can score the model's raw output directly (and measure its true length for the guard).
    """
    text = (text or "").strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    return text.strip().strip("$").strip()


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings (pure Python, O(len(a) * len(b)))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


@dataclass
class GradeResult:
    reward: float
    exact_match: bool
    char_error_rate: float
    length_factor: float = (
        1.0  # < 1.0 when the raw prediction was penalized for being overlong
    )


class LatexOCRRubric:
    """Normalized edit distance + exact-match bonus, with a length guard against padding.

    Args:
        exact_weight: fraction of the reward reserved for an exact match. With
            the default 0.4, partial answers score in ``[0, 0.6]`` and only an
            exact (whitespace-insensitive) match reaches ``1.0``.
        overlong_ratio: the raw prediction may be up to ``overlong_ratio`` times
            the target length before the reward decays; it reaches 0 at twice the
            allowance. Set ``<= 0`` to disable the length guard.
        overlong_floor: minimum allowed raw length (chars), so very short targets
            still tolerate normal formatting before the guard engages.
    """

    def __init__(
        self,
        exact_weight: float = 0.4,
        overlong_ratio: float = _DEFAULT_OVERLONG_RATIO,
        overlong_floor: int = _DEFAULT_OVERLONG_FLOOR,
    ) -> None:
        if not 0.0 <= exact_weight <= 1.0:
            raise ValueError("exact_weight must be in [0, 1]")
        self.exact_weight = exact_weight
        self.overlong_ratio = overlong_ratio
        self.overlong_floor = overlong_floor

    def grade(self, prediction: str, target: str) -> GradeResult:
        raw = prediction or ""
        # Clean the raw completion (fences, $-delimiters) before scoring; LaTeX spacing is
        # cosmetic, so score on the whitespace-stripped form: an exact match => CER 0.
        cleaned = clean_latex(raw)
        pred_canon = _strip_all_whitespace(normalize_latex(cleaned))
        target_norm = normalize_latex(target)
        target_canon = _strip_all_whitespace(target_norm)

        exact = pred_canon == target_canon

        # Length guard: penalize predictions whose RAW length (measured before whitespace is
        # stripped, so padding counts) far exceeds the target's. Compute this before edit
        # distance so inputs whose reward is already guaranteed to be zero cannot trigger
        # quadratic work.
        length_factor = 1.0
        if self.overlong_ratio > 0:
            allowed = max(self.overlong_floor, self.overlong_ratio * len(target_norm))
            if len(raw) > allowed:
                over = (len(raw) - allowed) / allowed
                length_factor = max(0.0, 1.0 - over)

        if length_factor == 0.0:
            return GradeResult(
                reward=0.0,
                exact_match=exact,
                char_error_rate=0.0 if exact else 1.0,
                length_factor=0.0,
            )

        if not target_canon:
            cer = 0.0 if not pred_canon else 1.0
        else:
            distance = levenshtein(pred_canon, target_canon)
            cer = min(1.0, distance / max(len(pred_canon), len(target_canon)))

        similarity = 1.0 - cer
        reward = (1.0 - self.exact_weight) * similarity + self.exact_weight * float(
            exact
        )
        reward *= length_factor

        return GradeResult(
            reward=round(reward, 6),
            exact_match=exact,
            char_error_rate=round(cer, 6),
            length_factor=round(length_factor, 6),
        )
