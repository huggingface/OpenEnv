# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Transforms specific to coding environments."""

import ast
import re

from openenv.core.env_server.base_transforms import CompositeTransform
from openenv.core.env_server.interfaces import Transform
from openenv.core.env_server.types import Observation

from ..models import CodeObservation


def _parse_code(code: str) -> ast.AST:
    return ast.parse(code)


class CodeSafetyTransform(Transform):
    """
    Assign penalties for obviously unsafe coding patterns.

    This is a reward heuristic, not a security sandbox. Container isolation is
    the security boundary; this transform only shapes rewards for common cases.
    """

    def __init__(self, penalty: float = -1.0):
        self.penalty = penalty
        self._fallback_patterns = [
            (re.compile(r"\bimport\s+os\b"), "import os"),
            (re.compile(r"\bimport\s+subprocess\b"), "import subprocess"),
            (re.compile(r"\beval\s*\("), "eval"),
            (re.compile(r"\bexec\s*\("), "exec"),
            (re.compile(r"\b__import__\s*\("), "__import__"),
            (re.compile(r"\bopen\s*\("), "open"),
        ]

    def _detect_text_violation(self, code: str) -> str | None:
        for pattern, violation in self._fallback_patterns:
            if pattern.search(code):
                return violation
        return None

    def _detect_violation(self, code: str) -> str | None:
        """
        Detect dangerous operations using AST analysis.

        AST-based detection avoids false positives from harmless string literals
        (e.g. ``print("import os")``) or similarly named user functions
        (e.g. ``myopen()``).
        """
        try:
            tree = _parse_code(code)
        except (SyntaxError, RecursionError, ValueError):
            # Fall back to the previous raw-text heuristic when AST parsing
            # cannot inspect malformed or pathologically nested code.
            return self._detect_text_violation(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level_module = alias.name.split(".", 1)[0]
                    if top_level_module in {"os", "subprocess"}:
                        return f"import {top_level_module}"

            if isinstance(node, ast.ImportFrom) and node.module:
                top_level_module = node.module.split(".", 1)[0]
                if top_level_module in {"os", "subprocess"}:
                    return f"import {top_level_module}"

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_name = node.func.id
                    if called_name in {"eval", "exec", "open", "__import__"}:
                        return called_name
                if isinstance(node.func, ast.Attribute) and node.func.attr == "open":
                    return "open"

        return None

    def __call__(self, observation: Observation) -> Observation:
        if not isinstance(observation, CodeObservation):
            return observation

        if "last_code" in observation.metadata:
            code = observation.metadata["last_code"]
            violation = self._detect_violation(code)
            if violation is not None:
                observation.reward = self.penalty
                observation.metadata["safety_violation"] = violation
            elif observation.reward is None:
                observation.reward = 0.0

        return observation


class CodeQualityTransform(Transform):
    """Evaluates and rewards code quality metrics."""

    def __init__(
        self,
        concise_bonus: float = 0.1,
        max_length_threshold: int = 100,
        syntax_penalty: float = -0.2,
    ):
        self.concise_bonus = concise_bonus
        self.max_length_threshold = max_length_threshold
        self.syntax_penalty = syntax_penalty

    def __call__(self, observation: Observation) -> Observation:
        if not isinstance(observation, CodeObservation):
            return observation

        quality_score = 0.0

        if "last_code" in observation.metadata:
            code = observation.metadata["last_code"]

            # Reward concise code
            if len(code.strip()) <= self.max_length_threshold:
                quality_score += self.concise_bonus

            # Check syntax (redundant but useful for quality assessment)
            try:
                _parse_code(code)
            except (SyntaxError, RecursionError, ValueError):
                quality_score += self.syntax_penalty

        # Add to existing reward
        if observation.reward is None:
            observation.reward = quality_score
        else:
            observation.reward += quality_score

        return observation


def create_safe_coding_transform() -> CompositeTransform:
    """Create a transform focused on safe coding practices and quality."""
    return CompositeTransform([CodeSafetyTransform(), CodeQualityTransform()])
