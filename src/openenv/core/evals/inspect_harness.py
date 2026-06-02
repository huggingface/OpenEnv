# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inspect AI harness integration for OpenEnv.

Requires the ``inspect-ai`` package: ``pip install 'inspect-ai>=0.3.0'``
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from openenv.core.evals.base import EvalHarness


class InspectAIHarness(EvalHarness):
    """Evaluation harness wrapping Inspect AI's ``eval()`` function.

    All ``inspect_ai`` imports are deferred to :meth:`run` so this class is
    importable without inspect-ai installed.  An ``ImportError`` with a clear
    message is raised at call time if the dependency is missing.

    Args:
        log_dir: Directory for evaluation log output. Defaults to None
            (Inspect AI writes logs to its default location).

    ``eval_parameters`` keys accepted by :meth:`run`:

    +--------------------------+----------+-----------------+-----------------------------------+
    | Key                      | Type     | Default         | Purpose                           |
    +==========================+==========+=================+===================================+
    | ``model``                | str      | *required*      | Model string, e.g. "openai/gpt-4o"|
    | ``model_base_url``       | str|None | None            | Override provider base URL        |
    | ``task``                 | str|None | ``dataset`` arg | Task file path or task string     |
    | ``task_args``            | dict     | ``{}``          | Arguments to pass to the task     |
    | ``max_samples``          | int|None | None            | Limit samples per task            |
    | ``max_connections``      | int|None | None            | Max concurrent model connections  |
    | ``temperature``          | float|None| None           | Model generation temperature      |
    | ``max_tokens``           | int|None | None            | Max generation tokens             |
    | ``epochs``               | int|None | None            | Number of evaluation epochs       |
    | ``solver``               | list|None| None            | Solver pipeline override          |
    | ``scorer``               | list|None| None            | Scorer override                   |
    | ``model_args``           | dict     | ``{}``          | Provider-specific model kwargs    |
    +--------------------------+----------+-----------------+-----------------------------------+
    """

    def __init__(
        self,
        *,
        log_dir: Optional[str] = None,
    ):
        self.log_dir = log_dir

    def run(
        self,
        harness_version: str,
        library_versions: Dict[str, str],
        dataset: str,
        eval_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run an Inspect AI evaluation.

        Args:
            harness_version: Version of inspect-ai being used.
            library_versions: Versions of supporting libraries.
            dataset: Default task string (used when ``task`` is not specified
                in *eval_parameters*).
            eval_parameters: See class docstring for accepted keys.

        Returns:
            Dictionary mapping metric names to scores.

        Raises:
            ImportError: If ``inspect-ai`` is not installed.
            ValueError: If ``model`` is missing from *eval_parameters*.
            RuntimeError: If the evaluation fails (log status is not "success").
        """
        try:
            from inspect_ai import eval as inspect_eval
        except ImportError:
            raise ImportError(
                "inspect-ai is required for InspectAIHarness. "
                "Install it with: pip install 'inspect-ai>=0.3.0'"
            )

        # Extract required model parameter
        model = eval_parameters.get("model")
        if model is None:
            raise ValueError(
                "eval_parameters must include 'model' "
                "(e.g. 'openai/gpt-4o', 'hf/meta-llama/...')."
            )

        # Task: explicit parameter or fall back to dataset
        task = eval_parameters.get("task", dataset)

        # Build eval kwargs
        eval_kwargs: Dict[str, Any] = {}

        task_args = eval_parameters.get("task_args", {})
        if task_args:
            eval_kwargs["task_args"] = task_args

        model_args = eval_parameters.get("model_args", {})
        if model_args:
            eval_kwargs["model_args"] = model_args

        for key in (
            "max_samples",
            "max_connections",
            "temperature",
            "max_tokens",
            "epochs",
        ):
            value = eval_parameters.get(key)
            if value is not None:
                eval_kwargs[key] = value

        if eval_parameters.get("solver") is not None:
            eval_kwargs["solver"] = eval_parameters["solver"]

        if eval_parameters.get("scorer") is not None:
            eval_kwargs["scorer"] = eval_parameters["scorer"]

        if self.log_dir is not None:
            eval_kwargs["log_dir"] = self.log_dir

        model_base_url = eval_parameters.get("model_base_url")
        if model_base_url is not None:
            eval_kwargs["model_base_url"] = model_base_url

        # Run evaluation
        logs = inspect_eval(task, model=model, **eval_kwargs)

        # Extract results from the first log
        if not logs:
            raise RuntimeError(
                "Inspect AI evaluation returned no logs. "
                "Check that the task and model arguments are valid."
            )
        log = logs[0]
        if log.status != "success":
            raise RuntimeError(
                f"Inspect AI evaluation failed with status: {log.status}"
            )

        return self._extract_scores(log)

    def _extract_scores(self, log: Any) -> Dict[str, Any]:
        """Parse an EvalLog's results into a flat score dictionary.

        Iterates over ``log.results.scores`` (a list of ``EvalScore``),
        flattening each scorer's ``metrics`` dict into a single output dict.

        Args:
            log: An ``inspect_ai`` ``EvalLog`` object.

        Returns:
            Dictionary mapping metric names to their values.
        """
        scores: Dict[str, Any] = {}
        if log.results is None:
            return scores

        for eval_score in log.results.scores:
            for metric_name, metric in eval_score.metrics.items():
                scores[metric_name] = metric.value

        return scores
