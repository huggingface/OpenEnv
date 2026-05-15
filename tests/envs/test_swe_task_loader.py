# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for SWE-bench Lite -> SWETask adaptation."""

from __future__ import annotations

import json
import random

import pytest
from mini_swe_env.task_loader_swebench_lite import (
    adapt_swebench_lite_row,
    coerce_swe_task,
    deterministic_train_eval_split,
    SOURCE_NAME,
    SWEBenchLiteAdapterError,
    SWETask,
    SWETaskValidationError,
)


def test_row_to_swetask_conversion_uses_fail_to_pass_for_verify() -> None:
    row = {
        "repo": "psf/requests",
        "instance_id": "requests__requests-12345",
        "base_commit": "a" * 40,
        "problem_statement": "Fix redirect edge case in Session.send.",
        "FAIL_TO_PASS": json.dumps(
            [
                "tests/test_sessions.py::test_redirect_loop",
                "tests/test_sessions.py::test_strip_auth",
            ]
        ),
        "PASS_TO_PASS": json.dumps(["tests/test_api.py::test_get"]),
        "created_at": "2024-01-01",
        "version": "1.0",
    }

    task = adapt_swebench_lite_row(row)

    assert task.source == SOURCE_NAME
    assert task.task_id == f"{SOURCE_NAME}::{row['instance_id']}"
    assert task.instance_id == row["instance_id"]
    assert task.repo == row["repo"]
    assert task.base_commit == row["base_commit"]
    assert task.setup == []
    assert task.verify == [
        "python -m pytest -q tests/test_sessions.py::test_redirect_loop",
        "python -m pytest -q tests/test_sessions.py::test_strip_auth",
    ]
    assert task.metadata["dataset"] == SOURCE_NAME
    assert task.metadata["fail_to_pass_count"] == 2
    assert task.metadata["pass_to_pass_count"] == 1


def test_deterministic_split_is_stable_for_seed_and_input_order() -> None:
    tasks = [
        SWETask(
            task_id=f"{SOURCE_NAME}::id-{idx:03d}",
            source=SOURCE_NAME,
            instance_id=f"id-{idx:03d}",
            repo="example/repo",
            base_commit="b" * 40,
            instruction=f"Task {idx}",
            setup=[],
            verify=["python -m pytest -q tests/test_sample.py::test_ok"],
        )
        for idx in range(40)
    ]

    train_a, eval_a = deterministic_train_eval_split(
        tasks,
        subset_size=20,
        train_size=16,
        seed=99,
    )

    shuffled = tasks[:]
    random.Random(123).shuffle(shuffled)
    train_b, eval_b = deterministic_train_eval_split(
        shuffled,
        subset_size=20,
        train_size=16,
        seed=99,
    )

    assert [task.task_id for task in train_a] == [task.task_id for task in train_b]
    assert [task.task_id for task in eval_a] == [task.task_id for task in eval_b]


def test_row_conversion_handles_unittest_style_names() -> None:
    row = {
        "repo": "django/django",
        "instance_id": "django__django-12915",
        "base_commit": "d" * 40,
        "problem_statement": "Add async handler method.",
        "FAIL_TO_PASS": json.dumps(
            [
                "test_get_async_response (staticfiles_tests.test_handlers.TestASGIStaticFilesHandler)"
            ]
        ),
    }

    task = adapt_swebench_lite_row(row)

    assert task.verify == [
        "python -m pytest -q "
        "staticfiles_tests/test_handlers.py::TestASGIStaticFilesHandler::test_get_async_response"
    ]


def test_schema_validation_failures_are_explicit() -> None:
    with pytest.raises(
        SWETaskValidationError, match="verify must contain at least one"
    ):
        coerce_swe_task(
            {
                "task_id": "x",
                "source": SOURCE_NAME,
                "instance_id": "x",
                "repo": "org/repo",
                "base_commit": "c" * 40,
                "instruction": "Do something",
                "setup": [],
                "verify": [],
            }
        )


def test_adapter_error_is_explicit_when_required_fields_missing() -> None:
    with pytest.raises(SWEBenchLiteAdapterError, match="missing required field"):
        adapt_swebench_lite_row({"instance_id": "missing_repo"})
