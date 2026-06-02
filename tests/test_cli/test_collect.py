# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the ``openenv collect`` CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from openenv.cli.__main__ import app
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def mock_pipeline():
    """Patch the collect pipeline internals so tests don't hit network or disk."""
    with (
        patch("openenv.cli.commands.collect.OpenSpielEnv") as env_cls,
        patch("openenv.cli.commands.collect.OpenSpielSessionFactory") as factory_cls,
        patch("openenv.cli.commands.collect.CollectRunner") as runner_cls,
        patch("openenv.cli.commands.collect.RolloutSerializer") as serializer_cls,
        patch("openenv.cli.commands.collect.MCPHarnessAdapter") as adapter_cls,
    ):
        collect_result = MagicMock()
        collect_result.num_collected = 3
        collect_result.num_skipped = 0
        collect_result.num_dropped = 0
        collect_result.num_failed = 0
        collect_result.avg_reward = 0.6
        collect_result.success_rate = 0.6
        collect_result.episode_ids = ["ep-000000", "ep-000001", "ep-000002"]

        runner_instance = MagicMock()
        runner_instance.run.return_value = collect_result
        runner_cls.return_value = runner_instance

        yield {
            "env_cls": env_cls,
            "factory_cls": factory_cls,
            "runner_cls": runner_cls,
            "runner_instance": runner_instance,
            "serializer_cls": serializer_cls,
            "adapter_cls": adapter_cls,
        }


def test_scripted_provider_requires_no_llm_client(tmp_path: Path, mock_pipeline):
    with patch("openenv.cli.commands.collect.create_llm_client") as create_client:
        result = runner.invoke(
            app,
            [
                "collect",
                "openspiel:tic_tac_toe",
                "--base-url",
                "https://example.hf.space",
                "--output-dir",
                str(tmp_path),
                "-n",
                "3",
                "--provider",
                "scripted",
            ],
        )

    assert result.exit_code == 0, result.output
    create_client.assert_not_called()
    mock_pipeline["runner_instance"].run.assert_called_once()


def test_openai_provider_builds_llm_client(tmp_path: Path, mock_pipeline):
    llm_client = MagicMock()
    with (
        patch(
            "openenv.cli.commands.collect.create_llm_client", return_value=llm_client
        ) as create_client,
        patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
    ):
        result = runner.invoke(
            app,
            [
                "collect",
                "openspiel:tic_tac_toe",
                "--base-url",
                "https://example.hf.space",
                "--output-dir",
                str(tmp_path),
                "-n",
                "5",
                "--provider",
                "openai",
                "--model",
                "gpt-5-mini",
            ],
        )

    assert result.exit_code == 0, result.output
    create_client.assert_called_once()
    kwargs = create_client.call_args.kwargs
    assert kwargs["provider"] == "openai"
    assert kwargs["model"] == "gpt-5-mini"
    assert kwargs["api_key"] == "sk-test"


def test_llm_endpoint_uses_llm_teacher_with_default_provider(
    tmp_path: Path, mock_pipeline
):
    llm_step = MagicMock(name="llm_step")
    scripted_step = MagicMock(name="scripted_step")

    with (
        patch(
            "openenv.cli.commands.collect._build_llm_model_step",
            return_value=llm_step,
        ) as build_llm,
        patch(
            "openenv.cli.commands.collect._build_scripted_model_step",
            return_value=scripted_step,
        ) as build_scripted,
    ):
        result = runner.invoke(
            app,
            [
                "collect",
                "openspiel:tic_tac_toe",
                "--base-url",
                "https://example.hf.space",
                "--output-dir",
                str(tmp_path),
                "--llm-endpoint",
                "http://localhost",
                "--model",
                "Qwen/Qwen2.5-7B-Instruct",
            ],
        )

    assert result.exit_code == 0, result.output
    build_llm.assert_called_once()
    build_scripted.assert_not_called()
    assert (
        mock_pipeline["runner_instance"].run.call_args.kwargs["model_step"] is llm_step
    )
    metadata = mock_pipeline[
        "serializer_cls"
    ].return_value.write_metadata.call_args.args[0]
    assert metadata["provider"] == "openai-compatible"
    assert metadata["llm_endpoint"] == "http://localhost"


def test_push_to_hub_triggers_upload(tmp_path: Path, mock_pipeline):
    with (
        patch("openenv.cli.commands.collect.push_to_hf_hub") as push_mock,
    ):
        push_mock.return_value = "https://huggingface.co/datasets/user/ttt"
        result = runner.invoke(
            app,
            [
                "collect",
                "openspiel:tic_tac_toe",
                "--base-url",
                "https://example.hf.space",
                "--output-dir",
                str(tmp_path),
                "-n",
                "3",
                "--provider",
                "scripted",
                "--push-to-hub",
                "user/ttt",
                "--private",
            ],
        )

    assert result.exit_code == 0, result.output
    push_mock.assert_called_once()
    push_kwargs = push_mock.call_args.kwargs
    assert push_kwargs["repo_id"] == "user/ttt"
    assert push_kwargs["private"] is True


def test_unknown_env_id_exits_nonzero(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "collect",
            "unknown:foo",
            "--base-url",
            "http://example",
            "--output-dir",
            str(tmp_path),
            "--provider",
            "scripted",
        ],
    )

    assert result.exit_code != 0
    assert (
        "unknown" in result.output.lower() or "not supported" in result.output.lower()
    )


def test_openai_provider_errors_without_model(tmp_path: Path, mock_pipeline):
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        result = runner.invoke(
            app,
            [
                "collect",
                "openspiel:tic_tac_toe",
                "--base-url",
                "https://example.hf.space",
                "--output-dir",
                str(tmp_path),
                "--provider",
                "openai",
            ],
        )

    assert result.exit_code != 0
    assert "model" in result.output.lower()


def test_keep_losses_disables_filter(tmp_path: Path, mock_pipeline):
    result = runner.invoke(
        app,
        [
            "collect",
            "openspiel:tic_tac_toe",
            "--base-url",
            "https://example.hf.space",
            "--output-dir",
            str(tmp_path),
            "--provider",
            "scripted",
            "--keep-losses",
        ],
    )

    assert result.exit_code == 0, result.output
    run_kwargs = mock_pipeline["runner_instance"].run.call_args.kwargs
    # When --keep-losses is passed, no filter should be installed.
    assert run_kwargs.get("should_keep") is None


def test_default_filters_losing_rollouts(tmp_path: Path, mock_pipeline):
    result = runner.invoke(
        app,
        [
            "collect",
            "openspiel:tic_tac_toe",
            "--base-url",
            "https://example.hf.space",
            "--output-dir",
            str(tmp_path),
            "--provider",
            "scripted",
        ],
    )

    assert result.exit_code == 0, result.output
    run_kwargs = mock_pipeline["runner_instance"].run.call_args.kwargs
    should_keep = run_kwargs.get("should_keep")
    assert should_keep is not None
    # Sanity: a winning record should be kept, a losing one dropped.
    winning = MagicMock(reward=1.0)
    losing = MagicMock(reward=-1.0)
    assert should_keep(winning) is True
    assert should_keep(losing) is False
