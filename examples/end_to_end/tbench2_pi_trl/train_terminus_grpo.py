#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "aiohttp>=3.9.0",
#   "accelerate>=1.10.0",
#   "datasets>=3.0.0",
#   "huggingface-hub>=0.35.0",
#   "openenv-core",
#   "openenv-terminus-env",
#   "peft>=0.17.0",
#   "trackio<0.25.0",
#   "transformers @ git+https://github.com/huggingface/transformers.git@e1a37d29cd4822d74f4f3323289fb69e1eec61a0",
#   "trl @ git+https://github.com/huggingface/trl.git@a7ba987d05b1e9dbbdbd2e9091264623746e3528",
#   "vllm==0.19.1",
# ]
# [tool.uv.sources]
# openenv-core = { git = "https://github.com/burtenshaw/OpenEnv.git", branch = "codex/terminus-pi-trl-space" }
# openenv-terminus-env = { git = "https://github.com/burtenshaw/OpenEnv.git", branch = "codex/terminus-env-harness", subdirectory = "envs/terminus_env" }
# ///
"""Run the Terminus async GRPO environment-factory training example."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openenv.core.harness import HarnessRunLimits, PiCLIHarnessAdapter
from terminus_env.client import TerminusEnv
from terminus_env.harness import TerminusSessionFactory, terminus_reward
from transformers import AutoTokenizer
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer

TASK_DATASET_ID = "burtenshaw/terminus-pi-trl-tasks"
MODEL = "Qwen/Qwen3-4B"
ENV_URL = os.environ.get("TERMINUS_ENV_URL", "http://localhost:8000")
OUTPUT_DIR = Path(os.environ.get("TERMINUS_OUTPUT_DIR", "/tmp/terminus-pi-trl-output"))
HUB_MODEL_ID = "burtenshaw/terminus-pi-trl-async-grpo-qwen3-4b"
TRACKIO_PROJECT = "terminus-pi-trl"
REPORT_TO = "trackio"
RUN_NAME = os.environ.get("JOB_ID", "local") + "-terminus"
VLLM_SERVER_URL = os.environ.get("TERMINUS_VLLM_SERVER_URL", "http://localhost:8001")
PI_VLLM_BASE_URL = os.environ.get(
    "TERMINUS_PI_VLLM_BASE_URL",
    VLLM_SERVER_URL.rstrip("/") + "/v1",
)

os.environ["TRACKIO_PROJECT"] = TRACKIO_PROJECT


class TerminusHarnessEnvironment:
    """Small TRL environment wrapper backed by the Pi CLI harness."""

    def __init__(
        self,
        session_factory: TerminusSessionFactory,
        pi_harness: PiCLIHarnessAdapter,
        limits: HarnessRunLimits,
    ):
        self._session_factory = session_factory
        self._pi_harness = pi_harness
        self._limits = limits
        self._session = None

    def terminal(self, command: str = "", final_answer: str = "") -> str:
        """Compatibility tool that runs Pi against the current Terminus task."""
        del command, final_answer
        if self._session is None:
            raise RuntimeError("environment was not reset")

        rollout = self._pi_harness.run_black_box(
            session=self._session,
            limits=self._limits,
        )
        verify = self._session.verify(
            transcript=rollout.messages,
            final_state={
                "done": rollout.done,
                "metrics": dict(rollout.metrics),
            },
        )
        result = rollout.tool_trace[-1].result if rollout.tool_trace else None
        data = result.data if result is not None else {}

        return json.dumps(
            {
                "tool_name": "terminal",
                "arguments": {"harness": "pi_cli"},
                "done": verify.done or rollout.done,
                "error": None if result is None else result.error,
                "output": data.get("output") if isinstance(data, dict) else data,
                "reward": 0.0
                if verify.env_reward is None
                else float(verify.env_reward),
            },
            sort_keys=True,
            default=str,
        )

    def reset(self, prompt: Any = None, **_: Any) -> None:
        if self._session is not None:
            self._session.close()
        self._session = self._session_factory.create(task=prompt)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    task_dataset = load_dataset(TASK_DATASET_ID, split="train")
    task = task_dataset[0]
    train_dataset = task_dataset.select_columns(["prompt"])
    limits = HarnessRunLimits(max_turns=task["max_turns"])
    session_factory = TerminusSessionFactory(
        client_factory=lambda: TerminusEnv(
            base_url=ENV_URL,
            connect_timeout_s=30.0,
            message_timeout_s=600.0,
        ).sync(),
        default_verify=list(task["verify"]),
    )
    pi_harness = PiCLIHarnessAdapter(
        model=MODEL,
        model_base_url=PI_VLLM_BASE_URL,
        timeout_s=600.0,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = AsyncGRPOTrainer(
        model=MODEL,
        args=AsyncGRPOConfig(
            output_dir=str(OUTPUT_DIR),
            max_steps=task["max_steps"],
            per_device_train_batch_size=task["batch_size"],
            gradient_accumulation_steps=1,
            num_generations=task["num_generations"],
            max_completion_length=task["max_completion_length"],
            max_tool_calling_iterations=1,
            max_inflight_tasks=task["num_generations"],
            learning_rate=1e-6,
            logging_steps=1,
            logging_strategy="steps",
            log_completions=True,
            report_to=REPORT_TO,
            run_name=RUN_NAME,
            project=TRACKIO_PROJECT,
            save_strategy="no",
            push_to_hub=True,
            hub_model_id=HUB_MODEL_ID,
            chat_template_kwargs={"enable_thinking": False},
            vllm_server_base_url=VLLM_SERVER_URL,
            request_timeout=600,
        ),
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=terminus_reward,
        environment_factory=lambda: TerminusHarnessEnvironment(
            session_factory,
            pi_harness,
            limits,
        ),
    )
    trainer.train()
    trainer.save_model()
    trainer.push_to_hub(commit_message=f"Async GRPO Terminus run {RUN_NAME}")


if __name__ == "__main__":
    main()
