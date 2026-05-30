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

os.environ["TRACKIO_PROJECT"] = TRACKIO_PROJECT


class TerminusHarnessEnvironment:
    """Small TRL environment wrapper backed by an OpenEnv resource session."""

    def __init__(self, session_factory: TerminusSessionFactory):
        self._session_factory = session_factory
        self._session = None

    def terminal(self, command: str = "", final_answer: str = "") -> str:
        """Run a shell command or submit final_answer inside Terminus.

        Args:
            command: Shell command to run in the sandbox.
            final_answer: Final answer to submit after the task is complete.

        Returns:
            A JSON string with the tool output, reward, done flag, and error.
        """
        command = str(command or "")
        final_answer = str(final_answer or "")
        arguments = {
            key: value
            for key, value in (("command", command), ("final_answer", final_answer))
            if value.strip()
        }

        if self._session is None:
            return json.dumps(
                {
                    "tool_name": "terminal",
                    "arguments": arguments,
                    "done": True,
                    "error": "environment was not reset",
                    "output": "",
                    "reward": 0.0,
                },
                sort_keys=True,
            )

        if command.strip() and final_answer.strip():
            first = self._session.call_tool("terminal", {"command": command})
            result = first
            if not first.done:
                result = self._session.call_tool(
                    "terminal",
                    {"final_answer": final_answer},
                )
                first_data = first.data if isinstance(first.data, dict) else {}
                result_data = result.data if isinstance(result.data, dict) else {}
                result.data = {
                    **result_data,
                    "output": "\n".join(
                        str(part)
                        for part in (
                            first_data.get("output"),
                            result_data.get("output"),
                        )
                        if part
                    ),
                }
        elif command.strip():
            result = self._session.call_tool("terminal", {"command": command})
        elif final_answer.strip():
            result = self._session.call_tool(
                "terminal",
                {"final_answer": final_answer},
            )
        else:
            result = self._session.call_tool("terminal", {"command": ""})
            arguments = {"command": ""}

        reward = result.metadata.get("reward")
        data = result.data if isinstance(result.data, dict) else {"output": result.data}
        return json.dumps(
            {
                "tool_name": "terminal",
                "arguments": arguments,
                "done": result.done,
                "error": result.error,
                "output": data.get("output"),
                "reward": 0.0 if reward is None else float(reward),
            },
            sort_keys=True,
            default=str,
        )

    def reset(self, prompt: Any = None, **_: Any) -> None:
        self._close_session()
        self._session = self._session_factory.create(task=prompt)

    def _close_session(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    task_dataset = load_dataset(TASK_DATASET_ID, split="train")
    task = task_dataset[0]
    train_dataset = task_dataset.select_columns(["prompt"])
    session_factory = TerminusSessionFactory(
        client_factory=lambda: TerminusEnv(
            base_url=ENV_URL,
            connect_timeout_s=30.0,
            message_timeout_s=600.0,
        ).sync(),
        default_verify=list(task["verify"]),
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
            max_tool_calling_iterations=task["max_turns"],
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
        environment_factory=lambda: TerminusHarnessEnvironment(session_factory),
    )
    trainer.train()
    trainer.save_model()
    trainer.push_to_hub(commit_message=f"Async GRPO Terminus run {RUN_NAME}")


if __name__ == "__main__":
    main()
