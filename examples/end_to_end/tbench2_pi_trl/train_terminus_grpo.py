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
"""Run Terminus async GRPO with PI rollouts owned by TRL."""

from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset
from terminus_env.client import TerminusEnv
from terminus_env.harness import TerminusSessionFactory
from transformers import AutoTokenizer
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer

from pi_rollout_worker import TerminusPiRolloutWorker, WorkerConfig

TASK_DATASET_ID = "burtenshaw/terminus-pi-trl-tasks"
MODEL = "Qwen/Qwen3-4B"
ENV_URL = os.environ.get("TERMINUS_ENV_URL", "http://localhost:8000")
OUTPUT_DIR = Path(os.environ.get("TERMINUS_OUTPUT_DIR", "/tmp/terminus-pi-trl-output"))
HUB_MODEL_ID = os.environ.get(
    "TERMINUS_HUB_MODEL_ID",
    "burtenshaw/terminus-pi-trl-async-grpo-qwen3-4b",
)
TRACKIO_PROJECT = "terminus-pi-trl"
TRACKIO_SPACE_ID = os.environ.get(
    "TRACKIO_SPACE_ID",
    "burtenshaw/terminus-pi-trl-trackio",
)
REPORT_TO = "trackio"
RUN_NAME = os.environ.get("TERMINUS_RUN_NAME") or (
    os.environ.get("JOB_ID", "local") + "-terminus"
)
VLLM_SERVER_URL = os.environ.get("TERMINUS_VLLM_SERVER_URL", "http://localhost:8001")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "openenv")

os.environ["TRACKIO_PROJECT"] = TRACKIO_PROJECT


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

    rank = int(
        os.environ.get("RANK")
        or os.environ.get("ACCELERATE_PROCESS_INDEX")
        or os.environ.get("SLURM_PROCID")
        or "0"
    )
    worker = None
    if rank == 0:
        worker = TerminusPiRolloutWorker(
            session_factory=session_factory,
            tasks=list(task_dataset),
            tokenizer=tokenizer,
            vllm_base_url=VLLM_SERVER_URL,
            vllm_model=MODEL,
            vllm_api_key=VLLM_API_KEY,
            chat_template_kwargs={"enable_thinking": False},
            config=WorkerConfig(
                max_inflight=task["num_generations"],
                max_turns=task["max_turns"],
                max_completion_tokens=task["max_completion_length"],
            ),
        )

    def unused_reward(**kwargs: object) -> list[float]:
        return [0.0] * len(kwargs.get("prompts", []))

    trainer = AsyncGRPOTrainer(
        model=MODEL,
        args=AsyncGRPOConfig(
            output_dir=str(OUTPUT_DIR),
            max_steps=task["max_steps"],
            per_device_train_batch_size=task["batch_size"],
            gradient_accumulation_steps=1,
            num_generations=task["num_generations"],
            max_completion_length=task["max_completion_length"],
            max_inflight_tasks=task["num_generations"],
            learning_rate=1e-6,
            temperature=1.0,
            weight_sync_steps=1,
            logging_steps=1,
            logging_strategy="steps",
            log_completions=True,
            report_to=REPORT_TO,
            run_name=RUN_NAME,
            project=TRACKIO_PROJECT,
            trackio_space_id=TRACKIO_SPACE_ID,
            save_strategy="no",
            push_to_hub=True,
            hub_model_id=HUB_MODEL_ID,
            chat_template_kwargs={"enable_thinking": False},
            vllm_server_base_url=VLLM_SERVER_URL,
            request_timeout=600,
            vllm_server_timeout=600,
        ),
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=unused_reward,
        rollout_worker=worker,
    )
    trainer.train()
    trainer.save_model()
    trainer.push_to_hub(commit_message=f"Async GRPO Terminus run {RUN_NAME}")


if __name__ == "__main__":
    main()
