# BrowserGym Harness Rollouts

This tutorial shows how to drive BrowserGym through the OpenEnv harness runtime
when a trainer needs to keep token sampling, logprobs, and reward assignment
inside the training loop.

```{note}
Use this pattern for tool-driven BrowserGym rollouts. For a standard
`reset()` / `step()` GRPO flow, keep using the Wordle GRPO tutorial.
```

## What You'll Build

- A BrowserGym session factory that creates one environment client per rollout.
- A harness rollout function that TRL can call during training.
- A model-step wrapper that converts generated BrowserGym action text into
  structured tool calls.

## Install Dependencies

Install OpenEnv, TRL, and the BrowserGym environment package:

```bash
pip install -U "trl[vllm]" peft trackio kernels
pip install -U git+https://github.com/meta-pytorch/OpenEnv.git
pip install -U "openenv-browsergym @ git+https://huggingface.co/spaces/openenv/browsergym_env"
```

## Build The Session Factory

`BrowserGymSessionFactory` adapts a BrowserGym client into the harness
`ResourceSession` interface. If your training setup already has an
`environment_factory`, pass that factory as `client_factory` so every rollout
gets a fresh environment instance.

```python
from browsergym_env import BrowserGymEnv
from browsergym_env.harness import BrowserGymSessionFactory

space_url = "https://openenv-browsergym-env.hf.space"


def environment_factory():
    return BrowserGymEnv(base_url=space_url)


session_factory = BrowserGymSessionFactory(
    client_factory=environment_factory,
    default_task="click-test",
)
```

The session exposes BrowserGym actions such as `click`, `fill`, `send_keys`,
`scroll`, and `noop` as MCP-style tools while still executing the corresponding
BrowserGym action strings under the hood.

## Wrap TRL Generation

The harness calls a `model_step` function for each turn. The model step should
use the trainer-owned generation path, then return a `ModelStepResult` with the
completion text, token ids, logprobs, and exactly one BrowserGym tool call.

```python
from browsergym_env.harness import build_browsergym_action_tool_call
from openenv.core.harness import ModelStepResult
from openenv.core.llm_client import LLMResponse
from trl.experimental.openenv import generate_rollout_completions


def build_trl_browsergym_model_step(trainer, tokenizer):
    def model_step(messages, tools, sampling):
        del tools, sampling
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        rollout_output = generate_rollout_completions(trainer, [prompt_text])[0]
        completion_text = rollout_output.get("text") or tokenizer.decode(
            rollout_output["completion_ids"],
            skip_special_tokens=True,
        )
        tool_call = build_browsergym_action_tool_call(completion_text)
        return ModelStepResult(
            response=LLMResponse(content=completion_text, tool_calls=[tool_call]),
            prompt_ids=list(rollout_output["prompt_ids"]),
            completion_ids=list(rollout_output["completion_ids"]),
            logprobs=list(rollout_output["logprobs"]),
        )

    return model_step
```

In practice, you should add a small parser around the completion text so common
outputs like `Action: click('13')` are normalized before calling
`build_browsergym_action_tool_call`.

## Create The Rollout Function

Pass the session factory, white-box harness adapter, and model-step builder to
`build_harness_rollout_func`:

```python
from openenv.core.harness import (
    HarnessRunLimits,
    MCPHarnessAdapter,
    build_harness_rollout_func,
)

rollout_func = build_harness_rollout_func(
    session_factory=session_factory,
    harness_adapter=MCPHarnessAdapter(),
    model_step_builder=lambda trainer, session: build_trl_browsergym_model_step(
        trainer,
        tokenizer,
    ),
    limits=HarnessRunLimits(max_turns=10),
)
```

The returned function accepts TRL prompts and a trainer, runs one harness-backed
BrowserGym episode per prompt, and returns `prompt_ids`, `completion_ids`,
`logprobs`, `env_reward`, and `verify_metrics`.

## Full Example

See [`examples/browsergym_harness.py`](https://github.com/meta-pytorch/OpenEnv/blob/main/examples/browsergym_harness.py)
for a complete TRL-oriented helper that includes action normalization and a
ready-to-use `build_browsergym_rollout_func`.
