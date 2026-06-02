---
title: REPL Environment Server
emoji: 🎮
colorFrom: yellow
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# REPL Environment for OpenEnv

`repl_env` is an OpenEnv-native Python REPL environment for Recursive Language Model style execution. It now follows the current OpenEnv client/server conventions:

- `REPLEnv` is the remote async `EnvClient`
- `.sync()` is the sync wrapper for remote usage
- `LocalREPLEnv` is the explicit in-process helper
- `LocalRLMRunner` is the higher-level orchestration loop for local recursive RLM runs

The architecture is intentionally split the same way the official `rlm` and DSPy implementations split things:

- the environment executes code and exposes tools
- the runner owns the iterative prompting loop
- recursive behavior lives in backend/controller modules, not in the executor

## Overview

Inside the REPL, the model can:

- inspect `context`
- execute Python code across multiple turns with persistent state
- call `llm_query(...)` and `llm_query_batched(...)`
- call `rlm_query(...)` and `rlm_query_batched(...)` for recursive child runs when configured
- finish with `FINAL(...)`, `FINAL_VAR(...)`, or `answer = {"content": ..., "ready": True}`

## Current Architecture

Main modules:

- [`client.py`](client.py): remote async OpenEnv client
- [`local.py`](local.py): explicit in-process local env helper
- [`runner.py`](runner.py): local RLM orchestration loop
- [`recursive_backends.py`](recursive_backends.py): direct and recursive backend implementations
- [`recursive_controller.py`](recursive_controller.py): server-side backend/broker composition
- [`rubrics.py`](rubrics.py): reward rubrics (OpenEnv RFC 004)
- [`server/repl_environment.py`](server/repl_environment.py): server-side execution environment
- [`server/app.py`](server/app.py): OpenEnv HTTP server app and env factory

## What Works Today

- Standard remote OpenEnv usage through `REPLEnv`
- Local in-process execution through `LocalREPLEnv`
- Local recursive RLM runs through `LocalRLMRunner`
- Server-backed recursive calls through the current controller/broker path
- Explicit recursion controls:
  - `max_depth`
  - `max_children_total`
  - `max_children_per_batch`
  - `per_child_timeout_s`
  - `result_truncation_limit`
- Lightweight child trace metadata on local runner results
- Rubric-based rewards (OpenEnv RFC 004):
  - `ExactMatchRubric`: binary outcome reward against ground truth
  - `FuzzyMatchRubric`: partial credit for containment matches
  - `CustomMetricRubric`: user-provided `metric(expected, predicted) -> float`
  - `CodeExecutionRubric`: per-step process reward for code errors
  - `REPLRubric`: composite rubric combining outcome + process
  - Ground truth injectable at reset via `expected_answer`

## Rewards

Rewards follow the OpenEnv Rubric system (RFC 004). The environment uses
`REPLRubric` by default, which combines:

- **Outcome reward** (on terminal steps): compares `final_answer` against
  `expected_answer` if provided. Returns 1.0 for match, 0.0 otherwise.
- **Process reward** (on non-terminal steps): returns -0.05 for code
  execution errors, 0.0 for successful steps.
- **Failure reward**: returns -0.1 when max iterations exhausted without an answer.

For RL training (GRPO, etc.), pass `expected_answer` at reset time:

```python
with LocalREPLEnv() as env:
    env.reset(
        context="...",
        task_prompt="...",
        expected_answer="42",  # ground truth for rubric scoring
    )
    result = env.execute("print(FINAL(42))")
    print(result.reward)  # 1.0 (correct)
```

Custom rubrics can be injected at construction:

```python
from repl_env import LocalREPLEnv, CustomMetricRubric, REPLRubric

def my_metric(expected, predicted):
    return 1.0 if expected.strip() == predicted.strip() else 0.0

env = LocalREPLEnv(rubric=REPLRubric(outcome=CustomMetricRubric(my_metric)))
```

## Quick Start

### Remote Server Usage

Async:

```python
import asyncio
from repl_env import REPLEnv


async def main():
    async with REPLEnv(base_url="http://127.0.0.1:8000") as env:
        result = await env.reset(
            context="alpha beta gamma",
            task_prompt="Count the words",
        )
        result = await env.execute("count = len(context.split())")
        result = await env.execute("print(FINAL(count))")
        print(result.done)


asyncio.run(main())
```

Sync:

```python
from repl_env import REPLEnv

with REPLEnv(base_url="http://127.0.0.1:8000").sync() as env:
    result = env.reset(
        context="alpha beta gamma",
        task_prompt="Count the words",
    )
    result = env.execute("count = len(context.split())")
    result = env.execute("print(FINAL(count))")
    print(result.observation.result.stdout)
```

### Local Environment Usage

```python
from repl_env import LocalREPLEnv

with LocalREPLEnv() as env:
    result = env.reset(
        context="The quick brown fox jumps over the lazy dog",
        task_prompt="Count the words",
    )
    result = env.execute("count = len(context.split())")
    result = env.execute("print(FINAL(count))")
    print(env.state().final_answer)
```

### Local Recursive RLM Usage

`LocalRLMRunner` takes any `chat_fn(messages, model=None) -> str`. It works
with HF Inference API, vLLM, SGLang, Ollama, or any OpenAI-compatible server.

With HF Inference API:

```python
from huggingface_hub import InferenceClient
from repl_env import LocalRLMRunner, RLM_SYSTEM_PROMPT

client = InferenceClient(model="Qwen/Qwen3.5-9B", timeout=300)

def chat_fn(messages, model=None):
    response = client.chat.completions.create(
        model=model or "Qwen/Qwen3.5-9B",
        messages=messages,
        max_tokens=2048,
        temperature=0.6,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content

runner = LocalRLMRunner(chat_fn, max_iterations=30, max_depth=2)
result = runner.run("The answer is 42", "What number is mentioned?")
print(result.final_answer)
```

With a local vLLM server:

```python
from openai import OpenAI
from repl_env import LocalRLMRunner

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

def chat_fn(messages, model=None):
    response = client.chat.completions.create(
        model=model or "Qwen/Qwen3.5-9B",
        messages=messages,
        max_tokens=2048,
        temperature=0.6,
    )
    return response.choices[0].message.content

runner = LocalRLMRunner(chat_fn, max_iterations=30, max_depth=2)
result = runner.run(context, task)
```

### Using Different Models for Outer and Inner Loops

The outer loop (code generation) can use a large model while inner
`llm_query`/`rlm_query` calls use a smaller, faster model. Pass a
custom `backend_factory` to the runner:

```python
from openai import OpenAI
from huggingface_hub import InferenceClient
from repl_env import LocalRLMRunner
from repl_env.recursive_backends import BackendLimits, LocalChildRLMBackend

# Outer loop: large local model via vLLM
vllm = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

def outer_chat(messages, model=None):
    r = vllm.chat.completions.create(
        model="Qwen/Qwen3-32B", messages=messages, max_tokens=2048,
    )
    return r.choices[0].message.content

# Inner calls (llm_query/rlm_query): smaller HF-hosted model
hf = InferenceClient(model="Qwen/Qwen3.5-9B")

def inner_chat(messages, model=None):
    r = hf.chat.completions.create(
        model=model or "Qwen/Qwen3.5-9B", messages=messages, max_tokens=2048,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return r.choices[0].message.content

def my_backend_factory(llm_chat_fn, **kwargs):
    return LocalChildRLMBackend(
        inner_chat,  # inner calls use the smaller model
        runner_factory=LocalRLMRunner,
        system_prompt=kwargs["system_prompt"],
        max_iterations=kwargs["max_iterations"],
        env_max_iterations_multiplier=kwargs["env_max_iterations_multiplier"],
        depth=kwargs["depth"],
        limits=BackendLimits(max_depth=2),
    )

runner = LocalRLMRunner(
    outer_chat,                        # outer loop: large model
    backend_factory=my_backend_factory, # inner calls: small model
    max_iterations=30,
    max_depth=2,
)
result = runner.run(context, task)
```

## Server

Run the local server:

```bash
PYTHONPATH=src:envs uvicorn envs.repl_env.server.app:app --host 127.0.0.1 --port 8000
```

The server uses a proper OpenEnv environment factory in [`server/app.py`](server/app.py).

## API Surface

### Remote Client

```python
class REPLEnv(EnvClient[REPLAction, REPLObservation, REPLState]):
    async def reset(...)
    async def execute(code: str)
    async def submit_final_answer(answer: str)
    async def state()
```

Use `.sync()` for synchronous code.

### Local Helpers

```python
class LocalREPLEnv:
    def reset(...)
    def execute(code: str)
    def state()
```

```python
class LocalRLMRunner:
    def run(context: str, task_prompt: str, *, model: str | None = None) -> RLMRunResult
```

### Actions and Observations

`REPLAction`

```python
code: str = ""
is_final: bool = False
final_answer: str | None = None
```

`REPLObservation`

```python
result: CodeBlockResult
context_preview: str | None
context_length: int
available_variables: list[str]
iteration: int
max_iterations: int
done: bool
reward: float | None
metadata: dict
```

## Injected REPL Helpers

When configured, the REPL namespace exposes:

- `llm_query(prompt, model=None)`
- `llm_query_batched(prompts, model=None)`
- `rlm_query(prompt, model=None)`
- `rlm_query_batched(prompts, model=None)`
- `FINAL(value)`
- `FINAL_VAR(name)`
- `SHOW_VARS()`

Notes:

- `rlm_query` is the recursive child-run surface.
- At max recursion depth, recursion falls back to direct LM calls rather than spawning more children.
- Lifecycle callbacks follow the official `rlm` pattern:
  - `on_subcall_start(depth, model, prompt_preview)`
  - `on_subcall_complete(depth, model, duration, error_or_none)`

## Finalization Patterns

### `FINAL(...)`

```python
result = env.execute("answer = 42")
result = env.execute("print(FINAL(answer))")
```

### `FINAL_VAR(...)`

```python
result = env.execute("my_answer = '42'")
result = env.execute('print(FINAL_VAR("my_answer"))')
```

### `answer` dict

```python
result = env.execute("answer['content'] = '42'")
result = env.execute("answer['ready'] = True")
```

## Prompt Utilities

[`prompts.py`](prompts.py) contains the current message-building and parsing helpers used by the examples and runner.

Important exports:

- `RLM_SYSTEM_PROMPT`
- `RLM_SYSTEM_PROMPT_QWEN`
- `QueryMetadata`
- `build_rlm_system_prompt(...)`
- `build_user_prompt(...)`
- `extract_code_blocks(...)`
- `format_observations(...)`

These prompts were updated to reflect the actual helper surface the environment provides, rather than documenting tools that do not exist.

## Examples

- [`examples/repl_with_llm.py`](../../examples/repl_with_llm.py)
- [`examples/repl_oolong_simple.py`](../../examples/repl_oolong_simple.py)

Default hosted model in the examples is currently `Qwen/Qwen3.5-9B`, but real hosted inference still depends on provider availability and token access.

## Environment Variables

Server-side configuration in [`server/app.py`](server/app.py):

- `LLM_MODEL`
- `HF_TOKEN`
- `REPL_MAX_ITERATIONS`
- `REPL_MAX_OUTPUT_LENGTH`
- `REPL_CONTEXT_PREVIEW_LENGTH`
- `REPL_RLM_MAX_DEPTH`
- `REPL_RLM_MAX_ITERATIONS`

## References

- [RLM Paper (arXiv:2512.24601)](https://huggingface.co/papers/2512.24601)
- [RLM Implementation](https://github.com/alexzhang13/rlm)
- [Alex Zhang's RLM Blog](https://alexzhang13.github.io/blog/2025/rlm/)
- [Prime Intellect RLM Blog](https://www.primeintellect.ai/blog/rlm)
