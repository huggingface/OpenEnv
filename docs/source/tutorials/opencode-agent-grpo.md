# Coding Agent Training with TRL (OpenCode)

This tutorial covers the black-box training path: training the actual
[`opencode`](https://opencode.ai) coding agent, with its own planner, tools,
context management, and stop condition, using TRL's experimental
`AsyncGRPOTrainer`. The agent owns its loop, and OpenEnv captures what it did.

> [!NOTE]
> Three GRPO patterns, three tutorials. For a standard `reset()` / `step()`
> flow where TRL drives the episode, see the
> [Wordle GRPO tutorial](wordle-grpo.md). For harness rollouts where the
> trainer still generates each turn (white-box), see the
> [BrowserGym harness tutorial](browsergym-harness.md). Use this page when you
> want to train a production agent as-is, without reimplementing its loop.

## How It Works

The full recipe lives in TRL. The moving pieces:

1. Each rollout runs the agent inside an OpenEnv session created by
   `OpenCodeSessionFactory` from
   [`opencode_env`](https://github.com/huggingface/OpenEnv/tree/main/envs/opencode_env),
   in `transparent_proxy` mode. A small proxy inside the sandbox forwards the
   agent's `/v1/chat/completions` calls to your vLLM server and records each
   turn's token ids and logprobs to a trace.
2. When the agent stops, TRL's `HarnessRolloutWorker` reads the trace, rebuilds
   the per-turn training rows from the recorded ids, and scores the final
   workspace with the session's `verify()` method (a held-out verifier the
   agent never sees).
3. `AsyncGRPOTrainer` trains on those rows, propagating the rollout reward to
   every trained token through the group-relative advantage. NCCL weight sync
   keeps the vLLM server on the current policy, so the agent always samples
   from the model being trained.

Each rollout gets its own isolated session: one sandbox, one proxy port, one
agent process. Three small functions adapt the recipe to your task:
`rollout_reward_fn` (outcome to scalar reward), `train_turn_fn` (which turns
receive gradient), and `agent_turn_fn` (which trace entries are real agent
turns rather than auxiliary calls like title generation). All three are
documented in
[TRL's harness training guide](https://huggingface.co/docs/trl/openenv#training-on-harnesses-training-a-real-coding-agent-opencode).

## Full Recipe

The reference script trains on competitive-coding problems from
`agentica-org/DeepCoder-Preview-Dataset`: the agent writes `solution.py`, and
the verifier runs it against held-out tests, returning the fraction passed. It
is self-contained, runs the agent in a local subprocess sandbox (no container
setup needed), needs two GPUs (one serving the policy with vLLM, one
training), and has been validated end to end on Qwen3. To scale
rollouts beyond a single node, a sibling script runs each rollout in its own
remote Hugging Face sandbox instead of a local subprocess.
Installation, the exact vLLM serving flags, and the run commands live next to
the recipe in TRL:

- [Training on harnesses](https://huggingface.co/docs/trl/openenv#training-on-harnesses-training-a-real-coding-agent-opencode)
  in TRL's OpenEnv docs: rollout semantics, the reward path, turn selection,
  and the trace contract.
- [`examples/async_grpo_opencode/async_grpo_opencode.py`](https://github.com/huggingface/trl/blob/main/examples/async_grpo_opencode/async_grpo_opencode.py)
  in TRL: the complete, runnable script.
- [`examples/async_grpo_opencode/opencode_hf_sandbox.py`](https://github.com/huggingface/trl/blob/main/examples/async_grpo_opencode/opencode_hf_sandbox.py)
  in TRL: the same recipe, but each rollout runs in its own remote Hugging Face
  sandbox, so rollouts scale out beyond one node.
- [`envs/opencode_env`](https://github.com/huggingface/OpenEnv/tree/main/envs/opencode_env):
  the OpenEnv side, including the session factory, sandbox backends, and the
  transparent interception proxy.
