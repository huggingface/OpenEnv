# Evaluate Claude Code in an Environment

This tutorial runs [Claude Code](https://docs.anthropic.com/en/docs/claude-code) *inside* an OpenEnv environment and evaluates it on [τ²-bench](../environments/tau2). It uses `HarnessEnvironment`, the agentic harness wrapper from [RFC 005](https://github.com/huggingface/OpenEnv/blob/main/rfcs/005-agentic-harnesses.md): the harness runs its own agent loop, the environment injects its tools into it over MCP, and each `step()` is one conversational turn.

The tools keep running in the environment's process: `HarnessEnvironment` serves them on a local MCP bridge, and Claude Code calls them over HTTP. This is the environment-side harness API, which is different from the trainer-side session runtime in the [BrowserGym harness tutorial](browsergym-harness).

The full code is in [`examples/claude_code_harness_eval`](https://github.com/huggingface/OpenEnv/tree/main/examples/claude_code_harness_eval).

> [!NOTE]
> RFC 005 does not capture token ids, so this is an evaluation path. To train a harness's policy, use [Harbor](../environments/harbor).

## What You'll Build

- An `AgenticHarnessAdapter` for Claude Code's headless mode.
- A `HarnessEnvironment` that gives Claude Code the tools and policy of a τ²-bench task, plays the task's simulated customer between turns, and scores the conversation with a rubric.
- The RFC 005 loop around it: one `step()` per customer message, until the customer is done.
- The same harness served in production mode, with you as the customer.

## Install Dependencies

You need the `claude` CLI, logged in or with `ANTHROPIC_API_KEY` set, and Python 3.12 or later, which τ²-bench requires. Then, from a clone of OpenEnv:

```bash
pip install -e . -e envs/tau2_env
```

τ²-bench's tasks and databases are not in its package. Fetch them once and point `TAU2_DATA_DIR` at them, as described in [the τ²-bench environment's docs](../environments/tau2#running-the-server). The simulated customer runs on [Inference Providers](https://huggingface.co/docs/inference-providers), so set `HF_TOKEN` too.

## The Adapter

An `AgenticHarnessAdapter` tells `HarnessEnvironment` how to start a harness, how to hand it tools, and how to read a turn. [`claude_code_adapter.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/claude_code_harness_eval/claude_code_adapter.py) keeps one long-lived `claude -p --input-format stream-json --output-format stream-json` process per episode:

| `AgenticHarnessAdapter` | Claude Code |
|---|---|
| `inject_tools(tools, bridge_url)` | `--mcp-config` with the bridge as an HTTP MCP server, and `--allowedTools mcp__env` so its tools run without a permission prompt |
| `start()` | spawns the process with `HarnessProcess`. Claude Code prints nothing until the first message |
| `send_message_streaming()` | writes `{"type": "user", "message": {...}}` on stdin and maps stdout events: `tool_use` → `TOOL_CALL`, `tool_result` → `TOOL_RESULT`, `text` → `TEXT_OUTPUT`, and `result` (end of turn) → `TURN_COMPLETE` |

Claude Code's own tools are turned off (`--tools ""`), so the agent can only act through the environment's tools: it has no shell, no file access and no way to open connections of its own. It also loads no settings files (`--setting-sources ""`), so your hooks and `CLAUDE.md` stay out of the run. To give it shell or file tools as well, list them in `--tools` and run it in a sandbox: the [container](#serve-it-in-a-container) keeps it away from your machine, and an egress allowlist (for example `ACASandboxProvider.deny_all_egress()`) keeps it to the model's API.

## Join It to τ²-bench

[`tau2_harness.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/claude_code_harness_eval/tau2_harness.py) defines `Tau2Harness`, a `HarnessEnvironment` for one τ²-bench task:

- **The tools and the policy.** Claude Code gets the domain's tools over MCP, and the domain's policy appended to its system prompt. It does not get `respond_to_user` or `done`, because it talks to the customer through its replies.
- **The customer.** After each Claude Code turn, the environment passes the reply to τ²-bench's simulated customer and returns the customer's next message in `observation.metadata["customer"]`. The customer is part of the environment, as in `tau2_env` itself.
- **The reward.** A rubric reads τ²-bench's score when the customer ends the conversation, so the reward is in `observation.reward` and `observation.done` marks the end.

```python
import tempfile

from openenv.core.harness import HarnessAction, HarnessConfig
from tau2_env.server.tau2_environment import Tau2Environment
from tau2_harness import Tau2Harness

config = HarnessConfig(
    name="claude-code",
    command=["claude"],
    working_directory=tempfile.mkdtemp(),
    model="haiku",
)
harness = Tau2Harness(Tau2Environment(domain="airline", split="test"), "2", config)

observation = harness.reset()  # the customer's opening message
while not observation.done:
    # One Claude Code turn, then the customer's answer
    observation = harness.step(HarnessAction(message=observation.metadata["customer"]))

print(observation.reward)
harness.close()  # stops Claude Code and the τ²-bench task
```

Each customer message is one `step(HarnessAction(message=...))`, and Claude Code keeps the conversation's context across steps. Its tool calls run against the same database the customer sees, so τ²-bench can score the result. If Claude Code exits or runs out of time, the step comes back with `done` and the error in its metadata.

## Run the Evaluation

```bash
PYTHONPATH=src:envs:examples/claude_code_harness_eval \
    python examples/claude_code_harness_eval/run_eval.py --domain airline --tasks 3 --model haiku
```

```
=== airline task 2
customer: Hi, I'd like to book a flight from San Francisco to New York for three passengers.
[...]
customer: [...] Can you look it up under my account? The delay was really frustrating.
  -> get_user_details({'user_id': 'noah_muller_9847'})
  -> get_reservation_details({'reservation_id': '4OG6T3'})
  -> get_flight_status({'flight_number': 'HAT018', 'date': '2024-05-11'})
  [...]
reward: 1.00 {'DB': 1.0, 'COMMUNICATE': 1.0}
...
pass^1: 3/3
```

The first airline test tasks are a gentle start. Tasks 2 and 6 only look things up, so they pass as long as the agent leaves the database as it found it. Task 8 books a reservation, and τ²-bench checks the database it leaves behind.

`--domain` takes any τ²-bench domain, `--task-ids` picks tasks, and `--user-model` changes the simulated customer's model. Each task uses a fresh `tau2_env` and a fresh Claude Code process.

## Serve It in Production Mode

The same harness can be served over the `WS /harness` route ([production mode](../guides/simulation-vs-production)). The server takes one connection at a time, each with a fresh Claude Code process and a fresh copy of the task's database. `/harness` sends your messages straight to Claude Code, without the environment's turn logic, so you play the customer and nothing is scored. The server has no authentication, and every connection spends your credentials, so keep it on localhost or behind your own auth:

```bash
PYTHONPATH=src:envs:examples/claude_code_harness_eval \
    python examples/claude_code_harness_eval/serve.py --domain airline --task-id 2 --port 8000
python examples/claude_code_harness_eval/chat.py ws://localhost:8000/harness \
    "Hi, I'm Noah Muller, user id noah_muller_9847. What reservations do I have?" \
    "Which of them has a delayed flight?"
```

### Serve It in a Container

RFC 005 runs the harness inside the environment's container, apart from the machine that serves it. [`examples/claude_code_harness_eval/Dockerfile`](https://github.com/huggingface/OpenEnv/blob/main/examples/claude_code_harness_eval/Dockerfile) does that for this recipe: Claude Code, `tau2_env` and `serve.py` run together in one image, as a non-root user. Build it from the repository root and pass the credentials at run time:

```bash
docker build -t claude-code-tau2 -f examples/claude_code_harness_eval/Dockerfile .
docker run -p 127.0.0.1:8000:8000 -e HF_TOKEN -e ANTHROPIC_API_KEY claude-code-tau2
python examples/claude_code_harness_eval/chat.py ws://localhost:8000/harness \
    "Hi, I'm Noah Muller, user id noah_muller_9847. What reservations do I have?"
```

Use `-e CLAUDE_CODE_OAUTH_TOKEN` instead of `ANTHROPIC_API_KEY` to run on a Claude subscription (`claude setup-token` creates the token). Add `serve.py` arguments after the image name to pick the domain, task or model.

## Things to Know

- Claude Code adds today's date to its context, while each τ²-bench policy states its own current time (airline is 2024-05-15). The agent prompt tells it to use the policy's time. Without that line, it books flights in the wrong year.
- When the Anthropic API fails a request (for example `API Error: Connection dropped (ECONNRESET)`), the adapter ends the turn as a harness failure, so the step comes back with `done` and `error_type` `harness_crashed`, and `run_eval.py` reports the task as `ERROR` and leaves it out of pass^1. Rerun it. The same goes for the simulated customer: if its model fails, the step ends with `error_type` `customer_failed`, and the task is an `ERROR` rather than a failure of Claude Code.

## Adapting It

- **Your own environment:** pass your FastMCP tools as `mcp`, your instructions as the adapter's `system_prompt`, and a [rubric](rubrics) to `HarnessEnvironment` to score the episode.
- **Another model:** `--model` accepts any model Claude Code does. `ANTHROPIC_BASE_URL` in `HarnessConfig.env_vars` points it at an Anthropic-compatible endpoint.
- **Another harness:** write another `AgenticHarnessAdapter` in the same shape. Pick a headless multi-turn mode, pass `bridge_url` to its MCP config, and map its events.
