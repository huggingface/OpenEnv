# Evaluate Claude Code in an Environment

This tutorial runs [Claude Code](https://docs.anthropic.com/en/docs/claude-code) *inside* an OpenEnv environment and evaluates it on [τ²-bench](../environments/tau2). It uses `HarnessEnvironment`, the agentic harness wrapper from [RFC 005](https://github.com/huggingface/OpenEnv/blob/main/rfcs/005-agentic-harnesses.md): the harness runs its own agent loop, the environment injects its tools into it over MCP, and each `step()` is one conversational turn.

The full code is in [`examples/claude_code_harness_eval`](https://github.com/huggingface/OpenEnv/tree/main/examples/claude_code_harness_eval).

> [!NOTE]
> RFC 005 does not capture token ids, so this is an evaluation path. To train a harness's policy, use [Harbor](../environments/harbor).

## What You'll Build

- An `AgenticHarnessAdapter` for Claude Code's headless mode.
- A `HarnessEnvironment` that gives Claude Code the tools and policy of a τ²-bench domain.
- An evaluation loop where τ²-bench's simulated customer talks to Claude Code until it is done, and τ²-bench scores the conversation.
- The same harness served in production mode, with you as the customer.

## Install Dependencies

You need the `claude` CLI, logged in or with `ANTHROPIC_API_KEY` set. Then, from a clone of OpenEnv:

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

Claude Code's own tools are turned off (`--tools ""`), so the agent can only act through the environment's tools. That makes the example safe to run on a laptop. To give it shell or file tools as well, list them in `--tools` and run it inside a sandbox.

## Join It to τ²-bench

[`tau2_harness.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/claude_code_harness_eval/tau2_harness.py) joins two environments:

- `tau2_env` holds the task: the simulated customer, the domain's database and its tools.
- `HarnessEnvironment` holds Claude Code. It gets the domain's tools and the domain's policy, appended to its system prompt. It does not get `respond_to_user` or `done`, because Claude Code talks to the customer through its replies.

```python
import tempfile

from claude_code_adapter import ClaudeCodeAdapter
from openenv.core.harness import HarnessAction, HarnessConfig, HarnessEnvironment
from tau2_env.server.tau2_environment import Tau2Environment, without_end_tokens
from tau2_harness import AGENT_PROMPT, domain_tools

tau2 = Tau2Environment(domain="airline", split="test")
observation = tau2.reset(task_id="2")

harness = HarnessEnvironment(
    adapter=ClaudeCodeAdapter(
        HarnessConfig(
            name="claude-code",
            command=["claude"],
            working_directory=tempfile.mkdtemp(),
            model="haiku",
        ),
        system_prompt=AGENT_PROMPT + observation.metadata["policy"],
    ),
    mcp=domain_tools(tau2),  # the domain's tools as a FastMCP server
)
harness.reset()

message = observation.metadata["user_message"]
while not tau2.state.done:
    turn = harness.step(HarnessAction(message=message))  # one Claude Code turn
    message = without_end_tokens(tau2.act(turn.metadata["response"]))  # the customer replies

print(tau2.state.reward)
```

Each customer message is one `step(HarnessAction(message=...))`, and Claude Code keeps the conversation's context across steps. Its tool calls run against the same database the customer sees, so τ²-bench can score the result.

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

`--domain` takes any τ²-bench domain, `--task-ids` picks tasks, and `--user-model` changes the simulated customer's model. Each task uses a fresh `tau2_env` and a fresh Claude Code process.

## Serve It in Production Mode

The same harness can be served over the `WS /harness` route ([production mode](../guides/simulation-vs-production)). Each connection gets its own Claude Code process and its own copy of the task's database, and `/reset` and `/step` are not exposed. Here you play the customer, so nothing is scored:

```bash
PYTHONPATH=src:envs:examples/claude_code_harness_eval \
    python examples/claude_code_harness_eval/serve.py --domain airline --task-id 2 --port 8000
python examples/claude_code_harness_eval/chat.py ws://localhost:8000/harness \
    "Hi, I'm Noah Muller, user id noah_muller_9847. What reservations do I have?" \
    "Which of them has a delayed flight?"
```

## Things to Know

- Claude Code adds today's date to its context, while each τ²-bench policy states its own current time (airline is 2024-05-15). The agent prompt tells it to use the policy's time. Without that line, it books flights in the wrong year.
- When the Anthropic API drops a request, Claude Code ends the turn with the error as its reply (`API Error: Connection dropped (ECONNRESET)`), and the customer sees it. Rerun the task if that happens.

## Adapting It

- **Your own environment:** pass your FastMCP tools as `mcp`, your instructions as the adapter's `system_prompt`, and a [rubric](rubrics) to `HarnessEnvironment` to score the episode.
- **Another model:** `--model` accepts any model Claude Code does. `ANTHROPIC_BASE_URL` in `HarnessConfig.env_vars` points it at an Anthropic-compatible endpoint.
- **Another harness:** write another `AgenticHarnessAdapter` in the same shape. Pick a headless multi-turn mode, pass `bridge_url` to its MCP config, and map its events.
