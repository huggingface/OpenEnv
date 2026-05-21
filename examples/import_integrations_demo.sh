#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${OPENENV_IMPORT_DEMO_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/openenv-import-demo.XXXXXX")}"
GENERATED_DIR="$WORK_DIR/generated"
ORS_SOURCE="$WORK_DIR/ors_source"
VERIFIERS_SOURCE="$WORK_DIR/verifiers_source"
PIDS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
    fi
  done
  if [[ -z "${OPENENV_IMPORT_DEMO_KEEP:-}" ]]; then
    rm -rf "$WORK_DIR"
  else
    printf 'Keeping demo workspace: %s\n' "$WORK_DIR"
  fi
}
trap cleanup EXIT

real_uv="$(command -v uv)"
mkdir -p "$WORK_DIR/fakebin"
cat > "$WORK_DIR/fakebin/uv" <<EOF
#!/usr/bin/env bash
if [[ "\${1:-}" == "lock" ]]; then
  exit 0
fi
exec "$real_uv" "\$@"
EOF
chmod +x "$WORK_DIR/fakebin/uv"
export PATH="$WORK_DIR/fakebin:$PATH"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

choose_port() {
  uv run python - "$1" <<'PY'
import socket
import sys

preferred = int(sys.argv[1])

def available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True

if preferred > 0 and available(preferred):
    print(preferred)
else:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        print(sock.getsockname()[1])
PY
}

write_sources() {
  mkdir -p "$ORS_SOURCE/ors" "$VERIFIERS_SOURCE/verifiers" "$GENERATED_DIR"

  cat > "$ORS_SOURCE/ors/__init__.py" <<'PY'
from .environment import (
    Environment,
    ListToolsOutput,
    RunToolOutput,
    Split,
    TextBlock,
    ToolOutput,
    ToolSpec,
)
PY

  cat > "$ORS_SOURCE/ors/environment.py" <<'PY'
class Model:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return dict(self.__dict__)


class Split(Model):
    pass


class ToolSpec(Model):
    pass


class ListToolsOutput(Model):
    pass


class TextBlock(Model):
    def __init__(self, text, detail=None, type="text"):
        super().__init__(text=text, detail=detail, type=type)


class ToolOutput(Model):
    pass


class RunToolSuccess:
    ok = True

    def __init__(self, output):
        self.output = output


class RunToolOutput:
    def __init__(self, output):
        self.root = RunToolSuccess(output)


class Environment:
    def __init__(self, task_spec=None, secrets=None):
        self.task_spec = task_spec or {}
        self.secrets = secrets or {}

    def setup(self):
        return None

    def teardown(self):
        return None
PY

  cat > "$ORS_SOURCE/demo_env.py" <<'PY'
from ors import Environment, ListToolsOutput, RunToolOutput, Split, TextBlock, ToolOutput, ToolSpec


class DemoEnvironment(Environment):
    @classmethod
    def list_splits(cls):
        return [Split(name="train", type="train")]

    @classmethod
    def list_tasks(cls, split):
        return [{"id": "task-1", "question": "What is 2 + 2?", "answer": "4"}]

    @classmethod
    def num_tasks(cls, split):
        return len(cls.list_tasks(split))

    @classmethod
    def get_task(cls, split, index):
        return cls.list_tasks(split)[index]

    @classmethod
    def get_task_range(cls, split, start=None, stop=None):
        return cls.list_tasks(split)[slice(start, stop)]

    @classmethod
    def list_tools(cls):
        return ListToolsOutput(
            tools=[
                ToolSpec(
                    name="answer",
                    description="Submit an answer",
                    input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
                )
            ]
        )

    def get_prompt(self):
        return [TextBlock(text=self.task_spec["question"])]

    async def _call_tool(self, name, input):
        value = str(input.get("value", ""))
        correct = value == self.task_spec["answer"]
        return RunToolOutput(
            ToolOutput(
                blocks=[TextBlock(text="correct" if correct else "wrong")],
                metadata={"submitted": value},
                reward=1.0 if correct else 0.0,
                finished=True,
            )
        )
PY

  cat > "$VERIFIERS_SOURCE/verifiers/__init__.py" <<'PY'
class Environment:
    pass


class Rubric:
    def __init__(self, funcs=None):
        self.funcs = funcs or [self.exact_match]

    async def exact_match(self, completion, answer, **kwargs):
        text = completion[-1]["content"] if completion else ""
        return 1.0 if answer and answer in text else 0.0

    async def score_rollout(self, state):
        metrics = {}
        reward = 0.0
        for func in self.funcs:
            score = await func(
                completion=state.get("completion") or [],
                answer=state.get("answer") or state.get("task", {}).get("answer", ""),
                state=state,
            )
            metrics[getattr(func, "__name__", "reward")] = float(score)
            reward += float(score)
        state["reward"] = reward
        state["metrics"] = metrics


class SingleTurnEnv(Environment):
    def __init__(self, dataset, eval_dataset=None, rubric=None):
        self._dataset = dataset
        self._eval_dataset = eval_dataset or dataset
        self.rubric = rubric or Rubric()

    def get_dataset(self):
        return self._dataset

    def get_eval_dataset(self):
        return self._eval_dataset
PY

  cat > "$VERIFIERS_SOURCE/simple_math.py" <<'PY'
import verifiers as vf


def load_environment() -> vf.Environment:
    dataset = [
        {
            "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
            "answer": "4",
            "example_id": 0,
        }
    ]
    return vf.SingleTurnEnv(dataset=dataset, rubric=vf.Rubric())
PY
}

import_env() {
  local source_dir="$1"
  local name="$2"

  printf '\n==> openenv import %s --name %s\n' "$source_dir" "$name"
  uv run python -m openenv.cli.__main__ import "$source_dir" \
    --name "$name" \
    --output-dir "$GENERATED_DIR"
}

start_server() {
  local package="$1"
  local port="$2"
  local log_file="$WORK_DIR/$package.log"

  printf '==> starting %s on http://127.0.0.1:%s\n' "$package" "$port"
  PYTHONPATH="$REPO_ROOT/src:$GENERATED_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    uv run python -m "$package.server.app" --port "$port" >"$log_file" 2>&1 &
  local pid="$!"
  PIDS+=("$pid")

  for _ in $(seq 1 80); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      printf 'Server %s exited early. Log:\n' "$package" >&2
      cat "$log_file" >&2
      return 1
    fi
    sleep 0.25
  done

  printf 'Timed out waiting for %s. Log:\n' "$package" >&2
  cat "$log_file" >&2
  return 1
}

exercise_server() {
  local label="$1"
  local env_name="$2"
  local port="$3"
  local tool_name="$4"
  local arguments_json="$5"

  LABEL="$label" ENV_NAME="$env_name" PORT="$port" TOOL_NAME="$tool_name" ARGUMENTS_JSON="$arguments_json" \
    uv run python <<'PY'
import json
import os
import urllib.request

base = f"http://127.0.0.1:{os.environ['PORT']}"
env_name = os.environ["ENV_NAME"]
tool_name = os.environ["TOOL_NAME"]
arguments = json.loads(os.environ["ARGUMENTS_JSON"])


def request(method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {}
    if data is not None:
        headers["content-type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode())


def show(name, value):
    print(f"\n{os.environ['LABEL']} {name}")
    print(json.dumps(value, indent=2, sort_keys=True))


show("environments", request("GET", "/list_environments"))
show("splits", request("GET", f"/{env_name}/splits"))
show("task", request("POST", f"/{env_name}/task", {"split": "train", "index": 0}))
show("tools/list", request("POST", "/mcp", {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {},
}))
show("tools/call", request("POST", "/mcp", {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {"name": tool_name, "arguments": arguments},
}))
PY
}

main() {
  printf 'Demo workspace: %s\n' "$WORK_DIR"
  write_sources

  import_env "$ORS_SOURCE" "ors_openenv_demo"
  import_env "$VERIFIERS_SOURCE" "verifiers_openenv_demo"

  ORS_PORT="${ORS_PORT:-$(choose_port 8000)}"
  VERIFIERS_PORT="${VERIFIERS_PORT:-$(choose_port 8001)}"
  if [[ "$ORS_PORT" == "$VERIFIERS_PORT" ]]; then
    VERIFIERS_PORT="$(choose_port 0)"
  fi

  start_server "ors_openenv_demo" "$ORS_PORT"
  start_server "verifiers_openenv_demo" "$VERIFIERS_PORT"

  exercise_server "ORS/OpenReward" "ors_openenv_demo" "$ORS_PORT" "answer" '{"value":"4"}'
  exercise_server "Verifiers" "verifiers_openenv_demo" "$VERIFIERS_PORT" "submit" '{"completion":"The answer is 4."}'

  printf '\nDemo completed successfully.\n'
}

main "$@"
