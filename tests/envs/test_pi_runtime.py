# SPDX-License-Identifier: BSD-3-Clause

"""Pure-builder tests for the Pi harness runtime (no sandbox required)."""

from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "envs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pi_env import pi_runtime as rt  # noqa: E402
from pi_env.config import PiConfig  # noqa: E402


def _cfg(**kw) -> PiConfig:
    base = dict(
        base_url="http://127.0.0.1:7000/v1", model="Qwen/Qwen3-4B-Instruct-2507"
    )
    base.update(kw)
    return PiConfig(**base)


def test_models_json_uses_full_model_id():
    cfg = _cfg(
        provider_name="intercepted", api_key="k", context_window=4096, max_tokens=512
    )
    doc = json.loads(rt.build_models_json(cfg))
    provider = doc["providers"]["intercepted"]
    assert provider["api"] == "openai-completions"
    assert provider["baseUrl"] == "http://127.0.0.1:7000/v1"
    assert provider["apiKey"] == "k"
    model = provider["models"][0]
    assert model["id"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert model["contextWindow"] == 4096
    assert model["maxTokens"] == 512
    # Pi requires a cost block on custom models.
    assert set(model["cost"]) == {"input", "output", "cacheRead", "cacheWrite"}


def test_install_cmd_bootstraps_node_and_installs_pi():
    cmd = rt.build_install_cmd(_cfg(sandbox_home="/root"))
    assert "command -v node" in cmd  # bootstrap guard
    assert "nodejs.org/dist/v22.19.0" in cmd
    assert "npm install -g --prefix /root/.pi-npm @mariozechner/pi-coding-agent" in cmd
    assert cmd.strip().endswith("/root/.pi-npm/bin/pi --version")


def test_hf_image_exposes_node_and_pi_on_path():
    dockerfile = os.path.join(_REPO_ROOT, "envs", "pi_env", "hf_image", "Dockerfile")
    with open(dockerfile, encoding="utf-8") as stream:
        contents = stream.read()

    path_env = 'ENV PATH="/root/.node/bin:/root/.pi-npm/bin:${PATH}"'
    assert path_env in contents
    assert contents.index(path_env) < contents.index("# Node 22")


def test_install_cmd_pins_version():
    cmd = rt.build_install_cmd(_cfg(pi_version="0.73.0"))
    assert "@mariozechner/pi-coding-agent@0.73.0" in cmd


def test_run_cmd_is_headless_json_and_points_at_provider():
    cfg = _cfg(sandbox_home="/root", tools=["read", "bash"])
    cmd = rt.build_run_cmd(cfg)
    for flag in ("-p", "--no-session", "--mode json", "--no-context-files"):
        assert flag in cmd
    assert "--provider intercepted" in cmd
    assert "--model" not in cmd
    assert "set -o pipefail" in cmd
    assert "--tools read,bash" in cmd
    assert '"$(cat /root/task/instruction.md)"' in cmd
    assert "tee /root/logs/agent/pi.jsonl" in cmd


def test_run_cmd_omits_tools_flag_when_unset():
    assert "--tools" not in rt.build_run_cmd(_cfg(tools=None))


def test_run_cmd_includes_system_prompt_when_set():
    cmd = rt.build_run_cmd(_cfg(system_prompt="be terse", sandbox_home="/root"))
    assert '--system-prompt "$(cat /root/task/system.md)"' in cmd


def test_env_vars_isolate_agent_dir_and_disable_startup_network():
    env = rt.build_env_vars(_cfg(sandbox_home="/root"))
    assert env["PI_CODING_AGENT_DIR"] == "/root/.pi/agent"
    assert env["PI_OFFLINE"] == "1"
    assert env["PI_SKIP_VERSION_CHECK"] == "1"


def test_paths_derive_from_sandbox_home():
    cfg = _cfg(sandbox_home="/home/user")
    assert rt.models_json_path(cfg) == "/home/user/.pi/agent/models.json"
    assert rt.instruction_path(cfg) == "/home/user/task/instruction.md"
    assert rt.agent_log_path(cfg) == "/home/user/logs/agent/pi.jsonl"
    assert rt.pi_bin_path(cfg) == "/home/user/.pi-npm/bin/pi"
    assert rt.proxy_trace_path(cfg) == "/home/user/logs/agent/proxy_trace.jsonl"
    assert rt.node_dir(cfg) == "/home/user/.node"
