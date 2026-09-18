# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure builders for Pi sandbox bootstrap artifacts.

These functions produce the exact files and shell commands the sandbox needs to
run Pi against a configured LLM endpoint. No IO, no sandbox coupling — the
sandbox backend is responsible for writing files and running commands.
"""

from __future__ import annotations

import json

from .config import PiConfig

# Pi requires Node >= 22.19; bootstrap this version when the image ships none.
_NODE_VERSION = "22.19.0"
# npm package for the Pi coding-agent CLI (project is mid-rename; this name
# resolves to the same tool as ``@earendil-works/pi-coding-agent``).
_PI_NPM_PACKAGE = "@mariozechner/pi-coding-agent"


def pi_agent_dir(config: PiConfig) -> str:
    return f"{config.sandbox_home}/.pi/agent"


def models_json_path(config: PiConfig) -> str:
    return f"{pi_agent_dir(config)}/models.json"


def instruction_path(config: PiConfig) -> str:
    return f"{config.sandbox_home}/task/instruction.md"


def system_prompt_path(config: PiConfig) -> str:
    return f"{config.sandbox_home}/task/system.md"


def agent_log_path(config: PiConfig) -> str:
    return f"{config.sandbox_home}/logs/agent/pi.jsonl"


def verifier_reward_path(config: PiConfig) -> str:
    return f"{config.sandbox_home}/logs/verifier/reward.txt"


def workdir_path(config: PiConfig) -> str:
    return f"{config.sandbox_home}/workdir"


def node_dir(config: PiConfig) -> str:
    return f"{config.sandbox_home}/.node"


def npm_prefix(config: PiConfig) -> str:
    return f"{config.sandbox_home}/.pi-npm"


def pi_bin_path(config: PiConfig) -> str:
    return f"{npm_prefix(config)}/bin/pi"


def proxy_dir(config: PiConfig) -> str:
    return f"{config.sandbox_home}/proxy"


def proxy_source_path(config: PiConfig) -> str:
    return f"{proxy_dir(config)}/interception.py"


def proxy_trace_path(config: PiConfig) -> str:
    return f"{config.sandbox_home}/logs/agent/proxy_trace.jsonl"


def proxy_log_path(config: PiConfig) -> str:
    return f"{config.sandbox_home}/logs/agent/proxy.log"


def build_models_json(config: PiConfig) -> str:
    """Return the serialized ``models.json`` registering one OpenAI-compatible provider."""
    doc = {
        "providers": {
            config.provider_name: {
                "baseUrl": config.base_url,
                "api": "openai-completions",
                "apiKey": config.api_key,
                "models": [
                    {
                        "id": config.model,
                        "name": "Intercepted Model",
                        "reasoning": False,
                        "input": ["text"],
                        "contextWindow": config.context_window,
                        "maxTokens": config.max_tokens,
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        },
                    }
                ],
            }
        }
    }
    return json.dumps(doc, indent=2)


def build_install_cmd(config: PiConfig) -> str:
    """Return the shell command that installs Pi (bootstrapping Node if needed)."""
    home = config.sandbox_home
    package = _PI_NPM_PACKAGE
    if config.pi_version and config.pi_version != "latest":
        package = f"{package}@{config.pi_version}"
    return (
        "set -e && "
        f"mkdir -p {home}/.pi/agent {home}/logs/agent {home}/logs/verifier {home}/task {home}/workdir && "
        # Bootstrap Node 22 when the image ships none new enough (pi needs >=22.19).
        "if ! command -v node >/dev/null 2>&1 || "
        "! node -e \"const v=process.versions.node.split('.').map(Number); "
        'process.exit(v[0]>22||(v[0]===22&&v[1]>=19)?0:1)"; then '
        "A=$(uname -m); case $A in x86_64) A=x64;; aarch64|arm64) A=arm64;; esac && "
        f"curl -fsSL https://nodejs.org/dist/v{_NODE_VERSION}/node-v{_NODE_VERSION}-linux-$A.tar.xz "
        f"| tar -xJ -C {home} && ln -sfn {home}/node-v{_NODE_VERSION}-linux-$A {node_dir(config)}; fi && "
        f'export PATH="{node_dir(config)}/bin:$PATH" && '
        f"npm install -g --prefix {npm_prefix(config)} {package} && "
        f"{pi_bin_path(config)} --version"
    )


def build_run_cmd(config: PiConfig) -> str:
    """Return the shell command that launches Pi headless against a task."""
    tools_flag = f"--tools {','.join(config.tools)} " if config.tools else ""
    system_flag = (
        f'--system-prompt "$(cat {system_prompt_path(config)})" '
        if config.system_prompt
        else ""
    )
    # Select by --provider only (a "/" in --model is parsed as provider/id).
    return (
        "set -o pipefail && "
        f'export PATH="{node_dir(config)}/bin:$PATH" && '
        f"cd {workdir_path(config)} && "
        f"{pi_bin_path(config)} -p --no-session --mode json "
        "--no-context-files --no-extensions --no-skills --no-prompt-templates "
        f"--provider {config.provider_name} "
        f"{tools_flag}{system_flag}"
        f'"$(cat {instruction_path(config)})" '
        f"2>&1 | tee {agent_log_path(config)}"
    ).strip()


def build_env_vars(config: PiConfig) -> dict[str, str]:
    """Return env vars to set on the Pi process.

    The endpoint lives in ``models.json`` (written under ``PI_CODING_AGENT_DIR``),
    so no base-url env var is needed. ``PI_OFFLINE`` blocks startup network ops
    (telemetry / version / model-list refresh) without affecting model calls.
    """
    env = dict(config.extra_env)
    env["PI_CODING_AGENT_DIR"] = pi_agent_dir(config)
    env["PI_OFFLINE"] = "1"
    env["PI_SKIP_VERSION_CHECK"] = "1"
    return env
