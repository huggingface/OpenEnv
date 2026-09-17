# CLI

The `openenv` CLI provides a set of commands for building, validating, and pushing environments to Hugging Face Spaces or a custom Docker registry. For an end-to-end tutorial on building environments with OpenEnv, see the [building an environment](../getting_started/environment-builder.md) guide.

## `openenv init`

[[autodoc]] openenv.cli.commands.init.init

## `openenv import`

Import a supported third-party source environment into a generated OpenEnv
wrapper package. The command detects the source format from the directory
contents, so ORS/OpenReward and Prime Intellect Verifiers sources do not
require `--type` in the common case.

The generated wrapper vendors the source tree into the package and includes
vendored files as package data, so non-secret fixture/data files are available to
the environment server at runtime. The importer carries portable dependencies
from source `pyproject.toml` and `requirements.txt` files into the generated
environment, skips VCS/cache/build directories and common secret file patterns
such as `.env`, `secrets.yaml`, and private key files, and excludes compiled
binary artifacts; review the generated `vendor/` directory before publishing a
wrapper.

```bash
openenv import path/to/source --name my_env --output-dir ./envs
openenv import path/to/source --name my_env --output-dir ./envs --env-class MyEnv
```

```{eval-rst}
.. automodule:: openenv.cli.commands.import_env
   :members:
   :undoc-members:
   :show-inheritance:
```

## `openenv build`

[[autodoc]] openenv.cli.commands.build.build

## `openenv validate`

[[autodoc]] openenv.cli.commands.validate.validate

## `openenv push`

[[autodoc]] openenv.cli.commands.push.push

## `openenv serve`

Local serving is not implemented in the CLI yet. This command exits non-zero
and prints alternative ways to run an environment server.

[[autodoc]] openenv.cli.commands.serve.serve

## `openenv fork`

[[autodoc]] openenv.cli.commands.fork.fork

## `openenv skills`

Installs an `openenv-cli` skill into your AI assistant's skills directory so
it knows the `openenv` CLI is available and what each command does. Supports
Claude Code, Cursor, Codex, and OpenCode.

**Install for a single assistant (project-local):**

```bash
openenv skills add --claude    # → .claude/skills/openenv-cli/
openenv skills add --cursor    # → .cursor/skills/openenv-cli/
openenv skills add --codex     # → .codex/skills/openenv-cli/
openenv skills add --opencode  # → .opencode/skills/openenv-cli/
```

Multiple flags can be combined — `openenv skills add --claude --cursor` installs
for both at once. The skill file is written to a central location
(`.agents/skills/openenv-cli/`) and each agent directory gets a symlink, so
there is only one copy to update.

**Install globally (user-level, across all projects):**

```bash
openenv skills add --claude --global  # → ~/.claude/skills/openenv-cli/
```

**Overwrite an existing installation** (e.g. after upgrading `openenv`):

```bash
openenv skills add --claude --force
```

**Preview the skill content without installing:**

```bash
openenv skills preview
```

**Install to a custom path** (for non-standard agent setups):

```bash
openenv skills add --dest /path/to/my-agent/skills/
```

[[autodoc]] openenv.cli.commands.skills.skills_add

[[autodoc]] openenv.cli.commands.skills.skills_preview

## `openenv collect`

Collect rollouts from a running environment with a teacher model and write
them as an SFT-ready `results.jsonl`. The teacher can be a hosted provider
(`--provider openai|anthropic`) or any self-hosted OpenAI-compatible server
such as vLLM, TGI or Ollama via `--llm-endpoint`.

```bash
# Scripted teacher, no API key needed
openenv collect openspiel:tic_tac_toe --base-url http://localhost:8001 \
    --output-dir ./rollouts -n 10 --provider scripted

# Self-hosted model (vLLM serving on port 8000)
openenv collect reasoning_gym:chain_sum --base-url http://localhost:8001 \
    --output-dir ./rollouts -n 50 \
    --llm-endpoint http://localhost:8000 --model Qwen/Qwen3-1.7B
```

`--llm-endpoint` takes a full base URL. `/v1` is appended when the URL has no
path, so `http://localhost:8000` and `http://localhost:8000/v1` are equivalent;
a URL with a path (for example a gateway prefix) is used as-is. `--llm-port` is
only needed when the URL does not include a port.

[[autodoc]] openenv.cli.commands.collect.collect

# API Reference

## Entry point

[[autodoc]] openenv.cli.__main__.main

## CLI helpers

[[autodoc]] openenv.cli._cli_utils.validate_env_structure

## Validation utilities

[[autodoc]] openenv.cli._validation.validate_running_environment

[[autodoc]] openenv.cli._validation.validate_multi_mode_deployment

[[autodoc]] openenv.cli._validation.get_deployment_modes

[[autodoc]] openenv.cli._validation.format_validation_report

[[autodoc]] openenv.cli._validation.build_local_validation_json_report
