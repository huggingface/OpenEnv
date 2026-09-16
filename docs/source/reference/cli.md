# CLI

The `openenv` CLI provides a set of commands for building, validating, and pushing environments to Hugging Face Spaces or a custom Docker registry. For an end-to-end tutorial on building environments with OpenEnv, see the [building an environment](../getting_started/environment-builder) guide.

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

[[autodoc]] openenv.cli.commands.import_env.import_env

## `openenv build`

[[autodoc]] openenv.cli.commands.build.build

## `openenv validate`

Run static declaration checks without Docker, or build and probe a package locally:

```bash
openenv validate ./my_env --level static
openenv validate ./my_env --level runtime --local --output report.json
```

Runtime validation requires Docker and a `validation.execution` declaration in
`openenv.yaml`. The declaration points to a bounded JSON replay plan containing
one reset and a sequence of actions. The validator builds an immutable image,
opens one WebSocket session, measures reward values, observation schemas and
state continuity, and removes the container even if a check fails.

This first runtime slice implements startup, reward, observation and state
checks. Other applicable Level 2 checks appear explicitly as `SKIP`; they make
the result `WARN`, which exits zero and does not mean Level 2 is complete.
`FAIL` exits 1, unsupported package formats exit 2, and internal or policy errors
exit 3. `--level semantic`
includes the available lower-level checks but does not claim semantic execution.
`--skip-build` runs declaration checks and skips runtime execution entirely.
The Docker provider currently supports CPU workloads and `public` network mode;
unsupported network or GPU requirements skip runtime before building.

Runtime reports use schema version 2 and severity policy v2. Static reports for
v1 manifests retain schema version 1 and policy v1. An explicit v1 policy with a
runtime ceiling is rejected. With `--output report.json`, a sibling
`report.artifacts/` directory contains the replay plan, bounded redacted evidence,
coverage inventory, provider settings, cleanup outcome and checksums. Treat these
as author-validation evidence; this command does not issue certification.

For a pinned, installed-wheel reproduction of the shared fixture and its fault
cases, follow [the runtime lab](../../../tests/validation_runtime/README.md).
The implementation and remaining check inventory are tracked in
[the Level 2 umbrella](https://github.com/huggingface/OpenEnv/issues/1177).

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
