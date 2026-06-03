# CLI

The `openenv` CLI provides a set of commands for building, validating, and pushing environments to Hugging Face Spaces or a custom Docker registry. For an end-to-end tutorial on building environments with OpenEnv, see the [building an environment](../getting_started/environment-builder.md) guide.

## `openenv init`

[[autodoc]] openenv.cli.commands.init.init

## `openenv build`

[[autodoc]] openenv.cli.commands.build.build

## `openenv validate`

[[autodoc]] openenv.cli.commands.validate.validate

## `openenv push`

[[autodoc]] openenv.cli.commands.push.push

## `openenv serve`

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
