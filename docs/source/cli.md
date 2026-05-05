# CLI

The `openenv` CLI provides a set of commands for building, validating, and pushing environments to Hugging Face Spaces or a custom Docker registry. For an end-to-end tutorial on building environments with OpenEnv, see the [building an environment](auto_getting_started/environment-builder.md) guide.

## `openenv init`

```{eval-rst}
.. automodule:: openenv.cli.commands.init
   :members:
   :undoc-members:
   :show-inheritance:
```

## `openenv build`

```{eval-rst}
.. automodule:: openenv.cli.commands.build
   :members:
   :undoc-members:
   :show-inheritance:
```

## `openenv validate`

```{eval-rst}
.. automodule:: openenv.cli.commands.validate
   :members:
   :undoc-members:
   :show-inheritance:
```

## `openenv push`

```{eval-rst}
.. automodule:: openenv.cli.commands.push
   :members:
   :undoc-members:
   :show-inheritance:
```

## `openenv serve`

```{eval-rst}
.. automodule:: openenv.cli.commands.serve
   :members:
   :undoc-members:
   :show-inheritance:
```

## `openenv fork`

```{eval-rst}
.. automodule:: openenv.cli.commands.fork
   :members:
   :undoc-members:
   :show-inheritance:
```

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

**Overwrite an existing installation** (e.g. after upgrading `openenv-core`):

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

```{eval-rst}
.. automodule:: openenv.cli.commands.skills
   :members:
   :undoc-members:
   :show-inheritance:
```

# API Reference

## Entry point

```{eval-rst}
.. automodule:: openenv.cli.__main__
   :members:
   :undoc-members:
   :show-inheritance:
```

## CLI helpers

```{eval-rst}
.. automodule:: openenv.cli._cli_utils
   :members:
   :undoc-members:
   :show-inheritance:
```

## Validation utilities

```{eval-rst}
.. automodule:: openenv.cli._validation
   :members:
   :undoc-members:
   :show-inheritance:
```
