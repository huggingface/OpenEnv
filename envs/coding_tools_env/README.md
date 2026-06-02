---
title: Coding Tools Environment Server
emoji: 🧰
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - coding
  - tools
  - e2b
short_description: SETA-style multi-tool coding environment backed by E2B
---

# Coding Tools Environment

`coding_tools_env` is an E2B-backed multi-tool coding environment with explicit
filesystem and shell tools. The tool surface is modeled after
[SETA](https://github.com/EnvCommons/SETA).

Each episode creates a fresh E2B sandbox. Reset accepts setup and verify
commands. Verify commands are used by `submit_solution`.

## Tools

- `bash(command, timeout=30)`
- `read(file_path, offset=None, limit=None)`
- `write(file_path, content)`
- `edit(file_path, old_string, new_string, replace_all=False)`
- `multi_edit(file_path, edits)`
- `glob(pattern, path=None)`
- `grep(pattern, path=None, include=None)`
- `ls(path=".", ignore=None)`
- `todo_write(todos)`
- `submit_solution()`

## Quick Start

```python
from coding_tools_env import CodingToolsEnv

with CodingToolsEnv(base_url="http://localhost:8000").sync() as env:
    env.reset(
        setup=["mkdir -p /home/user/work"],
        verify=["test -f /home/user/work/answer.txt"],
    )
    print(env.call_tool("write", file_path="/home/user/work/answer.txt", content="done\n"))
    print(env.call_tool("submit_solution"))
```

## Local Server

```bash
cd envs/coding_tools_env
E2B_API_KEY=e2b_... uv run --project . server
```

## Docker

```bash
cd envs/coding_tools_env
openenv build -t coding-tools-env
docker run -p 8000:8000 -e E2B_API_KEY=e2b_... coding-tools-env
```

## Configuration

- `E2B_API_KEY`: required when resetting an episode.
- `MAX_CONCURRENT_ENVS`: max concurrent sessions (default `4`).

