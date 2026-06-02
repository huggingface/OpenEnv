---
title: Maze Environment Server
emoji: 🎯
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Maze Environment

A gridworld maze where the agent must navigate from a start cell to an exit while avoiding walls.

## Maze Layout

The default environment ships with a single 8x8 maze:

```text
0 0 0 0 0 1 0 0
1 1 0 1 0 1 0 1
0 0 0 1 0 0 0 1
0 1 1 1 1 1 0 0
0 0 0 0 0 0 1 0
1 0 1 1 1 0 0 0
0 0 0 0 1 0 1 0
0 1 1 0 0 0 0 0
```

`0` is an empty cell and `1` is a wall. Coordinates are `(col, row)` with `(0, 0)` at the upper-left.

## Quick Start

The simplest way to use the Maze environment is through the `MazeEnv` client:

```python
from maze_env import MazeAction, MazeEnv

try:
    # Create environment from Docker image
    env = MazeEnv.from_docker_image("maze_env-env:latest")

    # Reset to start a new episode
    result = env.reset()
    print(f"Start position: {result.observation.current_position}")
    print(f"Legal actions: {result.observation.legal_actions}")

    # Play until done
    while not result.done:
        action_id = result.observation.legal_actions[0]
        result = env.step(MazeAction(action=action_id))
        print(f"Position: {result.observation.current_position}")
        print(f"Reward: {result.reward}, Done: {result.done}")

finally:
    # Always clean up
    env.close()
```

That's it! The `MazeEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

From the **environment directory** (`envs/maze_env/`):

```bash
docker build -t maze_env-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (envs/maze_env/)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/maze-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/maze-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

## Running the Default Maze

```bash
docker run -p 8000:8000 maze_env-env:latest
```

## Environment Details

### Action
**MazeAction**: Contains a single field
- `action` (int) - Movement action
  - 0 = move up
  - 1 = move down
  - 2 = move left
  - 3 = move right

### Observation
**MazeObservation**: Per-step observation payload
- `current_position` (List[int]) - Agent position as `[col, row]`
- `legal_actions` (List[int]) - Valid actions from the current position
- `metadata` (dict) - Extra info:
  - `maze`: 2D grid (0 = empty, 1 = wall)
  - `status`: "playing", "win", or "lose"
  - `exit_cell`: Exit position as `[col, row]`
  - `step`: Current step count

### State
**MazeState**: Server-side state snapshot
- `episode_id` (str) - Unique identifier for the current episode
- `step_count` (int) - Number of steps taken
- `done` (bool) - Whether the episode has ended
- `current_position` (List[int]) - Current agent position
- `exit_cell` (List[int]) - Exit position
- `status` (str) - "playing", "win", or "lose"

### Reward
The reward follows the underlying maze rules:
- Small penalty for a move (`-0.05`)
- Penalty for revisiting a cell (`-0.25`)
- Penalty for invalid move (`-0.75`)
- Reward for reaching the exit (`+10.0`)

## Configuration

### Environment Variables

This environment does not rely on environment variables. Customize the maze in code (see Development & Testing).

## Advanced Usage

### Connecting to an Existing Server

If you already have a Maze environment server running:

```python
from maze_env import MazeAction, MazeEnv

# Connect to existing server
env = MazeEnv(base_url="http://localhost:8000")

# Use as normal
result = env.reset()
result = env.step(MazeAction(action=result.observation.legal_actions[0]))

# Close connection (does NOT stop the server)
env.close()
```

### Connecting to HuggingFace Space

```python
from maze_env import MazeAction, MazeEnv

# Connect to remote Space
env = MazeEnv(base_url="https://your-username-maze-env.hf.space")

result = env.reset()
print(f"Position: {result.observation.current_position}")
print(f"Legal actions: {result.observation.legal_actions}")

result = env.step(MazeAction(action=result.observation.legal_actions[0]))
env.close()
```


## Project Structure

```
maze_env/
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── client.py              # MazeEnv client implementation
├── models.py              # Action, Observation, and State models
└── server/
    ├── __init__.py        # Server module exports
    ├── maze.py            # Maze logic and rewards
    ├── maze_env_environment.py  # Core environment implementation
    ├── app.py             # FastAPI application
    └── Dockerfile         # Environment container
```

## References

- [Reinforcement-Learning-Maze (original implementation)](https://github.com/erikdelange/Reinforcement-Learning-Maze)
