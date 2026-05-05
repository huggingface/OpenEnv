---
title: OpenSpiel Environment Server
emoji: 🎮
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# OpenSpiel Environment

Integration of OpenSpiel games with the OpenEnv framework. [OpenSpiel](https://github.com/google-deepmind/open_spiel) is DeepMind's collection of 70+ game environments for RL research.

## Supported Games

This environment supports 6 games across different categories:

### Single-Player Games (No Opponent)
1. **Catch** - Move horizontally to catch a falling ball
2. **Cliff Walking** - Navigate grid without falling off cliff (Sutton & Barto benchmark)
3. **2048** - Classic tile-merging puzzle game
4. **Blackjack** - Simplified blackjack (HIT/STAND only)

### Multi-Player Games (with Bot Opponent)
5. **Tic-Tac-Toe** - Classic 3x3 game
6. **Kuhn Poker** - 2-player simplified poker (game theory benchmark)

## Quick Start

The simplest way to use the OpenSpiel environment is through the `OpenSpielEnv` class:

```python
from openspiel_env import OpenSpielEnv, OpenSpielAction

try:
    # Create environment from Docker image
    env = OpenSpielEnv.from_docker_image("openspiel-env:latest")

    # Reset to start a new episode
    result = env.reset()
    print(f"Initial state: {result.observation.info_state}")
    print(f"Legal actions: {result.observation.legal_actions}")

    # Play until done
    while not result.done:
        action_id = result.observation.legal_actions[0]
        result = env.step(OpenSpielAction(action_id=action_id))
        print(f"Reward: {result.reward}, Done: {result.done}")

finally:
    # Always clean up
    env.close()
```

That's it! The `OpenSpielEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

OpenSpiel requires compilation from C++ source. The Docker build uses a **pre-built base image** by default to avoid long build times.

### Default Build (Recommended)

From the **environment directory** (`envs/openspiel_env/`):

```bash
# Uses pre-built base image from GHCR (fast, ~1-2 min)
docker build -t openspiel-env:latest -f server/Dockerfile .
```

This uses the pre-built `ghcr.io/meta-pytorch/openenv-openspiel-base` image which already contains compiled OpenSpiel.

### Building Your Own Base Image (Optional)

If you need to customize OpenSpiel or can't access the pre-built image:

```bash
# Step 1: Build the base image (compiles OpenSpiel, ~30-60 min)
docker build -t openspiel-base:latest -f server/Dockerfile.openspiel-base .

# Step 2: Build the environment using your local base image
docker build -t openspiel-env:latest \
  --build-arg OPENSPIEL_BASE_IMAGE=openspiel-base:latest \
  -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (envs/openspiel_env/)
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
- `--env-var`, `-e`: Public Space variable as `KEY=VALUE` (repeatable). Overrides matching keys from `variables:` in openenv.yaml.
- `--secret`: Private Space secret as `KEY=VALUE` (repeatable). Value is never logged.

Space variables and secrets are only applied on direct Hugging Face Space
pushes. They are not supported with `--registry`, and they cannot be staged via
`--create-pr`.

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/openspiel-env

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/openspiel-env --private

# Select a game at push time (overrides variables: in openenv.yaml)
openenv push -e OPENSPIEL_GAME=tic_tac_toe

# Push with a private secret (not logged, stored encrypted in the Space)
openenv push --secret OPENAI_API_KEY=sk-...
```

### Declaring defaults in `openenv.yaml`

Public variables that your server reads (e.g. `OPENSPIEL_GAME`) can be declared
in `openenv.yaml` and will be set automatically on each `openenv push`:

```yaml
variables:
  OPENSPIEL_GAME: catch
```

CLI `-e` overrides any matching key from the yaml. Secrets are **never** put in
`openenv.yaml` — only on the CLI via `--secret KEY=VALUE`.

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring

> **Note**: The default Dockerfile uses a pre-built base image with OpenSpiel already compiled, so deployment is fast and works with standard CPU hardware. If you build your own base image, compilation requires more resources and time.

## Running Specific Games

```bash
# Catch (default)
docker run -p 8000:8000 openspiel-env:latest

# Tic-Tac-Toe with random opponent
docker run -p 8000:8000 -e OPENSPIEL_GAME=tic_tac_toe openspiel-env:latest

# Kuhn Poker
docker run -p 8000:8000 -e OPENSPIEL_GAME=kuhn_poker openspiel-env:latest

# 2048
docker run -p 8000:8000 -e OPENSPIEL_GAME=2048 openspiel-env:latest

# Blackjack
docker run -p 8000:8000 -e OPENSPIEL_GAME=blackjack openspiel-env:latest

# Cliff Walking
docker run -p 8000:8000 -e OPENSPIEL_GAME=cliff_walking openspiel-env:latest
```

## Environment Details

### Action
**OpenSpielAction**: Contains the action to take
- `action_id` (int) - Action ID to execute
- `game_name` (str) - Game name (default: "catch")
- `game_params` (Dict) - Optional game parameters

### Observation
**OpenSpielObservation**: Contains the game state
- `info_state` (List[float]) - Agent's information state vector
- `legal_actions` (List[int]) - Legal action IDs
- `game_phase` (str) - "initial", "playing", or "terminal"
- `current_player_id` (int) - Current player (-1 for simultaneous)
- `opponent_last_action` (Optional[int]) - Last opponent action
- `done` (bool) - Whether the episode has ended
- `reward` (Optional[float]) - Reward for the last action

### State
**OpenSpielState**: Server-side state snapshot
- `episode_id` (str) - Unique identifier for the current episode
- `step_count` (int) - Number of steps taken
- `game_name` (str) - Game name
- `agent_player` (int) - Agent's player ID
- `opponent_policy` (str) - Opponent policy name
- `num_players` (int) - Total players

## Configuration

### Environment Variables

- `OPENSPIEL_GAME`: Game name (default: "catch")
- `OPENSPIEL_AGENT_PLAYER`: Player ID for agent (default: 0)
- `OPENSPIEL_OPPONENT_POLICY`: Opponent policy for multi-player games
  - `random`: Uniform random (default)
  - `first`: Always picks first legal action
  - `last`: Always picks last legal action

### Example: Tic-Tac-Toe with Fixed Opponent

```bash
docker run -p 8000:8000 \
  -e OPENSPIEL_GAME=tic_tac_toe \
  -e OPENSPIEL_OPPONENT_POLICY=first \
  openspiel-env:latest
```

## Advanced Usage

### Connecting to an Existing Server

If you already have an OpenSpiel environment server running:

```python
from openspiel_env import OpenSpielEnv, OpenSpielAction

# Connect to existing server
env = OpenSpielEnv(base_url="http://localhost:8000")

# Use as normal
result = env.reset()
result = env.step(OpenSpielAction(action_id=result.observation.legal_actions[0]))

# Close connection (does NOT stop the server)
env.close()
```

### Connecting to HuggingFace Space

```python
from openspiel_env import OpenSpielEnv, OpenSpielAction

# Connect to remote Space
env = OpenSpielEnv(base_url="https://your-username-openspiel.hf.space")

result = env.reset()
print(f"Game: {result.observation.game_phase}")
print(f"Legal actions: {result.observation.legal_actions}")

result = env.step(OpenSpielAction(action_id=result.observation.legal_actions[0]))
env.close()
```

## Game-Specific Information

### 1. Catch
- **Type**: Single-player
- **Action Space**: 3 actions (left, stay, right)
- **Observation**: 5x5 grid flattened (25 dimensions)
- **Reward**: +1 for catching ball, 0 otherwise
- **Episode Length**: ~10 steps

### 2. Tic-Tac-Toe
- **Type**: 2-player turn-based, perfect information
- **Players**: Agent (X) vs Random Bot (O)
- **Action Space**: 9 positions
- **Observation**: 27 dimensions (3x3 board + game state)
- **Reward**: +1 win, -1 loss, 0 draw/mid-game

### 3. Kuhn Poker
- **Type**: 2-player turn-based, imperfect information
- **Players**: Agent vs Random Bot
- **Action Space**: 2 actions (pass/fold, bet/call)
- **Observation**: 6 dimensions (card + betting history)
- **Reward**: Pot winnings (typically -1, 0, +1, +2)
- **Notes**: THE benchmark for imperfect-information RL

### 4. Cliff Walking
- **Type**: Single-player grid world
- **Action Space**: 4 actions (up, down, left, right)
- **Observation**: Position encoding
- **Reward**: -1 per step, -100 for falling off cliff
- **Notes**: Classic RL benchmark from Sutton & Barto

### 5. 2048
- **Type**: Single-player puzzle
- **Action Space**: 4 actions (up, down, left, right)
- **Observation**: 4x4 grid with tile values
- **Reward**: Points from merging tiles
- **Notes**: Stochastic tile spawning

### 6. Blackjack
- **Type**: Single-player vs dealer
- **Action Space**: 2 actions (HIT, STAND)
- **Observation**: Player hand + dealer's visible card
- **Reward**: +1 win, -1 loss, 0 draw
- **Notes**: Simplified version, no double/split

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server (requires OpenSpiel installed locally):

```python
from openspiel_env.server.openspiel_environment import OpenSpielEnvironment
from openspiel_env.models import OpenSpielAction

# Create environment directly
env = OpenSpielEnvironment(game_name="catch")

# Test reset
obs = env.reset()
print(f"Info state: {obs.info_state}")

# Test step
obs = env.step(OpenSpielAction(action_id=0))
print(f"Done: {obs.done}, Reward: {obs.reward}")
```

### Running Locally

Run the server locally for development (requires OpenSpiel installed):

```bash
# From the environment directory
cd envs/openspiel_env

# Install dependencies
uv venv && source .venv/bin/activate
uv pip install -e .

# Start the server
python -m uvicorn server.app:app --reload
```

Or using the CLI entry point:

```bash
uv run --project . server --port 8000
```

### Automated Testing (All 6 Games)

```bash
./test_docker_all_games.sh
```

This script will build and test all 6 supported games in Docker.

## Project Structure

```
openspiel_env/
├── __init__.py                    # Module exports
├── README.md                      # This file
├── openenv.yaml                   # OpenEnv manifest
├── pyproject.toml                 # Project metadata and dependencies
├── client.py                      # OpenSpielEnv client implementation
├── models.py                      # Action, Observation, and State models
├── test_docker_all_games.sh       # Automated test script
└── server/
    ├── __init__.py                # Server module exports
    ├── openspiel_environment.py   # Core OpenSpielEnvironment implementation
    ├── opponent_policies.py       # Opponent policies (random, fixed)
    ├── app.py                     # FastAPI application
    ├── Dockerfile                 # Environment container (uses pre-built base)
    └── Dockerfile.openspiel-base  # Base image with compiled OpenSpiel
```

## Limitations

- **Simultaneous-move games**: Only agent_player=0 supported
- **Multi-agent training**: Single agent only (no self-play yet)
- **Opponent policies**: Random and fixed only (no MCTS yet)
- **Build time**: Building your own base image takes ~30-60 min (compiles OpenSpiel C++). Using the pre-built image is fast (~1-2 min) and works with standard hardware.

## References

- [OpenSpiel Paper (2019)](https://arxiv.org/abs/1908.09453)
- [OpenSpiel GitHub](https://github.com/google-deepmind/open_spiel)
- [OpenSpiel Documentation](https://openspiel.readthedocs.io/)
