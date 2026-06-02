"""
Introduction & Quick Start
==========================

**Part 1 of 5** in the OpenEnv Getting Started Series

This notebook introduces OpenEnv, explains why it exists, and gets you
running your first environment.

.. note::
    **Time**: ~10 minutes | **Difficulty**: Beginner | **GPU Required**: No

What You'll Learn
-----------------

- **What is OpenEnv**: The unified framework for RL environments
- **Why OpenEnv**: How it compares to traditional solutions like Gym
- **RL Basics**: The observe-act-reward loop in 60 seconds
- **Quick Start**: Connect to and interact with your first environment
"""

# %%
# Setup: Enable nested async event loops
# --------------------------------------
#
# This is needed when running in environments like Sphinx-Gallery or Jupyter
# that already have an event loop running.

import nest_asyncio
nest_asyncio.apply()

# %%
# What is OpenEnv?
# ----------------
#
# OpenEnv is a **unified framework for building, sharing, and interacting with
# reinforcement learning environments**. It is openly governed by a technical
# committee that includes Hugging Face, Unsloth, Reflection, and Meta PyTorch.
#
# **The Goal**: Make environment creation as easy and standardized as model
# sharing on Hugging Face.
#
# Key Features
# ~~~~~~~~~~~~
#
# - **Standardized API**: Gymnasium-style ``reset()``, ``step()``, ``state()``
# - **Type-Safe**: Full IDE autocomplete and error checking
# - **Containerized**: Environments run in Docker for isolation and reproducibility
# - **Shareable**: Push to Hugging Face Hub with one command
# - **Language-Agnostic**: HTTP/WebSocket API works from any language

# %%
# RL in 60 Seconds
# ----------------
#
# Reinforcement Learning is simpler than you think. It's just a loop:
#
# .. code-block:: text
#
#     ┌─────────────────────────────────────────────────────────────┐
#     │                 THE RL LOOP                                 │
#     │                                                             │
#     │    ┌─────────┐         ┌─────────────┐                      │
#     │    │  AGENT  │─action─▶│ ENVIRONMENT │                      │
#     │    │         │◀─reward─│             │                      │
#     │    │         │◀──obs───│             │                      │
#     │    └─────────┘         └─────────────┘                      │
#     │                                                             │
#     │    1. Agent observes the environment                        │
#     │    2. Agent chooses an action                               │
#     │    3. Environment returns reward + new observation          │
#     │    4. Repeat until done                                     │
#     └─────────────────────────────────────────────────────────────┘
#
# In code, it looks like this:
#
# .. code-block:: python
#
#     result = env.reset()                    # Start episode
#     while not result.done:
#         action = agent.choose(result.observation)
#         result = env.step(action)           # Take action, get reward
#         agent.learn(result.reward)
#
# That's it. That's RL!

# %%
# Why OpenEnv? (vs. Traditional Solutions)
# ----------------------------------------
#
# Traditional RL environments (like OpenAI Gym/Gymnasium) have been the backbone
# of RL research for years. They provide a simple API for interacting with
# environments, and the community has built thousands of environments on top of them.
#
# However, as RL moves from research to production, several challenges emerge:
#
# The Problem with Traditional Approaches
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 1. **No Type Safety**: Observations are numpy arrays like ``obs[0][3]``. What does
#    index 3 mean? You have to read documentation or source code to find out.
#
# 2. **Same-Process Execution**: The environment runs in your training process.
#    A bug in the environment can crash your entire training run.
#
# 3. **Dependency Hell**: Sharing environments means copying files and hoping
#    the recipient has the same dependencies installed.
#
# 4. **Python Lock-in**: Want to use Rust or C++ for your agent? Too bad—Gym is Python-only.
#
# 5. **"Works on My Machine"**: Environments behave differently on different systems
#    due to floating-point differences, library versions, or OS quirks.
#
# How OpenEnv Solves These Problems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# +------------------+----------------------------------+----------------------------------+
# | Challenge        | Traditional (Gym)                | OpenEnv                          |
# +==================+==================================+==================================+
# | **Type Safety**  | ``obs[0][3]`` - what is it?      | ``obs.info_state`` - IDE knows!  |
# +------------------+----------------------------------+----------------------------------+
# | **Isolation**    | Same process (can crash)         | Docker container (isolated)      |
# +------------------+----------------------------------+----------------------------------+
# | **Deployment**   | "Works on my machine"            | Same container everywhere        |
# +------------------+----------------------------------+----------------------------------+
# | **Sharing**      | Copy files, manage deps          | ``openenv push`` to Hub          |
# +------------------+----------------------------------+----------------------------------+
# | **Language**     | Python only                      | Any language (HTTP/WebSocket)    |
# +------------------+----------------------------------+----------------------------------+
# | **Scaling**      | Single machine                   | Deploy to Kubernetes             |
# +------------------+----------------------------------+----------------------------------+
# | **Debugging**    | Cryptic numpy index errors       | Clear, typed error messages      |
# +------------------+----------------------------------+----------------------------------+
#
# Side-by-Side Code Comparison
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's compare the same workflow in both approaches:
#
# **Traditional Gym approach:**
#
# .. code-block:: python
#
#     import gym
#     import numpy as np
#
#     # Create environment - runs in your process
#     env = gym.make("CartPole-v1")
#
#     # Reset returns numpy arrays
#     obs, info = env.reset()
#     # obs = array([0.01, 0.02, -0.03, 0.01])
#     # What do these numbers mean? You have to check docs!
#
#     # Step returns multiple values
#     obs, reward, done, truncated, info = env.step(action)
#     # No IDE autocomplete, easy to mix up return values
#
#     # If env crashes, your whole training crashes
#     # Sharing requires: pip install gym[atari], hope versions match
#
# **OpenEnv approach:**
#
# .. code-block:: python
#
#     from openenv import AutoEnv, AutoAction
#
#     # Load environment and action classes via auto-discovery
#     OpenSpielEnv = AutoEnv.get_env_class("openspiel")
#     OpenSpielAction = AutoAction.from_env("openspiel")
#
#     # Connect to containerized environment
#     with OpenSpielEnv(base_url="http://localhost:8000") as env:
#         # Reset returns typed StepResult
#         result = env.reset()
#         # result.observation.legal_actions - IDE autocompletes!
#         # result.observation.info_state - you know exactly what this is
#
#         # Step with typed action
#         action = OpenSpielAction(action_id=1, game_name="catch")
#         result = env.step(action)
#         # result.reward, result.done - all typed
#
#         # Environment runs in Docker - isolated from your code
#         # Share via: openenv push my-env (one command!)

# %%
# Part 1: Environment Setup
# -------------------------
#
# Let's set up our environment. This works in Google Colab, locally, or
# anywhere Python runs.

import subprocess
import sys
from pathlib import Path

# Detect environment
try:
    import google.colab

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("=" * 70)
    print("   GOOGLE COLAB DETECTED - Installing OpenEnv...")
    print("=" * 70)

    # Install OpenEnv
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "openenv"],
        capture_output=True,
    )
    print("   OpenEnv installed!")
    print("=" * 70)
else:
    print("=" * 70)
    print("   RUNNING LOCALLY")
    print("=" * 70)
    print()
    print("If you haven't installed OpenEnv yet:")
    print("   pip install openenv")
    print()

    # Add src to path for local development (when running from docs folder)
    src_path = Path.cwd().parent.parent.parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    # Add envs to path
    envs_path = Path.cwd().parent.parent.parent / "envs"
    if envs_path.exists():
        sys.path.insert(0, str(envs_path.parent))

    print("=" * 70)

print()
print("Ready to explore OpenEnv!")

# %%
# Part 2: Your First Environment - OpenSpiel
# -------------------------------------------
#
# What is OpenSpiel?
# ~~~~~~~~~~~~~~~~~~
#
# `OpenSpiel <https://github.com/google-deepmind/open_spiel>`_ is an open-source
# collection of **70+ game environments** developed by DeepMind for research in
# reinforcement learning, game theory, and multi-agent systems.
#
# It includes:
#
# - **Classic board games**: Chess, Go, Backgammon, Tic-Tac-Toe
# - **Card games**: Poker variants, Blackjack, Bridge
# - **Simple RL benchmarks**: Catch, Cliff Walking, 2048
# - **Multi-agent games**: Hanabi, Kuhn Poker, Negotiation games
#
# OpenSpiel is widely used in RL research because it provides consistent,
# well-tested implementations with support for both single-player and multi-player
# scenarios.
#
# How OpenSpiel Connects to OpenEnv
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# OpenEnv wraps OpenSpiel games as **containerized, type-safe environments**.
# This means:
#
# 1. You get all the benefits of OpenSpiel's game library
# 2. Plus type-safe Python clients with IDE autocomplete
# 3. Plus Docker isolation for reproducibility
# 4. Plus easy sharing via Hugging Face Hub
#
# Currently, OpenEnv includes wrappers for 6 OpenSpiel games:
#
# +------------------+-------------+------------------------------------------+
# | Game             | Players     | Description                              |
# +==================+=============+==========================================+
# | **Catch**        | 1           | Catch a falling ball with a paddle       |
# +------------------+-------------+------------------------------------------+
# | **2048**         | 1           | Slide tiles to combine numbers           |
# +------------------+-------------+------------------------------------------+
# | **Blackjack**    | 1           | Classic card game against dealer         |
# +------------------+-------------+------------------------------------------+
# | **Cliff Walking**| 1           | Navigate a grid while avoiding cliffs    |
# +------------------+-------------+------------------------------------------+
# | **Tic-Tac-Toe**  | 2           | Classic 3×3 grid game                    |
# +------------------+-------------+------------------------------------------+
# | **Kuhn Poker**   | 2           | Simplified 3-card poker                  |
# +------------------+-------------+------------------------------------------+
#
# The Catch Game
# ~~~~~~~~~~~~~~
#
# For this tutorial, we'll use **Catch**—one of the simplest RL environments.
# It's perfect for learning because:
#
# - Simple rules (easy to understand)
# - Fast episodes (10 steps each)
# - Clear success metric (did you catch the ball?)
# - Optimal strategy is learnable (move toward the ball)
#
# **Game Rules:**
#
# .. code-block:: text
#
#     ⬜ ⬜ 🔴 ⬜ ⬜    <- Ball starts at random column (row 0)
#     ⬜ ⬜ ⬜ ⬜ ⬜
#     ⬜ ⬜ ⬜ ⬜ ⬜       The ball falls down one row
#     ⬜ ⬜ ⬜ ⬜ ⬜       each time step
#     ⬜ ⬜ ⬜ ⬜ ⬜
#     ⬜ ⬜ ⬜ ⬜ ⬜
#     ⬜ ⬜ ⬜ ⬜ ⬜
#     ⬜ ⬜ ⬜ ⬜ ⬜
#     ⬜ ⬜ ⬜ ⬜ ⬜
#     ⬜ ⬜ 🏓 ⬜ ⬜    <- Paddle at bottom (row 9)
#
# - **Grid Size**: 10 rows × 5 columns
# - **Ball**: Starts at a random column in row 0, falls one row per step
# - **Paddle**: Starts at center column, you control it
# - **Episode Length**: 10 steps (ball reaches bottom)
#
# **Actions:**
#
# +------------+------------------+
# | Action ID  | Movement         |
# +============+==================+
# | 0          | Move LEFT        |
# +------------+------------------+
# | 1          | STAY (no move)   |
# +------------+------------------+
# | 2          | Move RIGHT       |
# +------------+------------------+
#
# **Rewards:**
#
# - **+1.0** if the paddle is in the same column as the ball when it lands
# - **0.0** if you miss the ball
#
# **Optimal Strategy**: Track the ball's column and move toward it. A perfect
# policy wins 100% of the time since the paddle can always reach any column
# in 10 steps (grid is only 5 columns wide).
#
# Importing OpenEnv
# ~~~~~~~~~~~~~~~~~
#
# First, let's import the OpenSpiel environment client and models:

# Real imports from OpenEnv
try:
    # Direct imports from the openspiel_env package
    from openspiel_env.client import OpenSpielEnv
    from openspiel_env.models import OpenSpielAction, OpenSpielObservation, OpenSpielState

    OPENENV_AVAILABLE = True
    print("✓ OpenEnv imports successful!")
    print(f"  - OpenSpielEnv: {OpenSpielEnv}")
    print(f"  - OpenSpielAction: {OpenSpielAction}")
except ImportError as e:
    OPENENV_AVAILABLE = False
    print(f"✗ OpenEnv not fully installed: {e}")
    print("  Run: pip install openenv")
    print("  And: pip install -e ./envs/openspiel_env")

# %%
# Connecting to an Environment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# OpenEnv provides three ways to connect to environments:
#
# 1. **From Hugging Face Hub** (auto-downloads and starts container)
# 2. **From Docker image** (uses local image)
# 3. **From URL** (connects to running server)
#
# Let's examine the actual methods available on the client class:

print("=" * 70)
print("   THREE WAYS TO CONNECT")
print("=" * 70)
print()

if OPENENV_AVAILABLE:
    # Show actual method signatures from the class
    import inspect

    print("Connection methods available on OpenSpielEnv:")
    print()

    # Method 1: from_hub
    if hasattr(OpenSpielEnv, "from_hub"):
        sig = inspect.signature(OpenSpielEnv.from_hub)
        print(f"1. OpenSpielEnv.from_hub{sig}")
        print("   → Auto-downloads from Hugging Face, starts container, connects")
        print("   Example: env = OpenSpielEnv.from_hub('openenv/openspiel-env')")
        print()

    # Method 2: from_docker_image
    if hasattr(OpenSpielEnv, "from_docker_image"):
        sig = inspect.signature(OpenSpielEnv.from_docker_image)
        print(f"2. OpenSpielEnv.from_docker_image{sig}")
        print("   → Starts container from local image, connects")
        print("   Example: env = OpenSpielEnv.from_docker_image('openspiel-env:latest')")
        print()

    # Method 3: Direct connection
    sig = inspect.signature(OpenSpielEnv.__init__)
    print(f"3. OpenSpielEnv.__init__{sig}")
    print("   → Connects to already-running server")
    print("   Example: env = OpenSpielEnv(base_url='http://localhost:8000')")
    print()

    print("-" * 70)
    print("All three give you the same API - just different ways to start!")
else:
    print("(OpenEnv not installed - showing expected methods)")
    print()
    print("1. OpenSpielEnv.from_hub(repo_id, *, use_docker=True, ...)")
    print("   → Auto-downloads from Hugging Face, starts container, connects")
    print()
    print("2. OpenSpielEnv.from_docker_image(image, provider=None, ...)")
    print("   → Starts container from local image, connects")
    print()
    print("3. OpenSpielEnv(base_url, connect_timeout_s=10.0, ...)")
    print("   → Connects to already-running server")

# %%
# Part 3: Playing the Catch Game
# ------------------------------
#
# Now let's actually play! This code attempts to connect to a real server.
# If no server is running, we'll show what the interaction looks like.

import random

# Check if we can connect to a server
SERVER_URL = "http://localhost:8000"
SERVER_AVAILABLE = False

if OPENENV_AVAILABLE:
    try:
        # Try to connect using sync wrapper
        env = OpenSpielEnv(base_url=SERVER_URL)
        with env.sync() as client:
            # Quick test to verify connection
            pass
        SERVER_AVAILABLE = True
        print(f"✓ Connected to server at {SERVER_URL}")
    except Exception as e:
        print(f"✗ No server running at {SERVER_URL}")
        print(f"  Error: {e}")
        print()
        print("To start a server, run one of these:")
        print("  docker run -p 8000:8000 openenv/openspiel-env:latest")
        print("  # OR")
        print("  cd envs/openspiel_env && openenv serve")

# %%
# Playing with a Real Server
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# When connected to a real server, here's how the interaction works:

if OPENENV_AVAILABLE and SERVER_AVAILABLE:
    print("=" * 70)
    print("   PLAYING CATCH - LIVE!")
    print("=" * 70)

    env = OpenSpielEnv(base_url=SERVER_URL)
    with env.sync() as client:
        # Reset to start a new episode
        result = client.reset()

        print(f"\nEpisode started!")
        print(f"  Observation type: {type(result.observation).__name__}")
        print(f"  Legal actions: {result.observation.legal_actions}")
        print(f"  Done: {result.done}")

        # Play until the episode ends
        step_count = 0
        while not result.done:
            # Choose a random action from legal actions
            action_id = random.choice(result.observation.legal_actions)
            action = OpenSpielAction(action_id=action_id, game_name="catch")

            # Take the action
            result = client.step(action)
            step_count += 1

            print(f"\nStep {step_count}:")
            print(f"  Action: {action_id} ({'LEFT' if action_id == 0 else 'STAY' if action_id == 1 else 'RIGHT'})")
            print(f"  Reward: {result.reward}")
            print(f"  Done: {result.done}")

        # Get final state
        state = client.state()
        print(f"\nEpisode complete!")
        print(f"  Total steps: {state.step_count}")
        print(f"  Final reward: {result.reward}")
        print(f"  Result: {'CAUGHT!' if result.reward > 0 else 'MISSED!'}")

else:
    # Run a local simulation to demonstrate the gameplay
    print("=" * 70)
    print("   PLAYING CATCH - LOCAL SIMULATION")
    print("=" * 70)
    print()
    print("No server running - demonstrating with local simulation.")
    print("(This shows exactly what happens when playing the real game)")
    print()

    # Simulate the Catch game locally
    GRID_HEIGHT = 10
    GRID_WIDTH = 5

    # Initialize game state
    ball_col = random.randint(0, GRID_WIDTH - 1)
    paddle_col = GRID_WIDTH // 2  # Start in center

    print(f"Game initialized:")
    print(f"  Ball starting column: {ball_col}")
    print(f"  Paddle starting column: {paddle_col}")
    print(f"  Grid size: {GRID_HEIGHT} rows × {GRID_WIDTH} columns")
    print()

    # Simulate episode
    for step in range(GRID_HEIGHT):
        # Create observation (matching OpenSpiel format)
        info_state = [0.0] * (GRID_HEIGHT * GRID_WIDTH)
        info_state[step * GRID_WIDTH + ball_col] = 1.0  # Ball position
        info_state[(GRID_HEIGHT - 1) * GRID_WIDTH + paddle_col] = 1.0  # Paddle

        legal_actions = [0, 1, 2]  # LEFT, STAY, RIGHT

        # Choose random action
        action_id = random.choice(legal_actions)
        action_name = {0: "LEFT", 1: "STAY", 2: "RIGHT"}[action_id]

        # Execute action
        old_paddle = paddle_col
        if action_id == 0:  # LEFT
            paddle_col = max(0, paddle_col - 1)
        elif action_id == 2:  # RIGHT
            paddle_col = min(GRID_WIDTH - 1, paddle_col + 1)

        print(f"Step {step + 1}: Ball at row {step}, col {ball_col} | "
              f"Paddle: {old_paddle}→{paddle_col} ({action_name})")

    # Determine result
    caught = (paddle_col == ball_col)
    reward = 1.0 if caught else 0.0

    print()
    print(f"Episode complete!")
    print(f"  Ball landed at column: {ball_col}")
    print(f"  Paddle final column: {paddle_col}")
    print(f"  Reward: {reward}")
    print(f"  Result: {'CAUGHT! 🎉' if caught else 'MISSED! 😢'}")
    print()
    print("-" * 70)
    print("This is exactly how the real OpenSpielEnv works,")
    print("just running locally instead of via WebSocket to a server.")

# %%
# Part 4: Understanding the Response Types
# ----------------------------------------
#
# OpenEnv uses type-safe models for all interactions. Let's create actual
# instances and examine their attributes:

print("=" * 70)
print("   OPENENV TYPE SYSTEM - ACTUAL INSTANCES")
print("=" * 70)

# Create example instances that match what you'd get from the Catch game
# These are the actual Pydantic models used by OpenEnv

# 1. OpenSpielObservation - what the agent receives after each step
print("\n📦 OpenSpielObservation (returned in StepResult)")
print("-" * 50)

if OPENENV_AVAILABLE:
    # OpenSpielObservation was already imported above via auto-discovery
    # Create a sample observation like what Catch game returns
    sample_observation = OpenSpielObservation(
        info_state=[0.0, 0.0, 1.0, 0.0, 0.0] + [0.0] * 45,  # Ball at col 2, row 0
        legal_actions=[0, 1, 2],  # LEFT, STAY, RIGHT
        game_phase="playing",
        current_player_id=0,
        opponent_last_action=None,
    )

    print(f"  info_state: {sample_observation.info_state[:10]}... (length: {len(sample_observation.info_state)})")
    print(f"  legal_actions: {sample_observation.legal_actions}")
    print(f"  game_phase: {sample_observation.game_phase!r}")
    print(f"  current_player_id: {sample_observation.current_player_id}")
    print(f"  opponent_last_action: {sample_observation.opponent_last_action}")
else:
    # Create without imports to show the structure
    from dataclasses import dataclass
    from typing import List, Optional

    @dataclass
    class OpenSpielObservation:
        info_state: List[float]
        legal_actions: List[int]
        game_phase: str = "playing"
        current_player_id: int = 0
        opponent_last_action: Optional[int] = None

    sample_observation = OpenSpielObservation(
        info_state=[0.0, 0.0, 1.0, 0.0, 0.0] + [0.0] * 45,
        legal_actions=[0, 1, 2],
        game_phase="playing",
        current_player_id=0,
        opponent_last_action=None,
    )

    print(f"  info_state: {sample_observation.info_state[:10]}... (length: {len(sample_observation.info_state)})")
    print(f"  legal_actions: {sample_observation.legal_actions}")
    print(f"  game_phase: {sample_observation.game_phase!r}")
    print(f"  current_player_id: {sample_observation.current_player_id}")
    print(f"  opponent_last_action: {sample_observation.opponent_last_action}")

# 2. OpenSpielState - the environment's internal state
print("\n📊 OpenSpielState (returned by state())")
print("-" * 50)

if OPENENV_AVAILABLE:
    # OpenSpielState was already imported above via auto-discovery
    sample_state = OpenSpielState(
        game_name="catch",
        agent_player=0,
        opponent_policy="random",
        game_params={"rows": 10, "columns": 5},
        num_players=1,
    )

    print(f"  game_name: {sample_state.game_name!r}")
    print(f"  agent_player: {sample_state.agent_player}")
    print(f"  opponent_policy: {sample_state.opponent_policy!r}")
    print(f"  game_params: {sample_state.game_params}")
    print(f"  num_players: {sample_state.num_players}")
else:
    @dataclass
    class OpenSpielState:
        game_name: str = "catch"
        agent_player: int = 0
        opponent_policy: str = "random"
        game_params: dict = None
        num_players: int = 1

    sample_state = OpenSpielState(
        game_name="catch",
        agent_player=0,
        opponent_policy="random",
        game_params={"rows": 10, "columns": 5},
        num_players=1,
    )

    print(f"  game_name: {sample_state.game_name!r}")
    print(f"  agent_player: {sample_state.agent_player}")
    print(f"  opponent_policy: {sample_state.opponent_policy!r}")
    print(f"  game_params: {sample_state.game_params}")
    print(f"  num_players: {sample_state.num_players}")

# 3. OpenSpielAction - what you send to step()
print("\n🎮 OpenSpielAction (what you send to step())")
print("-" * 50)

if OPENENV_AVAILABLE:
    # OpenSpielAction was already imported above via auto-discovery
    sample_action = OpenSpielAction(
        action_id=1,  # STAY
        game_name="catch",
        game_params={"rows": 10, "columns": 5},
    )

    print(f"  action_id: {sample_action.action_id}  # 0=LEFT, 1=STAY, 2=RIGHT")
    print(f"  game_name: {sample_action.game_name!r}")
    print(f"  game_params: {sample_action.game_params}")
else:
    @dataclass
    class OpenSpielAction:
        action_id: int
        game_name: str = "catch"
        game_params: dict = None

    sample_action = OpenSpielAction(
        action_id=1,
        game_name="catch",
        game_params={"rows": 10, "columns": 5},
    )

    print(f"  action_id: {sample_action.action_id}  # 0=LEFT, 1=STAY, 2=RIGHT")
    print(f"  game_name: {sample_action.game_name!r}")
    print(f"  game_params: {sample_action.game_params}")

print("\n" + "=" * 70)
print("These are the actual Pydantic/dataclass models used by OpenEnv.")
print("Type safety helps catch errors before they reach the environment!")
print("=" * 70)

# %%
# Part 5: The Architecture
# ------------------------
#
# OpenEnv uses a client-server architecture:
#
# .. code-block:: text
#
#     ┌─────────────────────────────────────────────────────────────┐
#     │  YOUR CODE                                                  │
#     │                                                             │
#     │  from openenv import AutoEnv                                │
#     │  OpenSpielEnv = AutoEnv.get_env_class("openspiel")          │
#     │  env = OpenSpielEnv(base_url="http://localhost:8000")       │
#     │  result = env.reset()      # Sends WebSocket message        │
#     │  result = env.step(action) # Sends WebSocket message        │
#     │                                                             │
#     └────────────────────────┬────────────────────────────────────┘
#                              │
#                              │ WebSocket (persistent connection)
#                              │
#     ┌────────────────────────▼────────────────────────────────────┐
#     │  DOCKER CONTAINER                                           │
#     │                                                             │
#     │  ┌─────────────────────────────────────────────────────┐    │
#     │  │  FastAPI Server + Environment Logic                 │    │
#     │  │  - /ws (WebSocket endpoint)                         │    │
#     │  │  - Handles reset(), step(), state()                 │    │
#     │  │  - Runs the actual game simulation                  │    │
#     │  └─────────────────────────────────────────────────────┘    │
#     │                                                             │
#     │  Isolated • Reproducible • Scalable                         │
#     └─────────────────────────────────────────────────────────────┘
#
# **Key insight**: You never deal with HTTP/WebSocket directly.
# The OpenEnv client handles all the networking!

# %%
# Summary
# -------
#
# In this notebook, you learned:
#
# **What OpenEnv Is:**
#
# - A unified framework for RL environments
# - Containerized, type-safe, and shareable
#
# **Why Use OpenEnv:**
#
# - Type safety with IDE autocomplete
# - Isolated Docker containers
# - Easy sharing via Hugging Face Hub
#
# **How to Use It:**
#
# - ``env.reset()`` - Start a new episode
# - ``env.step(action)`` - Take an action
# - ``env.state()`` - Get current state
#
# Next Steps
# ----------
#
# **Continue to Notebook 2: Using Environments**
#
# In the next notebook, you'll:
#
# - Explore all available OpenEnv environments
# - Create different AI policies
# - Run evaluations and compare performance
# - Work with multi-player games
