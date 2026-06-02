"""
Building Environments
=====================

**Part 3 of 5** in the OpenEnv Getting Started Series

This notebook covers how to create your own OpenEnv environment, package it
with Docker, and share it on Hugging Face Hub.

.. note::
    **Time**: ~20 minutes | **Difficulty**: Intermediate | **GPU Required**: No

What You'll Learn
-----------------

- **Environment Structure**: The standard OpenEnv project layout
- **Defining Models**: Type-safe Action and Observation classes
- **Implementing Logic**: The reset() and step() methods
- **Docker Packaging**: Containerizing your environment
- **Sharing**: Deploying to Hugging Face Hub
"""

# %%
# Part 1: Setup
# -------------
#
# Let's set up our environment and imports.

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

    # Add src and envs to path for local development
    src_path = Path.cwd().parent.parent.parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    envs_path = Path.cwd().parent.parent.parent / "envs"
    if envs_path.exists():
        sys.path.insert(0, str(envs_path.parent))

    print("=" * 70)

print()

# %%
# Part 2: When to Build Your Own Environment
# -------------------------------------------
#
# Build a custom environment when you need:
#
# - A game or simulation not in existing libraries
# - Domain-specific RL tasks (robotics, finance, etc.)
# - Custom reward functions or observation spaces
# - Proprietary environments for your organization
#
# Prerequisites
# ~~~~~~~~~~~~~
#
# Before building, ensure you have:
#
# - Python 3.11+
# - Docker Desktop or Docker Engine
# - OpenEnv installed: ``pip install openenv``

print("=" * 70)
print("   PREREQUISITES")
print("=" * 70)

# Check Python version
import platform

python_version = platform.python_version()
print(f"\n✓ Python version: {python_version}")

# Check if Docker is available
try:
    result = subprocess.run(
        ["docker", "--version"], capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        print(f"✓ Docker: {result.stdout.strip()}")
    else:
        print("✗ Docker: Not found (required for deployment)")
except Exception:
    print("✗ Docker: Not found (required for deployment)")

# Check OpenEnv CLI
try:
    result = subprocess.run(
        ["openenv", "--help"], capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        print("✓ OpenEnv CLI: Available")
    else:
        print("✗ OpenEnv CLI: Not found")
except Exception:
    print("✗ OpenEnv CLI: Not found (install with: pip install openenv)")

print()

# %%
# Part 3: Environment Structure
# -----------------------------
#
# Every OpenEnv environment follows a standardized structure:
#
# .. code-block:: text
#
#     my_game/
#     ├── __init__.py              # Package exports
#     ├── models.py                # Action & Observation definitions
#     ├── client.py                # Client for connecting to env
#     ├── openenv.yaml             # Environment manifest
#     ├── README.md                # Documentation
#     └── server/
#         ├── __init__.py
#         ├── my_game_environment.py   # Core environment logic
#         ├── app.py               # FastAPI server
#         ├── Dockerfile           # Container definition
#         └── requirements.txt     # Python dependencies
#
# The ``openenv init`` command scaffolds this structure for you:
#
# .. code-block:: bash
#
#     openenv init my_game
#

print("=" * 70)
print("   ENVIRONMENT STRUCTURE")
print("=" * 70)
print()

# Let's explore an actual environment from the repo
envs_base = Path.cwd().parent.parent.parent / "envs"

# Look for openspiel_env as a real example
openspiel_path = envs_base / "openspiel_env"

if openspiel_path.exists():
    print("Exploring REAL environment structure from envs/openspiel_env/:")
    print()

    def show_tree(path: Path, prefix: str = "", max_depth: int = 2, current_depth: int = 0):
        """Display directory tree."""
        if current_depth > max_depth:
            return

        # Get items, sorted (directories first)
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        except PermissionError:
            return

        # Filter out __pycache__ and hidden files
        items = [i for i in items if not i.name.startswith('.') and i.name != '__pycache__']

        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{item.name}{'/' if item.is_dir() else ''}")

            if item.is_dir() and current_depth < max_depth:
                extension = "    " if is_last else "│   "
                show_tree(item, prefix + extension, max_depth, current_depth + 1)

    show_tree(openspiel_path, "    ")
    print()

    # Show which key files exist
    print("Key files detected:")
    key_files = [
        ("__init__.py", "Package exports"),
        ("models.py", "Action & Observation definitions"),
        ("client.py", "Client for connecting to env"),
        ("openenv.yaml", "Environment manifest"),
        ("README.md", "Documentation"),
        ("server/app.py", "FastAPI server"),
        ("server/Dockerfile", "Container definition"),
    ]

    for filename, description in key_files:
        filepath = openspiel_path / filename
        exists = "✓" if filepath.exists() else "✗"
        print(f"  {exists} {filename:<25} - {description}")

else:
    print("Standard OpenEnv environment layout:")
    print(
        """
    my_game/
    ├── __init__.py              # Package exports
    ├── models.py                # Action & Observation definitions
    ├── client.py                # Client for connecting to env
    ├── openenv.yaml             # Environment manifest
    ├── README.md                # Documentation
    └── server/
        ├── __init__.py
        ├── my_game_environment.py   # Core environment logic
        ├── app.py               # FastAPI server
        ├── Dockerfile           # Container definition
        └── requirements.txt     # Python dependencies
"""
    )

print()
print("Create a new environment with: openenv init my_game")

# %%
# Part 4: Defining Your Models
# ----------------------------
#
# The first step is defining type-safe Action and Observation classes.
# These are dataclasses that inherit from OpenEnv base classes.

# Import the base classes from OpenEnv
try:
    from openenv.core.client_types import StepResult

    CORE_IMPORTS_OK = True
    print("✓ OpenEnv core imports successful")
except ImportError as e:
    CORE_IMPORTS_OK = False
    print(f"✗ Could not import OpenEnv core: {e}")

# %%
# Let's create models for a simple "Number Guessing" game:

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class GuessAction:
    """
    Action for the Number Guessing game.

    The player guesses a number between min_value and max_value.
    """

    guess: int  # The player's guess


@dataclass
class GuessObservation:
    """
    Observation returned after each guess.

    Contains feedback about the guess and game state.
    """

    hint: str  # "too_low", "too_high", or "correct"
    guesses_remaining: int  # How many guesses left
    min_value: int  # Lower bound
    max_value: int  # Upper bound
    done: bool = False  # Is the episode over?
    reward: float = 0.0  # Reward for this step


@dataclass
class GuessState:
    """
    Episode state metadata.
    """

    episode_id: str
    step_count: int
    target_number: int  # The secret number (only revealed when done)
    max_guesses: int


# Create example instances to show they work
action = GuessAction(guess=50)
observation = GuessObservation(
    hint="too_low", guesses_remaining=5, min_value=1, max_value=100
)
state = GuessState(
    episode_id="ep_001", step_count=1, target_number=73, max_guesses=7
)

print("\nExample instances:")
print(f"  Action: {action}")
print(f"  Observation: {observation}")
print(f"  State: {state}")

# %%
# Part 5: Implementing Environment Logic
# --------------------------------------
#
# The environment class implements the core game mechanics.
# You need to implement two required methods:
#
# - ``reset()`` - Initialize a new episode
# - ``step(action)`` - Execute an action and return the result

import random
import uuid
from abc import ABC, abstractmethod


class NumberGuessingEnvironment:
    """
    A simple number guessing game environment.

    The environment picks a random number, and the agent tries to guess it.
    Feedback is given after each guess ("too_low", "too_high", "correct").
    """

    def __init__(self, min_value: int = 1, max_value: int = 100, max_guesses: int = 7):
        """
        Initialize the environment.

        Args:
            min_value: Minimum possible target value
            max_value: Maximum possible target value
            max_guesses: Maximum guesses allowed per episode
        """
        self.min_value = min_value
        self.max_value = max_value
        self.max_guesses = max_guesses

        # Episode state (set in reset())
        self._target: Optional[int] = None
        self._guesses_remaining: int = 0
        self._step_count: int = 0
        self._episode_id: Optional[str] = None

    def reset(self, seed: Optional[int] = None) -> GuessObservation:
        """
        Start a new episode.

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            Initial observation for the new episode
        """
        if seed is not None:
            random.seed(seed)

        # Initialize episode state
        self._target = random.randint(self.min_value, self.max_value)
        self._guesses_remaining = self.max_guesses
        self._step_count = 0
        self._episode_id = str(uuid.uuid4())[:8]

        return GuessObservation(
            hint="game_started",
            guesses_remaining=self._guesses_remaining,
            min_value=self.min_value,
            max_value=self.max_value,
            done=False,
            reward=0.0,
        )

    def step(self, action: GuessAction) -> GuessObservation:
        """
        Process a guess and return the result.

        Args:
            action: The player's guess

        Returns:
            Observation with hint and game state
        """
        self._step_count += 1
        self._guesses_remaining -= 1

        guess = action.guess

        # Determine hint and reward
        if guess == self._target:
            hint = "correct"
            reward = 1.0  # Win!
            done = True
        elif guess < self._target:
            hint = "too_low"
            reward = 0.0
            done = self._guesses_remaining <= 0
        else:
            hint = "too_high"
            reward = 0.0
            done = self._guesses_remaining <= 0

        # Penalty for running out of guesses without winning
        if done and hint != "correct":
            reward = -0.5

        return GuessObservation(
            hint=hint,
            guesses_remaining=self._guesses_remaining,
            min_value=self.min_value,
            max_value=self.max_value,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> GuessState:
        """Get current episode state."""
        return GuessState(
            episode_id=self._episode_id or "",
            step_count=self._step_count,
            target_number=self._target or 0,
            max_guesses=self.max_guesses,
        )


print("=" * 70)
print("   ENVIRONMENT IMPLEMENTATION")
print("=" * 70)

# Show the actual class structure using inspect
import inspect

print("\nNumberGuessingEnvironment class defined above with these methods:")
print()

for name, method in inspect.getmembers(NumberGuessingEnvironment, predicate=inspect.isfunction):
    if not name.startswith('_'):
        sig = inspect.signature(method)
        print(f"  • {name}{sig}")
        if method.__doc__:
            first_line = method.__doc__.strip().split('\n')[0]
            print(f"      {first_line}")

# Also show properties
for name, prop in inspect.getmembers(NumberGuessingEnvironment, lambda x: isinstance(x, property)):
    print(f"  • {name} (property)")
    if prop.fget and prop.fget.__doc__:
        first_line = prop.fget.__doc__.strip().split('\n')[0]
        print(f"      {first_line}")

# %%
# Part 6: Testing Your Environment Locally
# ----------------------------------------
#
# Before containerizing, let's test the environment locally:

print("=" * 70)
print("   LOCAL TESTING")
print("=" * 70)

# Create environment instance
env = NumberGuessingEnvironment(min_value=1, max_value=100, max_guesses=7)

# Reset to start a new episode
obs = env.reset(seed=42)
print(f"\nNew episode started!")
print(f"  Hint: {obs.hint}")
print(f"  Guesses remaining: {obs.guesses_remaining}")
print(f"  Range: {obs.min_value} - {obs.max_value}")
print(f"  (Secret target: {env.state.target_number})")

# Play a simple binary search strategy
low, high = obs.min_value, obs.max_value
step = 0

print(f"\nPlaying with binary search strategy:")
print("-" * 50)

while not obs.done:
    # Binary search: guess the middle
    guess = (low + high) // 2
    action = GuessAction(guess=guess)
    obs = env.step(action)
    step += 1

    print(f"  Step {step}: Guessed {guess} -> {obs.hint}", end="")
    if obs.done:
        print(f" (Reward: {obs.reward})")
    else:
        print(f" (Remaining: {obs.guesses_remaining})")

    # Update bounds based on hint
    if obs.hint == "too_low":
        low = guess + 1
    elif obs.hint == "too_high":
        high = guess - 1

print(f"\nEpisode complete!")
print(f"  Total steps: {env.state.step_count}")
print(f"  Result: {'Won!' if obs.reward > 0 else 'Lost!'}")

# %%
# Next Steps
# ----------
#
# You've learned the core concepts for building OpenEnv environments:
#
# - **Environment Structure**: Standard project layout
# - **Models**: Type-safe Action, Observation, and State classes
# - **Logic**: Implementing ``reset()`` and ``step()`` methods
# - **Testing**: Running and validating locally
#
# Ready to Deploy?
# ~~~~~~~~~~~~~~~~
#
# The :doc:`Packaging & Deploying </auto_getting_started/environment-builder>` reference guide
# covers everything you need to package and share your environment:
#
# - **Server**: Wrapping your environment with FastAPI
# - **Client**: Creating typed client access
# - **Docker**: Containerizing for deployment
# - **Deployment**: Pushing to Hugging Face Hub or other registries
# - **Quick Reference**: CLI commands and the 8-step process at a glance
#
# Example: NumberGuessing Environment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# For a complete implementation of the environment we built above, check out
# ``envs/number_guessing/`` in the repository. It shows the full structure:
#
# .. code-block:: text
#
#     number_guessing/
#     ├── __init__.py
#     ├── models.py            # The Action/Observation classes
#     ├── client.py            # The client implementation
#     ├── openenv.yaml         # Environment manifest
#     └── server/
#         ├── app.py           # FastAPI server
#         ├── environment.py   # The logic we built above
#         └── Dockerfile       # Container definition
#
# Summary
# -------
#
# In this tutorial, you learned:
#
# 1. **When to build** - Custom games, domain-specific tasks, proprietary environments
# 2. **Environment structure** - The standard OpenEnv project layout
# 3. **Defining models** - Type-safe Action, Observation, and State dataclasses
# 4. **Implementing logic** - The ``reset()`` and ``step()`` methods
# 5. **Testing locally** - Running your environment before deployment
#
# Congratulations!
# ----------------
#
# You've completed the hands-on notebooks in the OpenEnv Getting Started Series!
#
# **You can now:**
#
# - ✅ Understand what OpenEnv is and why it exists
# - ✅ Connect to and use existing environments
# - ✅ Build your own custom environments
# - ✅ Test environments locally
#
# **Continue the series:**
#
# - :doc:`Packaging & Deploying </auto_getting_started/environment-builder>` (Part 4) - Package and deploy your environment with the CLI
# - :doc:`Contributing to Hugging Face </auto_getting_started/contributing-envs>` (Part 5) - Share environments on Hugging Face Hub
# - :doc:`RL Training Tutorial </tutorials/rl-training-2048>` - Train agents on 2048
# - Explore ``envs/`` directory for more examples
#
# Happy building!
