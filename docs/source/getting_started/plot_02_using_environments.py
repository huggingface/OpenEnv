"""
Using Environments
==================

**Part 2 of 5** in the OpenEnv Getting Started Series

This notebook covers how to use OpenEnv environments: connecting to them,
creating AI policies, running evaluations, and working with different games.

.. note::
    **Time**: ~15 minutes | **Difficulty**: Beginner-Intermediate | **GPU Required**: No

What You'll Learn
-----------------

- **Connection Methods**: Hub, Docker, and direct URL connections
- **Available Environments**: OpenSpiel games, coding, browsing, and more
- **Creating Policies**: Random, heuristic, and learning-based strategies
- **Running Evaluations**: Measuring and comparing policy performance
"""

# %%
# Part 1: Setup
# -------------
#
# Let's set up our environment and imports.

import random
import subprocess
import sys
from pathlib import Path

import nest_asyncio
nest_asyncio.apply()

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
# Part 2: Available Environments
# ------------------------------
#
# OpenEnv includes a growing collection of environments for different RL tasks.
#
# OpenSpiel Games
# ~~~~~~~~~~~~~~~
#
# OpenSpiel (from DeepMind) provides 70+ game environments. OpenEnv wraps
# several of these:
#
# +------------------+-------------+------------------------------------------+
# | Game             | Players     | Description                              |
# +==================+=============+==========================================+
# | **Catch**        | 1           | Catch falling ball with paddle           |
# +------------------+-------------+------------------------------------------+
# | **2048**         | 1           | Slide tiles to combine numbers           |
# +------------------+-------------+------------------------------------------+
# | **Blackjack**    | 1           | Classic card game vs dealer              |
# +------------------+-------------+------------------------------------------+
# | **Cliff Walking**| 1           | Navigate grid, avoid cliffs              |
# +------------------+-------------+------------------------------------------+
# | **Tic-Tac-Toe**  | 2           | Classic 3x3 grid game                    |
# +------------------+-------------+------------------------------------------+
# | **Kuhn Poker**   | 2           | Simplified poker with 3 cards            |
# +------------------+-------------+------------------------------------------+
#
# Other Environment Types
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# +------------------+--------------------------------------------------+
# | Environment      | Description                                      |
# +==================+==================================================+
# | **Coding Env**   | Execute and evaluate code solutions              |
# +------------------+--------------------------------------------------+
# | **BrowserGym**   | Web browsing and interaction                     |
# +------------------+--------------------------------------------------+
# | **TextArena**    | Text-based game environments                     |
# +------------------+--------------------------------------------------+
# | **Atari**        | Classic Atari 2600 games                         |
# +------------------+--------------------------------------------------+
# | **Snake**        | Classic snake game                               |
# +------------------+--------------------------------------------------+

# %%
# Part 3: Connecting to Environments
# ----------------------------------
#
# OpenEnv provides three ways to connect to environments.

print("=" * 70)
print("   CONNECTION METHODS")
print("=" * 70)

# Import the environment client
try:
    from openspiel_env.client import OpenSpielEnv
    from openspiel_env.models import OpenSpielAction, OpenSpielObservation, OpenSpielState

    IMPORTS_OK = True
    print("✓ Imports successful")
except ImportError as e:
    IMPORTS_OK = False
    print(f"✗ Import error: {e}")

# %%
# Method 1: From Hugging Face Hub
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The easiest way to get started - automatically downloads and runs the container.
# Let's examine the actual method signature:

print("\n" + "-" * 70)
print("METHOD 1: FROM HUGGING FACE HUB")
print("-" * 70)

if IMPORTS_OK:
    import inspect

    if hasattr(OpenSpielEnv, "from_hub"):
        sig = inspect.signature(OpenSpielEnv.from_hub)
        print(f"\nSignature: OpenSpielEnv.from_hub{sig}")

        # Show docstring if available
        if OpenSpielEnv.from_hub.__doc__:
            doc_lines = OpenSpielEnv.from_hub.__doc__.strip().split("\n")[:3]
            print(f"Purpose: {doc_lines[0].strip()}")
    else:
        print("\nfrom_hub method not available in this version")

    print("\nUsage:")
    print("    env = OpenSpielEnv.from_hub('openenv/openspiel-env')")
    print("\nWhat happens:")
    print("    1. Pulls Docker image from HF registry")
    print("    2. Starts container on available port")
    print("    3. Connects via WebSocket")
    print("    4. Cleans up on close()")
else:
    print("\n(OpenEnv not installed - showing expected signature)")
    print("\nSignature: OpenSpielEnv.from_hub(repo_id, *, use_docker=True, ...)")

# %%
# Method 2: From Docker Image
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Use a locally built or pulled Docker image:

print("\n" + "-" * 70)
print("METHOD 2: FROM DOCKER IMAGE")
print("-" * 70)

if IMPORTS_OK:
    if hasattr(OpenSpielEnv, "from_docker_image"):
        sig = inspect.signature(OpenSpielEnv.from_docker_image)
        print(f"\nSignature: OpenSpielEnv.from_docker_image{sig}")

        if OpenSpielEnv.from_docker_image.__doc__:
            doc_lines = OpenSpielEnv.from_docker_image.__doc__.strip().split("\n")[:3]
            print(f"Purpose: {doc_lines[0].strip()}")
    else:
        print("\nfrom_docker_image method not available in this version")

    print("\nUsage:")
    print("    # Build image first:")
    print("    # docker build -t openspiel-env:latest ./envs/openspiel_env/server")
    print("    env = OpenSpielEnv.from_docker_image('openspiel-env:latest')")
else:
    print("\n(OpenEnv not installed - showing expected signature)")
    print("\nSignature: OpenSpielEnv.from_docker_image(image, provider=None, ...)")

# %%
# Method 3: Direct URL Connection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Connect to an already-running server:

print("\n" + "-" * 70)
print("METHOD 3: DIRECT URL CONNECTION")
print("-" * 70)

if IMPORTS_OK:
    sig = inspect.signature(OpenSpielEnv.__init__)
    print(f"\nSignature: OpenSpielEnv{sig}")
    print("\nUsage:")
    print("    # Start server first:")
    print("    # docker run -p 8000:8000 openenv/openspiel-env:latest")
    print("    env = OpenSpielEnv(base_url='http://localhost:8000')")
    print("\nNote: Does NOT manage container lifecycle - you control the server")
else:
    print("\n(OpenEnv not installed - showing expected signature)")
    print("\nSignature: OpenSpielEnv(base_url, connect_timeout_s=10.0, ...)")

# %%
# Using Context Managers
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Always use context managers to ensure proper cleanup. Let's verify the
# client supports the context manager protocol:

print("\n" + "-" * 70)
print("CONTEXT MANAGER SUPPORT")
print("-" * 70)

if IMPORTS_OK:
    has_enter = hasattr(OpenSpielEnv, "__enter__")
    has_exit = hasattr(OpenSpielEnv, "__exit__")
    print(f"\n__enter__ method: {'✓ Present' if has_enter else '✗ Missing'}")
    print(f"__exit__ method:  {'✓ Present' if has_exit else '✗ Missing'}")

    if has_enter and has_exit:
        print("\n✓ Context manager supported! Use with 'with' statement:")
        print("    with OpenSpielEnv(base_url='...') as env:")
        print("        result = env.reset()")
        print("        # ... use env ...")
        print("    # Automatically cleaned up")
else:
    print("\n(OpenEnv not installed)")
    print("Context managers are supported for automatic cleanup")

# %%
# Part 4: The Environment Loop
# ----------------------------
#
# Every OpenEnv interaction follows the same pattern:
#
# 1. ``reset()`` - Start a new episode
# 2. ``step(action)`` - Take action, get observation/reward
# 3. Repeat until ``done``
# 4. ``state()`` - Get episode metadata (optional)
#
# Let's demonstrate this with an actual episode:

print("=" * 70)
print("   THE ENVIRONMENT LOOP - LIVE DEMO")
print("=" * 70)
print()

# Run an actual demo episode
GRID_HEIGHT = 10
GRID_WIDTH = 5

# Create mock observation for demonstration
class DemoObservation:
    def __init__(self, info_state, legal_actions, done=False):
        self.info_state = info_state
        self.legal_actions = legal_actions
        self.done = done

class DemoResult:
    def __init__(self, observation, reward=0.0, done=False):
        self.observation = observation
        self.reward = reward
        self.done = done

# Initialize episode
ball_col = random.randint(0, GRID_WIDTH - 1)
paddle_col = GRID_WIDTH // 2

print(f"Episode Starting:")
print(f"  Ball column: {ball_col}")
print(f"  Paddle column: {paddle_col}")
print()

# Simulate the environment loop
step_count = 0
total_reward = 0.0

print("Step | Ball Row | Paddle | Action | Info State (first 10)")
print("-" * 65)

for ball_row in range(GRID_HEIGHT):
    # Build observation (same format as real OpenSpiel Catch)
    info_state = [0.0] * (GRID_HEIGHT * GRID_WIDTH)
    info_state[ball_row * GRID_WIDTH + ball_col] = 1.0  # Ball
    info_state[(GRID_HEIGHT - 1) * GRID_WIDTH + paddle_col] = 1.0  # Paddle

    obs = DemoObservation(info_state=info_state, legal_actions=[0, 1, 2])

    # Choose action (smart policy - move toward ball)
    if paddle_col < ball_col:
        action_id = 2  # RIGHT
    elif paddle_col > ball_col:
        action_id = 0  # LEFT
    else:
        action_id = 1  # STAY

    action_names = {0: "LEFT", 1: "STAY", 2: "RIGHT"}

    # Show state before action
    info_preview = [f"{v:.0f}" for v in info_state[:10]]
    print(f"  {step_count:2d}  |    {ball_row:2d}    |   {paddle_col}    | {action_names[action_id]:<5}  | {info_preview}")

    # Execute action
    if action_id == 0:
        paddle_col = max(0, paddle_col - 1)
    elif action_id == 2:
        paddle_col = min(GRID_WIDTH - 1, paddle_col + 1)

    step_count += 1

# Calculate final reward
caught = (paddle_col == ball_col)
reward = 1.0 if caught else 0.0

print("-" * 65)
print()
print(f"Episode Complete:")
print(f"  Steps: {step_count}")
print(f"  Ball landed at: column {ball_col}")
print(f"  Paddle position: column {paddle_col}")
print(f"  Reward: {reward}")
print(f"  Result: {'CAUGHT! ✓' if caught else 'MISSED! ✗'}")
print()
print("This is the exact same loop you'd run with a live server,")
print("just using local simulation for the game logic.")

# %%
# Part 5: Creating AI Policies
# ----------------------------
#
# A policy is a function that chooses actions based on observations.
# Let's create several policies of increasing sophistication.

import random
from typing import List
from dataclasses import dataclass


@dataclass
class PolicyResult:
    """Result of evaluating a policy."""

    name: str
    episodes: int
    wins: int
    total_reward: float
    avg_steps: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.episodes if self.episodes > 0 else 0.0


# %%
# Policy 1: Random Policy
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# The simplest policy - randomly choose from legal actions:


class RandomPolicy:
    """
    Random policy - baseline for comparison.

    Always picks a random action from the legal actions.
    Expected win rate for Catch: ~20% (1 in 5 columns)
    """

    name = "Random"

    def choose_action(self, observation) -> int:
        """Choose a random legal action."""
        return random.choice(observation.legal_actions)


# %%
# Policy 2: Heuristic Policy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A hand-coded policy that uses domain knowledge:


class SmartCatchPolicy:
    """
    Smart heuristic policy for the Catch game.

    Tracks the ball position and moves paddle toward it.
    Expected win rate: ~100% (optimal for Catch)
    """

    name = "Smart (Heuristic)"

    def __init__(self, grid_width: int = 5):
        self.grid_width = grid_width

    def choose_action(self, observation) -> int:
        """Move paddle toward ball position."""
        info_state = observation.info_state
        grid_width = self.grid_width

        # Find ball position (first 1.0 in the grid, excluding last row)
        ball_col = None
        for idx, val in enumerate(info_state[:-grid_width]):
            if abs(val - 1.0) < 0.01:
                ball_col = idx % grid_width
                break

        # Find paddle position (1.0 in last row)
        last_row = info_state[-grid_width:]
        paddle_col = None
        for idx, val in enumerate(last_row):
            if abs(val - 1.0) < 0.01:
                paddle_col = idx
                break

        if ball_col is None or paddle_col is None:
            return 1  # STAY if can't determine positions

        # Move toward ball
        if paddle_col < ball_col:
            return 2  # RIGHT
        elif paddle_col > ball_col:
            return 0  # LEFT
        else:
            return 1  # STAY


# %%
# Policy 3: Epsilon-Greedy Policy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Combines exploration (random) with exploitation (smart):


class EpsilonGreedyPolicy:
    """
    Epsilon-greedy policy - balances exploration and exploitation.

    With probability epsilon, takes random action (explore).
    Otherwise, uses smart policy (exploit).
    Epsilon decays over time to favor exploitation.
    """

    name = "Epsilon-Greedy"

    def __init__(self, epsilon: float = 0.3, decay: float = 0.99):
        self.epsilon = epsilon
        self.decay = decay
        self.smart_policy = SmartCatchPolicy()
        self.steps = 0

    def choose_action(self, observation) -> int:
        """Choose action with epsilon-greedy strategy."""
        self.steps += 1

        # Decay epsilon
        current_epsilon = self.epsilon * (self.decay**self.steps)

        if random.random() < current_epsilon:
            # Explore: random action
            return random.choice(observation.legal_actions)
        else:
            # Exploit: use smart policy
            return self.smart_policy.choose_action(observation)


# %%
# Policy 4: Always Stay Policy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A deliberately bad policy for comparison:


class AlwaysStayPolicy:
    """
    Always stay policy - deliberately bad baseline.

    Never moves the paddle. Only wins if ball lands on starting column.
    Expected win rate: ~20% (same as random)
    """

    name = "Always Stay"

    def choose_action(self, observation) -> int:
        """Always return STAY action."""
        return 1  # STAY


# %%
# Part 6: Running Evaluations
# ---------------------------
#
# Let's evaluate our policies! First, we'll create an evaluation function.


def evaluate_policy_live(
    policy,
    env,
    num_episodes: int = 50,
    game_name: str = "catch",
) -> PolicyResult:
    """
    Evaluate a policy against a live environment.

    Args:
        policy: Policy object with choose_action method
        env: Connected OpenSpielEnv client
        num_episodes: Number of episodes to run
        game_name: Name of the game to play

    Returns:
        PolicyResult with evaluation metrics
    """
    wins = 0
    total_reward = 0.0
    total_steps = 0

    for _ in range(num_episodes):
        result = env.reset()
        episode_steps = 0

        while not result.done:
            action_id = policy.choose_action(result.observation)
            action = OpenSpielAction(action_id=action_id, game_name=game_name)
            result = env.step(action)
            episode_steps += 1

        total_reward += result.reward if result.reward else 0
        total_steps += episode_steps
        if result.reward and result.reward > 0:
            wins += 1

    return PolicyResult(
        name=policy.name,
        episodes=num_episodes,
        wins=wins,
        total_reward=total_reward,
        avg_steps=total_steps / num_episodes,
    )


def evaluate_policy_simulated(
    policy,
    num_episodes: int = 50,
    grid_height: int = 10,
    grid_width: int = 5,
) -> PolicyResult:
    """
    Evaluate a policy using local simulation (no server needed).

    This simulates the Catch game locally for testing without a server.

    Args:
        policy: Policy object with choose_action method
        num_episodes: Number of episodes to run
        grid_height: Height of the game grid
        grid_width: Width of the game grid

    Returns:
        PolicyResult with evaluation metrics
    """
    wins = 0
    total_reward = 0.0
    total_steps = 0

    # Create a mock observation class
    class MockObservation:
        def __init__(self, info_state, legal_actions):
            self.info_state = info_state
            self.legal_actions = legal_actions

    for _ in range(num_episodes):
        # Initialize game
        ball_col = random.randint(0, grid_width - 1)
        paddle_col = grid_width // 2  # Start in center

        for step in range(grid_height):
            # Create observation
            info_state = [0.0] * (grid_height * grid_width)
            info_state[step * grid_width + ball_col] = 1.0  # Ball position
            info_state[(grid_height - 1) * grid_width + paddle_col] = 1.0  # Paddle

            observation = MockObservation(
                info_state=info_state, legal_actions=[0, 1, 2]
            )

            # Get action from policy
            action = policy.choose_action(observation)

            # Execute action
            if action == 0:  # LEFT
                paddle_col = max(0, paddle_col - 1)
            elif action == 2:  # RIGHT
                paddle_col = min(grid_width - 1, paddle_col + 1)
            # action == 1 is STAY, no movement

            total_steps += 1

        # Check if caught
        if paddle_col == ball_col:
            wins += 1
            total_reward += 1.0

    return PolicyResult(
        name=policy.name,
        episodes=num_episodes,
        wins=wins,
        total_reward=total_reward,
        avg_steps=total_steps / num_episodes,
    )


# %%
# Part 7: Policy Competition
# --------------------------
#
# Let's run a competition between all our policies!

# Create policy instances
policies = [
    RandomPolicy(),
    AlwaysStayPolicy(),
    SmartCatchPolicy(),
    EpsilonGreedyPolicy(epsilon=0.3),
]

# Check if we can connect to a live server
SERVER_URL = "http://localhost:8000"
USE_LIVE = False

if IMPORTS_OK:
    try:
        test_env = OpenSpielEnv(base_url=SERVER_URL)
        with test_env.sync() as client:
            pass  # Quick test to verify connection
        USE_LIVE = True
        print(f"✓ Connected to server at {SERVER_URL}")
    except Exception as e:
        USE_LIVE = False
        print(f"✗ No server running at {SERVER_URL}: {e}")

print("=" * 70)
if USE_LIVE:
    print("   POLICY COMPETITION - LIVE SERVER")
else:
    print("   POLICY COMPETITION - SIMULATION MODE")
print("=" * 70)
print()

NUM_EPISODES = 50
print(f"Running {NUM_EPISODES} episodes per policy...\n")

results = []

for policy in policies:
    print(f"  Evaluating {policy.name}...", end=" ", flush=True)

    if USE_LIVE:
        env = OpenSpielEnv(base_url=SERVER_URL)
        with env.sync() as client:
            result = evaluate_policy_live(policy, client, NUM_EPISODES)
    else:
        result = evaluate_policy_simulated(policy, NUM_EPISODES)

    results.append(result)
    print(f"Win rate: {result.win_rate * 100:.1f}%")

# %%
# Display Results
# ~~~~~~~~~~~~~~~

print()
print("=" * 70)
print("   FINAL RESULTS")
print("=" * 70)
print()

# Sort by win rate (descending)
results.sort(key=lambda r: r.win_rate, reverse=True)

# Display leaderboard
print(f"{'Rank':<6}{'Policy':<20}{'Win Rate':<12}{'Avg Steps':<12}{'Wins'}")
print("-" * 60)

for i, result in enumerate(results):
    rank = f"#{i + 1}"
    bar = "█" * int(result.win_rate * 20)
    print(
        f"{rank:<6}{result.name:<20}{result.win_rate * 100:>5.1f}%{'':<5}"
        f"{result.avg_steps:>6.1f}{'':<6}{result.wins}/{result.episodes}"
    )

print()
print("-" * 70)
print()
print("Key Insights:")
print("  • Random/AlwaysStay: ~20% (baseline - relies on luck)")
print("  • Smart Heuristic:   ~100% (optimal for Catch)")
print("  • Epsilon-Greedy:    ~85%+ (balances exploration/exploitation)")
print()

# %%
# Part 8: Working with Different Games
# ------------------------------------
#
# OpenSpiel supports multiple games. Let's create actual action instances
# for different games and examine their structure:

print("=" * 70)
print("   SWITCHING GAMES - ACTUAL ACTION INSTANCES")
print("=" * 70)
print()

# Create actual action instances for different games
if IMPORTS_OK:
    from openspiel_env.models import OpenSpielAction as ActionModel

    # Catch actions
    print("CATCH GAME ACTIONS:")
    print("-" * 40)
    catch_actions = {
        0: "Move LEFT",
        1: "STAY in place",
        2: "Move RIGHT",
    }
    for action_id, description in catch_actions.items():
        action = ActionModel(action_id=action_id, game_name="catch")
        print(f"  {action}  # {description}")

    print()

    # 2048 actions
    print("2048 GAME ACTIONS:")
    print("-" * 40)
    game_2048_actions = {
        0: "Slide UP",
        1: "Slide RIGHT",
        2: "Slide DOWN",
        3: "Slide LEFT",
    }
    for action_id, description in game_2048_actions.items():
        action = ActionModel(action_id=action_id, game_name="2048")
        print(f"  {action}  # {description}")

    print()

    # Tic-Tac-Toe actions
    print("TIC-TAC-TOE ACTIONS:")
    print("-" * 40)
    print("  Grid positions 0-8 (left-to-right, top-to-bottom):")
    print("    0 | 1 | 2")
    print("   ---|---|---")
    print("    3 | 4 | 5")
    print("   ---|---|---")
    print("    6 | 7 | 8")
    print()
    # Show a few examples
    for pos in [0, 4, 8]:
        action = ActionModel(action_id=pos, game_name="tic_tac_toe")
        corner = {0: "top-left", 4: "center", 8: "bottom-right"}[pos]
        print(f"  {action}  # {corner}")

    print()

    # Blackjack actions
    print("BLACKJACK ACTIONS:")
    print("-" * 40)
    blackjack_actions = {
        0: "STAND (keep current hand)",
        1: "HIT (request another card)",
    }
    for action_id, description in blackjack_actions.items():
        action = ActionModel(action_id=action_id, game_name="blackjack")
        print(f"  {action}  # {description}")

else:
    # Fallback using dataclass
    from dataclasses import dataclass

    @dataclass
    class ActionDemo:
        action_id: int
        game_name: str

    print("CATCH GAME ACTIONS:")
    print("-" * 40)
    for action_id, desc in [(0, "LEFT"), (1, "STAY"), (2, "RIGHT")]:
        print(f"  ActionDemo(action_id={action_id}, game_name='catch')  # {desc}")

    print()
    print("2048 GAME ACTIONS:")
    print("-" * 40)
    for action_id, desc in [(0, "UP"), (1, "RIGHT"), (2, "DOWN"), (3, "LEFT")]:
        print(f"  ActionDemo(action_id={action_id}, game_name='2048')  # {desc}")

print()
print("-" * 70)
print("Each game has its own action space - check legal_actions in observation!")

# %%
# Part 9: Multi-Player Games
# --------------------------
#
# Some games like Tic-Tac-Toe and Kuhn Poker support multiple players.
# Let's create actual observation instances to understand the structure:

print("=" * 70)
print("   MULTI-PLAYER GAMES - OBSERVATION STRUCTURE")
print("=" * 70)
print()

# Create observation instances for multi-player games
if IMPORTS_OK:
    from openspiel_env.models import OpenSpielObservation as ObsModel

    # Single-player observation (like Catch)
    print("SINGLE-PLAYER OBSERVATION (Catch):")
    print("-" * 50)
    single_player_obs = ObsModel(
        info_state=[0.0, 0.0, 1.0, 0.0, 0.0] + [0.0] * 45,
        legal_actions=[0, 1, 2],
        game_phase="playing",
        current_player_id=0,
        opponent_last_action=None,
    )
    print(f"  current_player_id:   {single_player_obs.current_player_id}  # Always 0 (you)")
    print(f"  opponent_last_action: {single_player_obs.opponent_last_action}  # None (no opponent)")
    print(f"  legal_actions:       {single_player_obs.legal_actions}")
    print(f"  game_phase:          {single_player_obs.game_phase!r}")
    print()

    # Multi-player observation - your turn (like Tic-Tac-Toe)
    print("MULTI-PLAYER OBSERVATION (Tic-Tac-Toe, YOUR turn):")
    print("-" * 50)
    your_turn_obs = ObsModel(
        info_state=[1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],  # X at 0, O at 4
        legal_actions=[1, 2, 3, 5, 6, 7, 8],  # Available positions
        game_phase="playing",
        current_player_id=0,  # Your turn!
        opponent_last_action=4,  # Opponent played center
    )
    print(f"  current_player_id:   {your_turn_obs.current_player_id}  # 0 = YOUR turn")
    print(f"  opponent_last_action: {your_turn_obs.opponent_last_action}  # Opponent played position 4 (center)")
    print(f"  legal_actions:       {your_turn_obs.legal_actions}")
    print(f"  game_phase:          {your_turn_obs.game_phase!r}")
    print()

    # Multi-player observation - opponent's turn
    print("MULTI-PLAYER OBSERVATION (Tic-Tac-Toe, OPPONENT's turn):")
    print("-" * 50)
    opponent_turn_obs = ObsModel(
        info_state=[1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],  # X at 0,8; O at 4
        legal_actions=[],  # No actions available when it's opponent's turn
        game_phase="playing",
        current_player_id=1,  # Opponent's turn
        opponent_last_action=None,  # Will be set after they move
    )
    print(f"  current_player_id:   {opponent_turn_obs.current_player_id}  # 1 = OPPONENT's turn")
    print(f"  legal_actions:       {opponent_turn_obs.legal_actions}  # Empty - wait for opponent")
    print(f"  game_phase:          {opponent_turn_obs.game_phase!r}")
    print()

    # Terminal state observation
    print("TERMINAL OBSERVATION (Game Over):")
    print("-" * 50)
    terminal_obs = ObsModel(
        info_state=[1.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],  # X wins top row
        legal_actions=[],  # No more moves
        game_phase="terminal",
        current_player_id=-1,  # No current player
        opponent_last_action=4,
    )
    print(f"  current_player_id:   {terminal_obs.current_player_id}  # -1 = Game over")
    print(f"  game_phase:          {terminal_obs.game_phase!r}")
    print(f"  legal_actions:       {terminal_obs.legal_actions}  # Empty - game ended")

else:
    # Fallback demonstration
    from dataclasses import dataclass
    from typing import List, Optional

    @dataclass
    class ObsDemo:
        current_player_id: int
        opponent_last_action: Optional[int]
        legal_actions: List[int]
        game_phase: str

    print("SINGLE-PLAYER (Catch):")
    print(f"  current_player_id: 0  # Always your turn")
    print(f"  opponent_last_action: None")
    print()

    print("MULTI-PLAYER - YOUR TURN (Tic-Tac-Toe):")
    print(f"  current_player_id: 0  # 0 = your turn")
    print(f"  opponent_last_action: 4  # What opponent just played")
    print(f"  legal_actions: [1, 2, 3, 5, 6, 7, 8]  # Available moves")
    print()

    print("MULTI-PLAYER - OPPONENT'S TURN:")
    print(f"  current_player_id: 1  # Wait for opponent")
    print(f"  legal_actions: []  # Can't move during opponent's turn")

print()
print("-" * 70)
print("KEY INSIGHT: Only act when current_player_id == 0 (your turn)!")
print("The environment automatically handles opponent moves.")

# %%
# Summary
# -------
#
# In this notebook, you learned:
#
# **Connection Methods:**
#
# - ``from_hub()`` - Auto-download from Hugging Face
# - ``from_docker_image()`` - Use local Docker image
# - Direct URL - Connect to running server
#
# **Creating Policies:**
#
# - Random: Baseline comparison
# - Heuristic: Domain knowledge encoded
# - Epsilon-Greedy: Balance exploration/exploitation
#
# **Running Evaluations:**
#
# - Measure win rates and rewards
# - Compare policy performance
# - Run competitions
#
# **Multi-Game Support:**
#
# - Switch games via ``game_name`` parameter
# - Handle multi-player games
# - Work with different action spaces
#
# Next Steps
# ----------
#
# **Continue to Notebook 3: Building & Sharing Environments**
#
# In the next notebook, you'll:
#
# - Create your own custom environment
# - Package it with Docker
# - Deploy to Hugging Face Hub
# - Share with the community
