---
title: Wildfire Environment Server
emoji: 🔥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - wildfire
  - simulation
---

# 🌲 Wildfire Environment

Autonomous wildfire-control simulation for reinforcement-learning agents, built on the [OpenEnv](https://github.com/openenv) framework.
Agents must contain spreading fires using **water**, **firebreaks**, and **timing strategies** under changing **wind** and **humidity** conditions.

[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://hub.docker.com/)
[![Python](https://img.shields.io/badge/python-3.10+-green)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/backend-fastapi-teal)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## 📋 Table of Contents

1. [Why Wildfire Simulation?](#-why-wildfire-simulation)
2. [Quick Start](#-quick-start)
3. [Environment Overview](#-environment-overview)
4. [Grid Format & Encoding](#-grid-format--encoding)
5. [Actions](#-actions)
6. [Observations](#-observations)
7. [Reward Structure](#-reward-structure)
8. [Fire Spread Mechanics](#-fire-spread-mechanics)
9. [Configuration](#-configuration)
10. [Installation & Usage](#-installation--usage)
11. [API Reference](#-api-reference)
12. [Examples](#-examples)
13. [Web Interface](#-web-interface)
14. [Troubleshooting](#-troubleshooting)
15. [References](#-references)

---

## 🔥 Why Wildfire Simulation?

Wildland fires are intensifying globally due to climate change — increasing the urgency for **AI-assisted decision-making**.
This environment explores how intelligent systems can **control** fire spread in real time, under limited resources.

### Research Motivation
✅ Based on real wildfire science inspired by:
- **Rothermel Surface Fire Spread Model** (USDA Forest Service)
- **MITRE Fireline's SimFire** — physics-informed RL fire simulator
- **SimHarness** — RL evaluation for disaster response

### Application Goals
| Research Theme | Role in This Environment |
|---|---|
| Resource-Constrained Planning | Finite water + firebreak budgets |
| Fire Spread + Containment Strategy | Directional wind & moisture effects |
| Disaster Response RL | Safety-focused reward design |
| LLM Agents for Control Tasks | Text-based action decision making |

This makes WildfireEnv a **fast, controllable**, and **open benchmark** for applied RL and LLM reasoning.

---

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Build base image (first time only)
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .

# Build wildfire environment
docker build -t wildfire-env:latest -f envs/wildfire_env/server/Dockerfile .

# Run container
docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true wildfire-env:latest
```

**Note:** The web interface can be enabled with `ENABLE_WEB_INTERFACE=true`. Access it at `http://localhost:8000/web` when enabled.

### Basic Python Client

```python
from envs.wildfire_env import WildfireEnv, WildfireAction

# Connect to running server
env = WildfireEnv(base_url="http://localhost:8000")

# Reset environment
result = env.reset()
obs = result.observation
print(f"Grid: {obs.width}x{obs.height}, Fires: {obs.burning_count}, Water: {obs.remaining_water}")

# Take action (water a burning cell)
result = env.step(WildfireAction(action="water", x=10, y=15))
print(f"Reward: {result.reward:.2f}, Burning: {result.observation.burning_count}")

# Create firebreak
result = env.step(WildfireAction(action="break", x=12, y=15))

# Wait (fire spreads)
result = env.step(WildfireAction(action="wait"))

env.close()
```

---

## 🔥 Environment Overview

This environment models **forest-fire dynamics** influenced by:
- **Wind direction** (8 directions + calm) - accelerates fire spread in wind direction
- **Humidity** (0.0-1.0) - suppresses ignition probability
- **Fuel type and spread rate** - vegetation burns and spreads to neighbors
- **Limited resources** (water units, break materials) - strategic resource management
- **Time pressure** (each step costs small reward penalty)

The goal is to **minimize fire spread** and **total burned area** while using resources efficiently.

### Episode Termination

An episode ends when:
- **All fires are extinguished** (`burning_count == 0`) - **Success!**
- **Maximum steps reached** (`step_count >= max_steps`) - Time limit exceeded

---

## 🧱 Grid Format & Encoding

### Grid Structure

The grid is returned as a **flat 1D array** in the observation. To access cell at position `(x, y)`:

```python
index = y * width + x
cell_value = observation.grid[index]
```

**Example:** For a 32×32 grid, cell at (10, 15):
```python
index = 15 * 32 + 10  # = 490
cell_value = observation.grid[490]
```

### Cell Encoding

| Code | Meaning        | Color (Visualization) | Behavior |
|------|----------------|-----------------------|----------|
| `0`  | Ash (burned)   | Black ⚫              | Burned out, cannot reignite |
| `1`  | Fuel           | Green 🟩              | Healthy vegetation, can ignite |
| `2`  | Burning        | Red 🔥                | Currently on fire, spreads to neighbors |
| `3`  | Firebreak      | Brown 🟫              | Barrier, fire cannot cross |
| `4`  | Water/Damp     | Blue 🔵               | Dampened, immune to ignition temporarily |

### Grid Visualization Example

```python
import numpy as np

obs = env.reset().observation
grid_2d = np.array(obs.grid).reshape(obs.height, obs.width)

# Now grid_2d[y][x] gives the cell value at position (x, y)
print(grid_2d[15][10])  # Cell at x=10, y=15
```

---

## 🎮 Actions

### Action Types

#### 1. `water` - Apply Water
**Extinguishes burning cells and dampens fuel to prevent ignition.**

```python
WildfireAction(action="water", x=10, y=15)
```

**Effects:**
- **Burning cell (2)**: Extinguishes → becomes Water/Damp (4), gives **+0.25 reward**
- **Fuel cell (1)**: Dampens → becomes Water/Damp (4), gives **-0.10 reward** (preventive, slight penalty)
- **Water/Damp cell (4)**: Redundant watering, gives **-0.05 reward**
- **Ash/Break (0, 3)**: Wasteful, gives **-0.05 reward**

**Resource Cost:** 1 water unit per action
**Requires:** `remaining_water > 0` and valid coordinates

**Best Use:** Extinguish active fires before they spread

---

#### 2. `break` - Create Firebreak
**Builds a fire-resistant barrier that stops fire spread.**

```python
WildfireAction(action="break", x=12, y=15)
```

**Effects:**
- **Fuel/Water cell (1, 4)**: Creates firebreak → becomes Firebreak (3), gives **+0.15 reward**
- **Burning cell (2)**: Extinguishes → becomes Firebreak (3), gives **-0.02 reward** (less effective than water)
- **Firebreak (3)**: Redundant, gives **-0.01 reward**
- **Ash (0)**: Wasteful, gives **-0.02 reward**

**Resource Cost:** 1 firebreak material per action
**Requires:** `remaining_breaks > 0` and valid coordinates

**Best Use:** Create barriers ahead of fire front to contain spread

---

#### 3. `wait` - Do Nothing
**Let natural fire dynamics occur (fire spreads).**

```python
WildfireAction(action="wait")
```

**Effects:**
- No resource cost
- No coordinate required
- Fire spreads naturally to neighboring cells
- Small time penalty (-0.01 reward per step)

**Best Use:** When fire is contained, waiting for it to burn out

---

### Invalid Actions

Actions that fail (give **-0.05 reward**):
- Invalid coordinates (out of bounds)
- Using water when `remaining_water == 0`
- Using break when `remaining_breaks == 0`
- Missing required coordinates for water/break actions

---

## 👁️ Observations

### `WildfireObservation`

Returned after every `reset()` or `step()`:

```python
@dataclass
class WildfireObservation(Observation):
    grid: List[int]          # Flat array: [1,1,2,1,...] length = width × height
    width: int               # Grid width (default: 32)
    height: int              # Grid height (default: 32)
    step: int                # Current step number (0 at reset)
    wind_dir: str            # "N", "NE", "E", "SE", "S", "SW", "W", "NW", "CALM"
    humidity: float          # [0.0, 1.0] - higher = less fire spread
    burning_count: int       # Number of cells currently on fire
    burned_count: int        # Total number of ash cells (cumulative)
    remaining_water: int     # Water units left
    remaining_breaks: int    # Firebreak materials left
    reward_hint: float       # Shaping reward (for debugging)
    done: bool               # Episode ended?
    reward: float            # Step reward
```

### Example Observation

```python
result = env.reset()
obs = result.observation

print(f"Step: {obs.step}")                    # 0
print(f"Grid size: {obs.width}x{obs.height}") # 32x32
print(f"Grid cells: {len(obs.grid)}")         # 1024
print(f"Active fires: {obs.burning_count}")   # 2
print(f"Wind: {obs.wind_dir}")                # "NE"
print(f"Humidity: {obs.humidity:.2f}")        # 0.24
print(f"Water left: {obs.remaining_water}")   # 8
print(f"Breaks left: {obs.remaining_breaks}") # 50
```

---

## 💰 Reward Structure

### Step Rewards

| Action | Condition | Reward |
|--------|-----------|--------|
| **Water burning cell** | Extinguishes fire | **+0.25** |
| **Water fuel cell** | Preventive dampening | **-0.10** |
| **Create firebreak** | From fuel/water | **+0.15** |
| **Fire spreads** | Each new burning cell | **-0.15 per cell** |
| **Fire shrinks** | Each extinguished cell | **+0.10 per cell** |
| **New burned area** | Each cell turns to ash | **-0.05 per cell** |
| **Time penalty** | Every step | **-0.01** |
| **Invalid action** | Out of bounds, no resources | **-0.05** |
| **Redundant action** | Watering already damp cell | **-0.05** |

### Episode End Bonuses

When episode terminates (`done == True`):

- **Fire contained** (`burning_count == 0`):
  - **+0.5** base bonus
  - **+0.5 × saved_ratio** bonus (proportion of cells not burned)

- **Fallback reward**:
  - **+0.2 × (1.0 - burned_ratio)** bonus

**Example:** Perfect containment (no burned cells):
```text
Reward = +0.5 + 0.5 × 1.0 = +1.0
```

### Reward Interpretation

- **Positive rewards**: Good containment actions, extinguishing fires
- **Negative rewards**: Fire spread, resource waste, time penalty
- **Goal**: Maximize cumulative reward = minimize fire damage

---

## 🌪️ Fire Spread Mechanics

### Spread Model

Fire spreads using an **8-directional neighbor model**:

1. **Burning cells persist** for `burn_lifetime = 3` ticks before turning to ash
2. Each burning cell can ignite **neighboring fuel cells** (8 directions)
3. Spread probability depends on:
   - **Base ignition probability**: `0.30` (30% chance)
   - **Humidity factor**: `(1.0 - humidity)` - higher humidity = less spread
   - **Wind multiplier**:
     - **+2.0x** in wind direction
     - **+0.5x** against wind
     - **+1.0x** perpendicular
   - **Diagonal factor**: `0.6x` for diagonal neighbors (slower spread)

4. **Water/Damp cells (4)** are **immune** to ignition while damp
5. **Firebreaks (3)** **cannot** be crossed by fire
6. **Ash cells (0)** cannot reignite

### Wind Effects

| Wind Direction | Effect on Fire Spread |
|----------------|----------------------|
| **In wind direction** | 2× faster ignition probability |
| **Against wind** | 0.5× slower ignition probability |
| **Perpendicular** | Normal (1×) ignition probability |
| **CALM** | No directional bias |

### Water Dampening Duration

Watered cells (4) remain damp for **6 ticks** before reverting to fuel (1).

### Example Fire Spread

```
Step 0:     Step 1:     Step 2:
🟩🟩🟩      🟩🟥🟩      🟫🟥🟫
🟩🟥🟩  →   🟥🟥🟥  →   🟥🟥🟥  (Wind: E, spreading east)
🟩🟩🟩      🟩🟥🟩      🟫🟥🟫
```

---

## ⚙️ Configuration

### Environment Variables

Set these **before starting the server**:

| Variable | Description | Default | Range |
|-----------|-------------|---------|-------|
| `WILDFIRE_WIDTH` | Grid width in cells | `32` | 8-128 |
| `WILDFIRE_HEIGHT` | Grid height in cells | `32` | 8-128 |
| `WILDFIRE_HUMIDITY` | Initial humidity level | `0.25` | 0.0-1.0 |
| `WILDFIRE_WIND` | Wind direction (fixed) | Random | `N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`, `CALM` |
| `WILDFIRE_SEED` | Random seed | `3407` | Any integer |
| `WILDFIRE_MAX_STEPS` | Max steps per episode | `128` | 10-1000 |
| `WILDFIRE_WATER_CAPACITY` | Initial water units | `8` | 1-100 |
| `WILDFIRE_BREAK_CAPACITY` | Initial firebreak materials | `50` | 1-200 |

### Python API Configuration

```python
from envs.wildfire_env.server.wildfire_environment import WildfireEnvironment

env = WildfireEnvironment(
    width=64,
    height=64,
    humidity=0.3,
    init_sources=3,          # Number of initial fires
    max_steps=200,
    water_capacity=10,
    break_capacity=75,
    seed=42
)
```

### Docker Configuration

```bash
docker run -p 8000:8000 \
  -e WILDFIRE_WIDTH=64 \
  -e WILDFIRE_HEIGHT=64 \
  -e WILDFIRE_HUMIDITY=0.4 \
  -e WILDFIRE_WIND=N \
  -e WILDFIRE_WATER_CAPACITY=12 \
  wildfire-env:latest
```

### Custom Configuration

```bash
# Build and run with custom configuration
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .
docker build -t wildfire-env:latest -f envs/wildfire_env/server/Dockerfile .
docker run -p 8000:8000 \
  -e ENABLE_WEB_INTERFACE=true \
  -e WILDFIRE_WIDTH=64 \
  -e WILDFIRE_HEIGHT=64 \
  -e WILDFIRE_HUMIDITY=0.5 \
  wildfire-env:latest
```

---

## 🚀 Installation & Usage

### Option 1: Docker (Recommended)

**Manual setup:**
```bash
# Build base image (first time only)
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .

# Build wildfire environment
docker build -t wildfire-env:latest -f envs/wildfire_env/server/Dockerfile .

# Run container
docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true wildfire-env:latest
```

This approach:
- Builds the base image if needed
- Rebuilds the wildfire image
- Starts the container
- Shows logs in real-time

**Alternative: Using build_docker.sh script:**
```bash
# Build base image (first time only)
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .

# Build wildfire environment using the script
cd src/envs/wildfire_env/server
./build_docker.sh

# Run container
docker run -d -p 8000:8000 --name wildfire-env-container wildfire-env:latest

# View logs
docker logs -f wildfire-env-container

# Stop container
docker stop wildfire-env-container

# Remove container
docker rm wildfire-env-container
```

### Option 2: Local Development (No Docker)

**Requirements:**
```bash
pip install fastapi uvicorn numpy matplotlib requests
```

**Run server:**
```bash
# From OpenEnv root directory
python -m envs.wildfire_env.server.app
```

**Or with environment variables:**
```bash
WILDFIRE_WIDTH=64 WILDFIRE_HUMIDITY=0.3 python -m envs.wildfire_env.server.app
```

---

## 📚 API Reference

### Client Class

```python
from envs.wildfire_env import WildfireEnv

# Connect to existing server
env = WildfireEnv(base_url="http://localhost:8000")

# Or create from Docker image
env = WildfireEnv.from_docker_image("wildfire-env:latest")
```

### Methods

#### `reset() -> StepResult[WildfireObservation]`

Resets the environment to initial state.

```python
result = env.reset()
obs = result.observation
print(f"New episode: {obs.step == 0}")
```

#### `step(action: WildfireAction) -> StepResult[WildfireObservation]`

Takes an action and returns new observation.

```python
action = WildfireAction(action="water", x=10, y=15)
result = env.step(action)
print(f"Reward: {result.reward}, Done: {result.done}")
```

#### `state -> WildfireState`

Access current environment state.

```python
state = env.state
print(f"Episode ID: {state.episode_id}")
print(f"Total burned: {state.total_burned}")
print(f"Total extinguished: {state.total_extinguished}")
```

#### `close()`

Closes the connection (for HTTP clients, this is a no-op but good practice).

```python
env.close()
```

### Data Classes

#### `WildfireAction`

```python
@dataclass
class WildfireAction(Action):
    action: str              # "water" | "break" | "wait"
    x: Optional[int] = None  # Target X coordinate (required for water/break)
    y: Optional[int] = None  # Target Y coordinate (required for water/break)
```

**Examples:**
```python
WildfireAction(action="water", x=10, y=15)
WildfireAction(action="break", x=12, y=15)
WildfireAction(action="wait")  # x, y not needed
```

#### `WildfireObservation`

See [Observations](#-observations) section for full details.

#### `WildfireState`

```python
@dataclass
class WildfireState(State):
    episode_id: str
    step_count: int
    total_burned: int
    total_extinguished: int
    last_action: str
    width: int
    height: int
    wind_dir: str
    humidity: float
    remaining_water: int
    remaining_breaks: int
    grid: List[int]
    burn_timers: List[int]
```

---

## 📖 Examples

### Example 1: Simple Containment Strategy

```python
from envs.wildfire_env import WildfireEnv, WildfireAction
import numpy as np

env = WildfireEnv(base_url="http://localhost:8000")
result = env.reset()
obs = result.observation

grid_2d = np.array(obs.grid).reshape(obs.height, obs.width)
total_reward = 0

while not result.done:
    # Find burning cells
    burning_indices = np.where(grid_2d == 2)

    if len(burning_indices[0]) > 0 and obs.remaining_water > 0:
        # Water the first burning cell
        y, x = burning_indices[0][0], burning_indices[1][0]
        action = WildfireAction(action="water", x=int(x), y=int(y))
    else:
        # Wait if no water or no fires
        action = WildfireAction(action="wait")

    result = env.step(action)
    obs = result.observation
    total_reward += result.reward or 0

    # Update grid
    grid_2d = np.array(obs.grid).reshape(obs.height, obs.width)

    print(f"Step {obs.step}: Burning={obs.burning_count}, Reward={result.reward:.3f}")

print(f"\nEpisode ended. Total reward: {total_reward:.2f}")
print(f"Final stats: Burned={obs.burned_count}, Extinguished={env.state.total_extinguished}")
env.close()
```

### Example 2: Firebreak Strategy

```python
from envs.wildfire_env import WildfireEnv, WildfireAction
import numpy as np

env = WildfireEnv(base_url="http://localhost:8000")
result = env.reset()
obs = result.observation

def create_firebreak_barrier(obs, env):
    """Create firebreak ahead of fire front based on wind direction."""
    grid_2d = np.array(obs.grid).reshape(obs.height, obs.width)
    wind = obs.wind_dir

    # Find burning cells
    burning_y, burning_x = np.where(grid_2d == 2)

    if len(burning_x) == 0 or obs.remaining_breaks == 0:
        return WildfireAction(action="wait")

    # Calculate fire front position
    if wind == "E":
        target_x = int(np.max(burning_x)) + 2  # Ahead of easternmost fire
        target_y = int(np.mean(burning_y))
    elif wind == "W":
        target_x = int(np.min(burning_x)) - 2
        target_y = int(np.mean(burning_y))
    elif wind == "N":
        target_x = int(np.mean(burning_x))
        target_y = int(np.min(burning_y)) - 2
    elif wind == "S":
        target_x = int(np.mean(burning_x))
        target_y = int(np.max(burning_y)) + 2
    else:
        # Fallback: water nearest burning cell
        return WildfireAction(action="water", x=int(burning_x[0]), y=int(burning_y[0]))

    # Ensure within bounds
    target_x = max(0, min(obs.width - 1, target_x))
    target_y = max(0, min(obs.height - 1, target_y))

    return WildfireAction(action="break", x=target_x, y=target_y)

total_reward = 0
while not result.done:
    action = create_firebreak_barrier(obs, env)
    result = env.step(action)
    obs = result.observation
    total_reward += result.reward or 0

    if obs.step % 10 == 0:
        print(f"Step {obs.step}: Fires={obs.burning_count}, Water={obs.remaining_water}, Breaks={obs.remaining_breaks}")

env.close()
```

### Example 3: Visualization with Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from envs.wildfire_env import WildfireEnv, WildfireAction

env = WildfireEnv(base_url="http://localhost:8000")
result = env.reset()
obs = result.observation

# Setup colormap
cmap = mcolors.ListedColormap([
    "black",         # 0 = ash
    "green",         # 1 = fuel
    "red",           # 2 = burning
    "saddlebrown",   # 3 = firebreak
    "blue"           # 4 = water
])
norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)

fig, ax = plt.subplots(figsize=(8, 8))
plt.ion()

for step in range(50):
    if result.done:
        break

    # Render grid
    grid_2d = np.array(obs.grid).reshape(obs.height, obs.width)
    ax.clear()
    ax.imshow(grid_2d, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title(
        f"Step {obs.step} | Fires: {obs.burning_count} | Burned: {obs.burned_count}\n"
        f"Wind: {obs.wind_dir} | Humidity: {obs.humidity:.2f} | "
        f"Water: {obs.remaining_water} | Breaks: {obs.remaining_breaks}"
    )
    plt.pause(0.1)

    # Take action (simple: water first burning cell)
    if obs.burning_count > 0 and obs.remaining_water > 0:
        burning_indices = np.where(grid_2d == 2)
        if len(burning_indices[0]) > 0:
            y, x = burning_indices[0][0], burning_indices[1][0]
            action = WildfireAction(action="water", x=int(x), y=int(y))
        else:
            action = WildfireAction(action="wait")
    else:
        action = WildfireAction(action="wait")

    result = env.step(action)
    obs = result.observation

plt.ioff()
plt.show()
env.close()
```

### Example 4: Training Loop for RL

```python
from envs.wildfire_env import WildfireEnv, WildfireAction
import random

env = WildfireEnv(base_url="http://localhost:8000")

num_episodes = 10
episode_rewards = []

for episode in range(num_episodes):
    result = env.reset()
    obs = result.observation
    episode_reward = 0
    episode_steps = 0

    while not result.done:
        # Random policy (replace with your RL agent)
        if random.random() < 0.4 and obs.remaining_water > 0:
            action = WildfireAction(
                action="water",
                x=random.randint(0, obs.width - 1),
                y=random.randint(0, obs.height - 1)
            )
        elif random.random() < 0.3 and obs.remaining_breaks > 0:
            action = WildfireAction(
                action="break",
                x=random.randint(0, obs.width - 1),
                y=random.randint(0, obs.height - 1)
            )
        else:
            action = WildfireAction(action="wait")

        result = env.step(action)
        obs = result.observation
        episode_reward += result.reward or 0
        episode_steps += 1

    episode_rewards.append(episode_reward)
    state = env.state
    print(
        f"Episode {episode + 1}: "
        f"Reward={episode_reward:.2f}, "
        f"Steps={episode_steps}, "
        f"Burned={state.total_burned}, "
        f"Extinguished={state.total_extinguished}"
    )

print(f"\nAverage reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
env.close()
```

---

## 🌐 Web Interface

The Wildfire Environment includes a **custom web interface** with visual grid display and wildfire-specific features.

### Accessing the Web Interface

#### Using Docker

```bash
# Build base image (first time only)
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .

# Build wildfire environment
docker build -t wildfire-env:latest -f envs/wildfire_env/server/Dockerfile .

# Run container
docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true wildfire-env:latest
```

Then open: `http://localhost:8000/web`

#### Local Testing (No Docker)

```bash
# Enable web interface with flag
ENABLE_WEB_INTERFACE=true PYTHONPATH=src uvicorn src.envs.wildfire_env.server.app:app --reload --host 0.0.0.0 --port 8000
```

### Web Interface Features

#### Left Pane: Action Interface
- **Wildfire-specific action form**
  - Action dropdown: Water (Extinguish Fire), Break (Create Firebreak), Wait (Do Nothing)
  - Coordinate inputs (X, Y) - auto-populated when clicking grid cells
  - Coordinates show/hide based on action type
- **Environment stats display**
  - Step count
  - Water remaining
  - Breaks remaining
  - Burning cells count
- **Current state display**
  - Status (Reset/Running)
  - Episode ID
  - Wind direction
  - Humidity
- **Control buttons**
  - Reset Environment
  - Get State

#### Right Pane: Visual Grid & Logs
- **Visual 2D Grid Display** 🔥
  - 16×16 grid rendered as color-coded cells
  - **Color coding:**
    - 🟩 **Green** = Fuel (safe, value 1)
    - 🔥 **Orange/Red** = Burning (fire, value 2)
    - ⬛ **Dark Gray** = Ash (burned, value 0)
    - 🟫 **Brown** = Firebreak (value 3)
    - 🟦 **Blue** = Watered/Damp (value 4)
  - **Interactive:** Click cells to set coordinates for water/break actions
  - **Auto-updates:** Grid refreshes automatically via WebSocket
- **Legend**
  - Color-coded legend explaining all cell types
- **Action history**
  - Log of all actions with timestamps
  - Shows action, observation, reward, and done status

#### Additional Features
- **WebSocket connection** - Real-time state updates without page refresh
- **Instructions panel** - Collapsible environment documentation
- **Grid status indicator** - Shows grid dimensions and cell count

### Using the Web Interface

1. **Start the server** (see above)
2. **Open browser** to: `http://localhost:8000/web`
3. **Click "Reset Environment"** to initialize and display the grid
4. **Interact with the grid:**
   - Click on a cell to set coordinates for water/break actions
   - Or manually enter X, Y coordinates
5. **Select action:**
   - Choose `water`, `break`, or `wait` from the dropdown
6. **Click "Execute Action"**
7. **Watch the grid update in real-time:**
   - Fire spreads automatically
   - Cells change color based on state
   - Stats update automatically
8. **Monitor resources** in the stats panel (water, breaks, burning count)

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Connection Errors

**Problem:** `ConnectionRefusedError` or `Cannot connect to server`

**Solutions:**
- Verify server is running: `curl http://localhost:8000/health`
- Check Docker container: `docker ps | grep wildfire`
- Ensure port 8000 is not in use: `lsof -i :8000`

#### 2. Index Errors

**Problem:** `IndexError: list index out of range`

**Solution:** Ensure coordinates are within bounds:
```python
# Always check bounds before accessing
if 0 <= x < obs.width and 0 <= y < obs.height:
    action = WildfireAction(action="water", x=x, y=y)
```

#### 3. Invalid Action Warnings

**Problem:** Actions returning -0.05 reward repeatedly

**Solutions:**
- Check `remaining_water` and `remaining_breaks` before using resources
- Verify coordinates are integers and within grid bounds
- Use `action="wait"` when resources are exhausted

#### 4. Grid Format Confusion

**Problem:** How to access grid cells?

**Solution:**
```python
# Convert flat array to 2D
grid_2d = np.array(obs.grid).reshape(obs.height, obs.width)

# Access cell at (x, y)
cell_value = grid_2d[y][x]

# Or use flat index
index = y * obs.width + x
cell_value = obs.grid[index]
```

#### 5. Docker Build Failures

**Problem:** `failed to solve: openenv-base:latest`

**Solution:**
```bash
# Build base image first
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .

# Then build wildfire image
docker build -t wildfire-env:latest -f envs/wildfire_env/server/Dockerfile .
```

### Debugging Tips

1. **Enable verbose logging:**
   ```bash
   docker logs -f wildfire-env-container
   ```

2. **Check environment state:**
   ```python
   state = env.state
   print(f"State: {state}")
   ```

3. **Validate actions:**
   ```python
   obs = env.reset().observation
   print(f"Bounds: 0 <= x < {obs.width}, 0 <= y < {obs.height}")
   print(f"Resources: Water={obs.remaining_water}, Breaks={obs.remaining_breaks}")
   ```

4. **Monitor grid changes:**
   ```python
   prev_grid = obs.grid.copy()
   result = env.step(action)
   new_grid = result.observation.grid
   changes = [i for i, (a, b) in enumerate(zip(prev_grid, new_grid)) if a != b]
   print(f"Changed cells: {len(changes)}")
   ```

---

## 📊 Performance Considerations

### Grid Size Impact

- **Small grids (16×16)**: Fast, good for quick testing
- **Medium grids (32×32)**: Default, balanced performance
- **Large grids (64×64+)**: Slower, more realistic but requires more compute

### Resource Limits

- **Low water (4-8)**: Forces strategic decisions
- **High water (20+)**: More forgiving, easier to succeed
- **Low breaks (25)**: Emphasizes firebreak placement strategy
- **High breaks (100+)**: More freedom, less constraint

### Episode Length

- **Short episodes (50 steps)**: Fast iteration, good for debugging
- **Medium episodes (128 steps)**: Default, balanced
- **Long episodes (200+ steps)**: Better for complex strategies

---

## 🧭 References

### Papers & Research

- **Rothermel Model**: [USDA Forest Service - Surface Fire Spread Model](https://www.fs.fed.us/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf)
- **SimFire**: [MITRE Fireline Project](https://github.com/mitrefireline/simfire)
- **RL for Wildfires**: [arXiv:2311.15925](https://arxiv.org/abs/2311.15925)

### OpenEnv Framework

- **Main Repository**: [OpenEnv GitHub](https://github.com/openenv)
- **Documentation**: See `rfcs/` directory for design documents
- **Other Environments**: See `src/envs/` for more environment examples

### Related Tools

- **FastAPI**: [FastAPI Documentation](https://fastapi.tiangolo.com/)
- **Reinforcement Learning**: [Spinning Up in Deep RL](https://spinningup.openai.com/)
- **Docker**: [Docker Documentation](https://docs.docker.com/)

---

## 📝 License

This environment is part of the OpenEnv project. See the main LICENSE file for details.

---

## 🤝 Contributing

Contributions welcome! Please see `CONTRIBUTING.md` in the main OpenEnv repository.

---

## 🔖 Citations

```bibtex
@techreport{rothermel2022surface,
  title     = {The Rothermel Surface Fire Spread Model and Associated Developments},
  author    = {Andrews, Patricia L. and Rothermel, Richard C.},
  year      = {2022},
  institution = {USDA Forest Service},
  number    = {RMRS-GTR-371},
  url       = {https://www.fs.usda.gov/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf}
}

@article{tapley2023reinforcement,
  title   = {Reinforcement Learning for Wildfire Mitigation in Simulated Disaster Environments},
  author  = {Tapley, A. and Dotter, M. and Doyle, M. and others},
  journal = {arXiv preprint arXiv:2311.15925},
  year    = {2023},
  url     = {https://arxiv.org/abs/2311.15925}
}

@misc{mitrefireline2023simfire,
  author = {{MITRE Fireline Project}},
  title  = {SimFire: Wildfire Simulator for Decision-Support and AI Research},
  year   = {2023},
  howpublished = {\url{https://github.com/mitrefireline/simfire}}
}

@misc{wildfire-openenv-2025,
  title  = {Wildfire Environment for OpenEnv: Containment-Focused RL Simulation},
  author = {OpenEnv Contributors},
  year   = {2025},
  url    = {https://github.com/openenv/openenv}
}
```

---

**Happy firefighting! 🔥🚒**
