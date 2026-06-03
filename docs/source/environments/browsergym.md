<!-- openenv-source: browsergym_env -->
# BrowserGym Environment

BrowserGym is a unified framework for web-based agent tasks that provides access to multiple benchmarks under a single Gymnasium-compatible API. This integration brings the complete training-to-evaluation pipeline for web agents into OpenEnv.

## Why BrowserGym?

BrowserGym provides a complete pipeline for developing web agents: train on simple tasks, then evaluate on realistic websites.

**What are these benchmarks?**

- **MiniWoB++ (Training)**: 100+ synthetic web tasks like "click this button", "fill out this form", "select from dropdown". Each task is a simple webpage with a clear objective. Fast resets, randomized variations, dense rewards. Perfect for learning basic web navigation skills. **No external setup needed** - tasks run in isolated browser sessions.

- **WebArena (Evaluation)**: 812 tasks on real websites (e-commerce, forums, GitLab, Wikipedia). Tasks like "find the cheapest laptop and add to cart" or "create a merge request for bug #123". Multistep, requires reasoning, sparse rewards. Tests if your agent can handle actual websites. **Requires running 7 backend services** (shopping site, GitLab instance, etc.).

- **VisualWebArena**: Similar to WebArena but requires visual understanding - agents need to interpret images, identify UI elements visually, handle multimodal content.

- **WorkArena**: Enterprise software tasks (CRM, project management, business workflows). Tests automation on corporate-style applications.

**The training → evaluation pipeline:**
1. Train on MiniWoB (simple, controlled, fast iterations)
2. Evaluate on WebArena (complex, realistic, measures real-world capability)

**Key advantage**: You can start training immediately with MiniWoB. No need to set up infrastructure just to test if your code works.

## Quick Start - Training (MiniWoB)

### No Setup Required! 🎉

```python
from browsergym_env import BrowserGymEnv, BrowserGymAction

# Create environment for MiniWoB training task
env = BrowserGymEnv.from_docker_image(
    "ghcr.io/openenv/browsergym-env:latest",
    environment={
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_TASK_NAME": "click-test",  # or "click-button", "click-dialog", etc.
    }
)

# Train your agent!
for episode in range(1000):
    result = env.reset()
    print(f"Goal: {result.observation.goal}")

    done = False
    while not done:
        # Your agent decides what to do
        action_str = agent.get_action(result.observation.text)
        action = BrowserGymAction(action_str=action_str)

        result = env.step(action)
        done = result.done

        print(f"Reward: {result.reward}")

env.close()
```

## Harness Sessions for TRL

If you want BrowserGym to participate in a tool-driven harness instead of a
hand-written `env.reset()` / `env.step()` loop, use the BrowserGym session
factory:

```python
from browsergym_env import BrowserGymEnv
from browsergym_env.harness import BrowserGymSessionFactory
from openenv.core.harness import (
    HarnessRunLimits,
    MCPHarnessAdapter,
    build_harness_rollout_func,
)

session_factory = BrowserGymSessionFactory(
    client_factory=lambda: BrowserGymEnv(base_url="https://openenv-browsergym-env.hf.space"),
)

rollout_func = build_harness_rollout_func(
    session_factory=session_factory,
    harness_adapter=MCPHarnessAdapter(),
    model_step_builder=...,  # trainer-owned model sampling
    limits=HarnessRunLimits(max_turns=10),
)
```

BrowserGym exposes `click`, `fill`, `send_keys`, `scroll`, and `noop` as MCP-style
tools while still translating them back into the underlying `BrowserGymAction`
strings. See [examples/browsergym_harness.py](https://github.com/meta-pytorch/OpenEnv/blob/main/examples/browsergym_harness.py)
for a full TRL-oriented example.

### Available Tasks by Benchmark

#### MiniWoB++ Tasks (Training - 100+ tasks)

MiniWoB tasks are organized by difficulty and type. Here are the main categories:

**Click Tasks** (Basic interaction)

| Task Name | Description | Difficulty |
|-----------|-------------|------------|
| `click-test` | Click a single button | ⭐ Easy |
| `click-button` | Click button with specific text | ⭐ Easy |
| `click-button-sequence` | Click buttons in order | ⭐⭐ Medium |
| `click-checkboxes` | Select specific checkboxes | ⭐⭐ Medium |
| `click-checkboxes-soft` | Select checkboxes (multiple valid) | ⭐⭐ Medium |
| `click-checkboxes-large` | Many checkboxes to select from | ⭐⭐ Medium |
| `click-checkboxes-transfer` | Transfer learning variation | ⭐⭐ Medium |
| `click-dialog` | Click correct button in dialog | ⭐ Easy |
| `click-dialog-2` | More complex dialog | ⭐⭐ Medium |
| `click-link` | Click on a link | ⭐ Easy |
| `click-option` | Select from dropdown | ⭐⭐ Medium |
| `click-pie` | Click on pie chart slice | ⭐⭐ Medium |
| `click-scroll-list` | Click item in scrollable list | ⭐⭐⭐ Hard |
| `click-shades` | Click on specific color shade | ⭐⭐ Medium |
| `click-shape` | Click on specific shape | ⭐⭐ Medium |
| `click-tab` | Switch between tabs | ⭐⭐ Medium |
| `click-tab-2` | More complex tab switching | ⭐⭐⭐ Hard |
| `click-widget` | Click on UI widget | ⭐⭐ Medium |

**Text Entry Tasks** (Typing and forms)

| Task Name | Description | Difficulty |
|-----------|-------------|------------|
| `enter-text` | Type text into input field | ⭐ Easy |
| `enter-text-dynamic` | Dynamic text entry | ⭐⭐ Medium |
| `enter-text-2` | Multiple text fields | ⭐⭐ Medium |
| `enter-password` | Fill password field | ⭐ Easy |
| `enter-date` | Enter a date | ⭐⭐ Medium |
| `enter-time` | Enter a time | ⭐⭐ Medium |
| `login-user` | Complete login form | ⭐⭐ Medium |
| `login-user-popup` | Login via popup | ⭐⭐⭐ Hard |

**Navigation Tasks** (Multi-step interaction)

| Task Name | Description | Difficulty |
|-----------|-------------|------------|
| `navigate-tree` | Navigate through tree structure | ⭐⭐⭐ Hard |
| `search-engine` | Use search interface | ⭐⭐ Medium |
| `use-autocomplete` | Interact with autocomplete | ⭐⭐⭐ Hard |
| `book-flight` | Book a flight (complex form) | ⭐⭐⭐⭐ Very Hard |
| `choose-date` | Pick date from calendar | ⭐⭐⭐ Hard |
| `choose-date-easy` | Simplified date picker | ⭐⭐ Medium |
| `choose-date-medium` | Medium difficulty date picker | ⭐⭐⭐ Hard |
| `choose-list` | Select from long list | ⭐⭐ Medium |

**Visual/Spatial Tasks** (Requires visual understanding)

| Task Name | Description | Difficulty |
|-----------|-------------|------------|
| `count-sides` | Count sides of shape | ⭐⭐ Medium |
| `count-shape` | Count specific shapes | ⭐⭐ Medium |
| `find-word` | Find word in text | ⭐⭐ Medium |
| `focus-text` | Focus on text element | ⭐ Easy |
| `focus-text-2` | More complex focus task | ⭐⭐ Medium |
| `grid-coordinate` | Click grid coordinate | ⭐⭐ Medium |
| `guess-number` | Guess a number game | ⭐⭐⭐ Hard |
| `identify-shape` | Identify shape type | ⭐⭐ Medium |
| `read-table` | Extract info from table | ⭐⭐⭐ Hard |
| `read-table-2` | More complex table reading | ⭐⭐⭐ Hard |

**Email/Social Tasks** (Realistic scenarios)

| Task Name | Description | Difficulty |
|-----------|-------------|------------|
| `email-inbox` | Manage email inbox | ⭐⭐⭐⭐ Very Hard |
| `email-inbox-forward` | Forward emails | ⭐⭐⭐⭐ Very Hard |
| `email-inbox-nl` | Natural language email task | ⭐⭐⭐⭐ Very Hard |
| `email-inbox-star-reply` | Star and reply to emails | ⭐⭐⭐⭐ Very Hard |
| `social-media` | Social media interaction | ⭐⭐⭐⭐ Very Hard |
| `social-media-some` | Partial social media task | ⭐⭐⭐ Hard |

**Total:** 100+ tasks across all categories

**Usage:**
```python
# Easy task for quick testing
env = BrowserGymEnv(environment={"BROWSERGYM_TASK_NAME": "click-test"})

# Medium difficulty for training
env = BrowserGymEnv(environment={"BROWSERGYM_TASK_NAME": "click-checkboxes"})

# Hard task for evaluation
env = BrowserGymEnv(environment={"BROWSERGYM_TASK_NAME": "email-inbox"})
```

#### WebArena Tasks (Evaluation - 812 tasks)

WebArena tasks are organized by website and difficulty. Tasks are numbered 0-811.

**By Website:**

| Website | Task Count | Description | Example Tasks |
|---------|------------|-------------|---------------|
| Shopping | ~200 | E-commerce site | Search products, add to cart, checkout |
| Shopping Admin | ~150 | Admin panel | Manage products, orders, customers |
| Reddit | ~150 | Forum/social | Post, comment, search discussions |
| GitLab | ~200 | Code repository | Create issues, merge requests, review code |
| Wikipedia | ~100 | Knowledge base | Search, read, extract information |
| Map | ~12 | Location service | Find places, get directions |

**By Difficulty:**

| Difficulty | Task Count | Steps Required | Example |
|------------|------------|----------------|---------|
| Easy | ~200 | 1-5 steps | "Find the price of product X" |
| Medium | ~400 | 5-15 steps | "Add cheapest laptop to cart" |
| Hard | ~212 | 15+ steps | "Create merge request for bug fix" |

**Usage:**

```python
# Task 0 (usually easy)
env = BrowserGymEnv(environment={
    "BROWSERGYM_BENCHMARK": "webarena",
    "BROWSERGYM_TASK_NAME": "0",
    "SHOPPING": "http://your-server:7770",
    # ... other URLs
})

# Task 156 (GitLab merge request)
env = BrowserGymEnv(environment={
    "BROWSERGYM_BENCHMARK": "webarena",
    "BROWSERGYM_TASK_NAME": "156",
    # ... URLs
})
```

**Note:** WebArena tasks require the full backend infrastructure. See [WebArena setup guide](https://github.com/web-arena-x/webarena/tree/main/environment_docker).

#### VisualWebArena Tasks (910 tasks)

Similar to WebArena but requires visual understanding. Tasks involve:
- Image-based reasoning
- Visual element identification
- Multimodal interaction (text + images)

#### WorkArena Tasks

Enterprise software automation tasks:
- CRM operations
- Project management
- Business workflows

**Full task lists:**
- [MiniWoB++ tasks](https://github.com/Farama-Foundation/miniwob-plusplus/tree/master/miniwob/environment)
- [WebArena tasks](https://github.com/web-arena-x/webarena/blob/main/config_files/)
- [BrowserGym documentation](https://github.com/ServiceNow/BrowserGym)

## Evaluation (WebArena)

### Prerequisites

WebArena requires setting up backend infrastructure. See the [WebArena documentation](https://github.com/web-arena-x/webarena/tree/main/environment_docker).

### Usage

```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

# Create environment for WebArena evaluation
env = BrowserGymEnv.from_docker_image(
    "ghcr.io/openenv/browsergym-env:latest",
    environment={
        "BROWSERGYM_BENCHMARK": "webarena",
        "BROWSERGYM_TASK_NAME": "0",  # Task ID
        # WebArena backend URLs (required)
        "SHOPPING": "http://your-server:7770",
        "SHOPPING_ADMIN": "http://your-server:7780/admin",
        "REDDIT": "http://your-server:9999",
        "GITLAB": "http://your-server:8023",
        "MAP": "http://your-server:3000",
        "WIKIPEDIA": "http://your-server:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
        "HOMEPAGE": "http://your-server:4399",
    }
)

# Evaluate your trained agent
result = env.reset()
while not result.done:
    action_str = agent.get_action(result.observation)
    action = BrowserGymAction(action_str=action_str)
    result = env.step(action)

print(f"Success: {result.reward}")
env.close()
```

## Building the Docker Image

### Prerequisites

1. **Base Image**: Build the OpenEnv base image first:

```bash
# From the OpenEnv repository root
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .
```

### Build the BrowserGym Environment

```bash
# From the browsergym_env directory
cd envs/browsergym_env
docker build -t browsergym-env:latest -f server/Dockerfile .
```

### Run the Server

#### For MiniWoB (Training):

```bash
docker run -p 8000:8000 \
  -e BROWSERGYM_BENCHMARK="miniwob" \
  -e BROWSERGYM_TASK_NAME="click-test" \
  browsergym-env:latest
```

#### For WebArena (Evaluation):

```bash
docker run -p 8000:8000 \
  -e BROWSERGYM_BENCHMARK="webarena" \
  -e BROWSERGYM_TASK_NAME="0" \
  -e SHOPPING="http://your-server:7770" \
  -e SHOPPING_ADMIN="http://your-server:7780/admin" \
  -e REDDIT="http://your-server:9999" \
  -e GITLAB="http://your-server:8023" \
  -e MAP="http://your-server:3000" \
  -e WIKIPEDIA="http://your-server:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing" \
  -e HOMEPAGE="http://your-server:4399" \
  browsergym-env:latest
```

## Environment Details

### Action

Actions in BrowserGym are natural language strings that describe browser operations:

```python
from envs.browsergym_env import BrowserGymAction

# Click actions
action = BrowserGymAction(action_str="click('Submit button')")
action = BrowserGymAction(action_str="click('element_id_123')")

# Type actions
action = BrowserGymAction(action_str="fill('username', 'john@example.com')")
action = BrowserGymAction(action_str="fill('password', 'secret123')")

# Navigate actions
action = BrowserGymAction(action_str="goto('https://example.com')")

# Keyboard actions
action = BrowserGymAction(action_str="press('Enter')")
action = BrowserGymAction(action_str="press('Tab')")

# Scroll actions
action = BrowserGymAction(action_str="scroll('down')")
```

### Observation

Observations contain multiple modalities:

```python
result = env.step(action)
obs = result.observation

# Text observations
print(obs.text)          # Primary text representation (AXTree or DOM)
print(obs.axtree_txt)    # Accessibility tree
print(obs.pruned_html)   # Pruned HTML (interactive elements only)

# Page metadata
print(obs.url)           # Current URL
print(obs.goal)          # Task goal/instruction

# Visual (if enabled)
if obs.screenshot is not None:
    print(obs.screenshot.shape)  # [height, width, channels]

# Error handling
if obs.last_action_error:
    print(f"Action failed: {obs.error}")

# Episode status
print(obs.done)          # True if episode ended
print(obs.reward)        # Reward for the step

# Access full BrowserGym data (includes timestamps, etc.)
print(obs.metadata["browsergym_obs"])  # Full observation dict from BrowserGym
print(obs.metadata["browsergym_info"]) # Full info dict (timestamps, page state, etc.)
```

#### Advanced: Accessing Raw BrowserGym Data

For VisualWebArena or custom training, you may need additional data like timestamps or browser state. The full BrowserGym observation and info dicts are preserved in `metadata`:

```python
result = env.step(action)

# Access timestamps (if available)
info = result.observation.metadata["browsergym_info"]
if "timestamp" in info:
    print(f"Action timestamp: {info['timestamp']}")

# Access additional observation fields
obs_dict = result.observation.metadata["browsergym_obs"]
if "dom_object" in obs_dict:
    dom = obs_dict["dom_object"]
    # Work with raw DOM object

# Access page performance data
if "performance" in info:
    print(f"Page load time: {info['performance']}")
```

### State

The environment state tracks progress:

```python
state = env.state()

print(f"Benchmark: {state.benchmark}")     # 'miniwob', 'webarena', etc.
print(f"Task: {state.task_name}")          # Task name/ID
print(f"Episode: {state.episode_id}")      # Unique episode ID
print(f"Steps: {state.step_count}")        # Number of steps taken
print(f"Total Reward: {state.cum_reward}") # Cumulative reward
print(f"Goal: {state.goal}")               # Task instruction
print(f"URL: {state.current_url}")         # Current page URL
```

## Configuration

Environment variables:

### Common Settings
- `BROWSERGYM_BENCHMARK`: Benchmark to use (`miniwob`, `webarena`, `visualwebarena`, `workarena`)
- `BROWSERGYM_TASK_NAME`: Specific task name (optional, will use first available if not set)
- `BROWSERGYM_HEADLESS`: Run browser in headless mode (default: `true`)
- `BROWSERGYM_VIEWPORT_WIDTH`: Browser viewport width (default: `1280`)
- `BROWSERGYM_VIEWPORT_HEIGHT`: Browser viewport height (default: `720`)
- `BROWSERGYM_TIMEOUT`: Action timeout in milliseconds (default: `10000`)

### WebArena-Specific (only needed for WebArena benchmark)
- `SHOPPING`: Shopping website URL
- `SHOPPING_ADMIN`: Shopping admin panel URL
- `REDDIT`: Reddit-like forum URL
- `GITLAB`: GitLab instance URL
- `MAP`: Map service URL
- `WIKIPEDIA`: Wikipedia instance URL
- `HOMEPAGE`: Homepage URL

## Supported Benchmarks

### 1. MiniWoB++ (Training) ✅ Recommended for Training

- **100+ tasks** ranging from simple (click buttons) to complex (form filling, navigation)
- **Fast**: Instant resets, quick episodes
- **Randomized**: Task variations for generalization
- **No setup**: Works out-of-the-box
- **Dense rewards**: Immediate feedback for learning

**Use Case**: Train agents on fundamental web navigation skills

### 2. WebArena (Evaluation) 📊 Benchmark

- **812 realistic tasks** across 6 websites
- **Complex**: Multi-step reasoning, real web interfaces
- **Requires setup**: Need to run 7 backend services
- **Sparse rewards**: Binary success/failure
- **Evaluation-focused**: Test real-world performance

**Use Case**: Evaluate agents on realistic web tasks

### 3. VisualWebArena (Evaluation) 👁️ Visual Benchmark

- **910 tasks** requiring visual understanding
- **Multimodal**: Both text and visual observations
- **Requires setup**: Similar to WebArena
- **Challenging**: Requires visual reasoning

**Use Case**: Test visual web navigation capabilities

### 4. WorkArena (Evaluation) 💼 Enterprise Benchmark

- **Enterprise tasks**: CRM, project management, etc.
- **Realistic workflows**: Real enterprise software
- **Requires setup**: Enterprise software instances

**Use Case**: Evaluate on business automation tasks

## Typical Training Pipeline

```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

# Stage 1: Train on MiniWoB (simple tasks, fast)
train_env = BrowserGymEnv.from_docker_image(
    "browsergym-env:latest",
    environment={
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_TASK_NAME": "click-button",
    }
)

# Train your agent (RL, imitation learning, etc.)
agent.train(train_env, num_episodes=10000)
train_env.close()

# Stage 2: Evaluate on WebArena (complex tasks, realistic)
eval_env = BrowserGymEnv.from_docker_image(
    "browsergym-env:latest",
    environment={
        "BROWSERGYM_BENCHMARK": "webarena",
        "BROWSERGYM_TASK_NAME": "0",
        # ... WebArena URLs
    }
)

# Test performance
success_rate = agent.evaluate(eval_env, num_tasks=812)
print(f"WebArena Success Rate: {success_rate:.2%}")
eval_env.close()
```

## Development & Testing

### Running Tests

```bash
# From the OpenEnv repository root
pytest tests/envs/test_browsergym_env.py
```

### Local Development

```bash
# Install in development mode
cd /path/to/OpenEnv
pip install -e .

# Install BrowserGym
pip install browsergym browsergym-miniwob browsergym-webarena

# Run the server locally
cd envs/browsergym_env/server
export BROWSERGYM_BENCHMARK=miniwob
export BROWSERGYM_TASK_NAME=click-test
python app.py
```

## Project Structure

```
browsergym_env/
├── __init__.py              # Module exports
├── models.py                # Action, Observation, State dataclasses
├── client.py                # HTTPEnvClient implementation
├── README.md                # This file
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI application
    ├── browsergym_environment.py  # Environment implementation
    ├── Dockerfile           # Container specification
    └── requirements.txt     # Python dependencies
```

## References

- [BrowserGym GitHub](https://github.com/ServiceNow/BrowserGym)
- [MiniWoB++ Paper](https://arxiv.org/abs/1802.08802)
- [WebArena Paper](https://arxiv.org/abs/2307.13854)
- [WebArena Website](https://webarena.dev/)
- [VisualWebArena Paper](https://jykoh.com/vwa)
- [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv)
