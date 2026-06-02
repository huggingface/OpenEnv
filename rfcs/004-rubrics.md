# RFC 004: Rubric System for Reward Computation

## Summary

This RFC introduces the Rubric system—a composable, nn.Module-inspired abstraction for computing rewards in OpenEnv environments. Environment authors implement `__init__` and `forward(action, observation) -> float`; the framework handles composition, observability, and parallel evaluation.

---

## Motivation

**Problem 1: No standard reward interface**

Currently, environment authors must roll their own reward computation. This leads to:
- Inconsistent APIs across environments
- No reusability of common patterns (gating, weighted sums)
- Training frameworks can't introspect reward components

**Problem 2: Multi-criteria evaluation is hard**

Modern RL requires evaluating outputs on multiple dimensions (correctness, style, safety). Without a composition abstraction:
- Authors manually aggregate scores with ad-hoc logic
- No standard patterns for hierarchical gating ("must compile before style matters")
- No visibility into which criteria caused low rewards

**Problem 3: Parallel evaluation for I/O-bound rubrics**

LLM judges and sandboxed execution are slow (100-1000ms). Without async support:
- Batch evaluation becomes a bottleneck
- Training throughput suffers

---

## Proposed Solution

A **Rubric** is a callable that computes a reward:

```python
class Rubric:
    def forward(self, action, observation) -> float:
        """Implement this. Return 0.0-1.0."""
        raise NotImplementedError
```

The API is modeled after PyTorch's `nn.Module`:
- Users implement `__init__` and `forward`
- Child rubrics auto-register when assigned as attributes
- Hooks provide observability without polluting the base class
- `state_dict()` / `load_state_dict()` for serialization

### Rubrics Live Inside Environments

Rubrics are **server-side only**. Each environment defines its rubric in `__init__`, and the rubric executes during `step()`. Access via `env.rubric`:

```python
class CodeEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.rubric = CodeRubric()  # Required attribute

    def step(self, action: CodeAction) -> CodeObservation:
        # ... execute action ...
        reward = self.rubric(action, observation)
        return observation.with_reward(reward)
```

Training infrastructure accesses rubrics for introspection:

```python
# Log all component scores
for name, r in env.rubric.named_rubrics():
    print(f"{name}: {r.last_score}")
```

**Important**: The `Environment` base class requires a `rubric` attribute. All environments must define `self.rubric` in their constructor.

---

## Usage Examples

### Basic Rubric

A leaf rubric implements a single evaluation criterion:

```python
class Compiles(Rubric):
    def forward(self, action, observation) -> float:
        try:
            ast.parse(action.content)
            return 1.0
        except SyntaxError:
            return 0.0
```

### Composite Rubrics

The nn.Module-like design enables hierarchical composition, a pattern we see across recent academic work on rubric-based rewards. Rubicon (Huang et al., 2024) uses multi-dimensional rubrics with gating; AdvancedIF (He et al., 2024) combines multiple criteria with all-or-nothing aggregation; RLTF (Liu et al., 2023) applies hierarchical evaluation with fail-fast semantics.

In our design, child rubrics are automatically registered when assigned as attributes, enabling patterns like:

```python
class CodeRubric(Rubric):
    def __init__(self, test_weight: float = 0.7):
        super().__init__()
        self.test_weight = test_weight
        self.compiles = Compiles()      # Auto-registered
        self.tests = PassesTests()
        self.style = LLMStyleJudge()

    def forward(self, action, observation) -> float:
        if self.compiles(action, observation) < 1.0:
            return 0.0  # Gate: must compile
        t = self.tests(action, observation)
        s = self.style(action, observation)
        return t * self.test_weight + s * (1 - self.test_weight)
```

The framework can traverse the rubric hierarchy via `named_rubrics()`, enabling automatic logging of all component scores.

### Container Rubrics

The core Rubric spec is just `forward(action, observation) -> float`. Together with the spec, we provide container rubrics for common aggregation patterns, similar to how PyTorch provides `nn.Sequential` alongside `nn.Module`:

- **Sequential**: Runs child rubrics in order. If any returns 0, stops immediately and returns 0 (fail-fast). This implements the hierarchical gating pattern from RLTF where syntax checks run before execution checks.

- **Gate**: Wraps a rubric with a threshold. Returns the child's score if it meets the threshold, otherwise returns 0. Useful for hard constraints ("must pass 50% of tests").

- **WeightedSum**: Computes a weighted combination of child rubrics. The standard aggregation pattern, though research (Gonzalez et al., 2024) suggests holistic judging sometimes outperforms point-summing.

- **RubricList**: A container for dynamic lists of rubrics, analogous to `nn.ModuleList`. Does not define aggregation—use within a parent rubric that implements custom logic.

- **RubricDict**: A container for named rubrics, analogous to `nn.ModuleDict`. Enables keyed access for multi-task environments where different tasks require different rubrics:

  ```python
  class AtariRubric(Rubric):
      def __init__(self):
          super().__init__()
          self.games = RubricDict({
              "pong": PongRubric(),
              "breakout": BreakoutRubric(),
              "space_invaders": SpaceInvadersRubric(),
          })

      def forward(self, action, obs) -> float:
          return self.games[obs.game_id](action, obs)
  ```

  Access: `env.rubric.games["pong"]`

- **LLMJudge**: A rubric that calls an LLM endpoint to evaluate the response. Takes a prompt template and endpoint configuration. The LLM call happens via the configured MCP service, keeping the rubric portable across deployments.

These containers compose naturally:

```python
# Hierarchical gating: compile → pass tests → score style
rubric = Sequential(
    Gate(Compiles()),
    Gate(PassesTests(), threshold=0.5),
    WeightedSum([PassesTests(), StyleRubric()], weights=[0.7, 0.3])
)
```

### Observability via Hooks

Just like `nn.Module`, we support pre- and post-forward hooks. Hooks are called before and after `forward()` executes, without modifying the rubric's return value. This is particularly useful for observability—training systems can install hooks to capture every component score for logging and debugging, without rubric authors needing to do anything special.

```python
scores = {}
for name, r in rubric.named_rubrics():
    r.register_forward_hook(
        lambda r, a, o, result, n=name: scores.__setitem__(n, result)
    )

reward = rubric(action, observation)
wandb.log(scores)  # {"compiles": 1.0, "tests": 0.8, "style": 0.7}
```

Hooks also enable extensions like caching, timing, or validation without modifying the base class.

### Batch Evaluation via Environment Stacking

**Key architectural decision**: Environments do not support multiplexed trajectories. One environment instance handles one trajectory. To generate batches, stack multiple environment instances.

Rubric authors write synchronous `forward()` methods—no async knowledge required. For batch evaluation, the training infrastructure manages a pool of environments:

```python
async def train():
    # Stack of 64 environments, each with its own rubric
    envs = EnvPool("code_env", n=64)

    for batch in dataloader:
        actions = await model.generate(batch.prompts)
        # EnvPool orchestrates calls across all environments
        observations = await envs.step_batch(actions)
        rewards = [obs.reward for obs in observations]
        loss = compute_loss(rewards)
```

Each environment computes its own reward via `self.rubric(action, observation)` during `step()`. The `EnvPool` helper orchestrates parallel execution across the stack—training code never instantiates rubrics directly.

For single-environment scripts, just call `env.step(action)` synchronously. The reward is computed inside the environment and returned in the observation.

---

## What Changes

| Component | Change |
|-----------|--------|
| New `Rubric` base class | `forward(action, observation) -> float`, auto-registration, hooks, `get_rubric()` |
| New containers | `Sequential`, `Gate`, `WeightedSum`, `RubricList`, `RubricDict`, `LLMJudge` |
| `Environment` base class | Requires `rubric: Rubric` attribute; `step()` uses rubric for reward |
| New `EnvPool` helper | Orchestrates batch evaluation across stacked environments |

### Environment Base Class Changes

The `Environment` base class gains a required `rubric` attribute:

```python
class Environment(Generic[ActT, ObsT, StateT]):
    rubric: Rubric  # Required - must be set in __init__

    def step(self, action: ActT) -> ObsT:
        # Implementation calls self.rubric(action, observation)
        # Reward returned in observation
        ...
```

All environments must define `self.rubric` in their constructor. The framework validates this at environment startup.

---

## API Reference

### Rubric Base Class

| Method | Description |
|--------|-------------|
| `forward(action, observation) -> float` | Implement this. Can be sync or async. Compute reward. |
| `__call__(action, observation)` | Evaluation with hooks. Auto-detects async `forward()` and awaits. |
| `register_forward_hook(fn)` | Called after `forward()`. Can be sync or async. |
| `register_forward_pre_hook(fn)` | Called before `forward()`. Can be sync or async. |
| `children()` / `named_children()` | Iterate immediate child rubrics. |
| `rubrics()` / `named_rubrics()` | Iterate all descendant rubrics. |
| `get_rubric(path: str)` | Access nested rubric by dot-separated path (e.g., `"code.syntax"`). |
| `reset()` | Reset any internal state (sync). Override in subclasses. |
| `state_dict()` / `load_state_dict(d)` | Serialize/deserialize configuration. |

**Note on async support**: Rubrics support both sync and async `forward()` implementations. The base class auto-detects which variant is used and handles both. Sync rubrics remain fully backward compatible. Async rubrics can make LLM calls, use MCP tools, or perform I/O without blocking.

### Container Rubrics

| Container | Behavior |
|-----------|----------|
| `Sequential(*rubrics)` | Run in order; stop and return 0 if any returns 0 (sequential await for async) |
| `Gate(rubric, threshold=1.0)` | Return 0 if score < threshold (awaits async child) |
| `WeightedSum(rubrics, weights)` | Weighted combination (parallel await via `asyncio.gather()` for async) |
| `RubricList(rubrics)` | Container for dynamic lists (no aggregation) |
| `RubricDict(rubrics: Dict[str, Rubric])` | Container for named rubrics with keyed access |
| `LLMJudge(prompt_template, endpoint)` | Call LLM via MCP for evaluation |

**Note on async containers**: All containers auto-detect async children and handle them appropriately. `Sequential` and `Gate` await children sequentially (preserving fail-fast semantics). `WeightedSum` executes all children in parallel using `asyncio.gather()` for maximum throughput.

---

## Alternatives Considered

**Two separate classes (Criterion + Rubric)**

We considered having `Criterion` for leaf evaluators and `Rubric` for composites. This mirrors some academic terminology. However, the distinction adds complexity without benefit—a rubric with no children is just a leaf. PyTorch's `nn.Module` doesn't distinguish between "layer" and "model"; neither do we.

**Declarative YAML/JSON schema**

A declarative format would enable non-programmers to define rubrics and allow static validation. However, interesting rubrics involve LLM calls, sandboxed execution, database queries, and complex branching logic. These don't fit declarative formats well. Python code is more expressive and debuggable. The ecosystem (TRL, OpenRLHF, veRL) has converged on code-based rewards.

**Sync forward() with async wrapper (original approach)**

We initially considered keeping `forward()` sync with an async `evaluate()` wrapper. However, with the async-first client API (PR #343), we chose to make rubrics async-compatible by default:

- `forward()` can be either sync or async—the base class detects which and handles both
- Sync rubrics still work unchanged (backward compatible)
- Async rubrics (e.g., LLM judges, MCP tool calls) are now natural
- Container rubrics like `WeightedSum` automatically parallelize async children using `asyncio.gather()`
- The sync `evaluate()` wrapper is no longer needed since the async path is built into `__call__()`

**Magic proxy types for automatic parallelism**

We prototyped a design where `rubric()` returns a proxy object that resolves lazily, enabling automatic parallelism like PyTorch's CUDA tensors. This proved fragile: overriding `__add__`, `__lt__`, and other operators led to confusing edge cases. Explicit `evaluate_batch()` is clearer and sufficient for the primary use case (batch evaluation in training).

---

## Open Questions

**How do rubrics access external services (LLM endpoints, databases)?**

LLM judges and other rubrics need to call external services. For now, rubrics can simply make RPC calls inside their `forward()` as needed—there are no restrictions on what a rubric can do internally. A future RFC will address reusable patterns for remote resources (connection pooling, retry logic) and lifecycle management of those connections.

**Error handling in forward()**

Currently, exceptions in `forward()` propagate to the caller. Rubric authors should handle expected errors (network timeouts, invalid inputs) internally. We may add optional error handling in the base class (e.g., return 0.0 on exception with logging) based on user feedback.

**Concurrency limits**

For rubrics that make external API calls (LLM judges, database queries), consider implementing rate limiting or connection pooling inside the rubric. The framework does not impose concurrency limits—async rubrics can execute as many concurrent calls as the event loop allows.

---

## Delayed Rewards

The per-step `Rubric.forward(action, observation) -> float` API handles immediate rewards well, but many environments require delayed rewards where the score depends on future events:

- **Cursor Plan Mode**: Reward for writing a plan depends on later execution success
- **Codenames**: Spymaster's clue quality depends on Operative's subsequent guesses
- **Chess**: Win/loss only known at game end, needs discounting back to earlier moves

### Self-Accumulating TrajectoryRubric

Since OpenEnv doesn't batch (one env = one trajectory), the rubric itself can accumulate the trajectory internally. No separate trajectory buffer is needed.

**The pattern**:
1. `TrajectoryRubric.__call__(action, obs)` records step internally
2. Returns `0.0` (or configurable intermediate reward) until `obs.done=True`
3. On done, computes final score from accumulated trajectory
4. `reset()` clears the internal buffer
5. Composes naturally with `Sequential`, `RubricDict`, etc.

### Memory Model: CPU-Only Trajectories

**Constraint**: Trajectories must not consume GPU memory.

**Design**: `TrajectoryRubric` stores observations in CPU memory only. Environments with GPU tensors in observations must detach/move to CPU before returning from `step()`.

**Future extension**: If CPU memory becomes problematic at scale, a reference-based approach could store `(episode_id, step_index)` tuples referencing external replay buffers.

### TrajectoryRubric Base Class

```python
from abc import abstractmethod
from typing import List, Tuple, Any


class TrajectoryRubric(Rubric):
    """Abstract base for rubrics that score based on full trajectories.

    Subclasses implement:
    - score_trajectory(): Compute final score from trajectory
    - compute_step_rewards(): Define credit assignment strategy

    The __call__ method accumulates steps and returns rewards according
    to the subclass's implementation.

    IMPORTANT: Trajectories are stored in CPU memory to avoid GPU pressure.
    Environments with GPU tensors in observations must move them to CPU
    before returning from step().

    Known limitation: Very long episodes (thousands of steps) may consume
    significant CPU memory. For such cases, consider streaming rubrics.
    """

    def __init__(self, intermediate_reward: float = 0.0):
        super().__init__()
        self.intermediate_reward = intermediate_reward
        self._trajectory: List[Tuple[Any, Observation]] = []

    def __call__(self, action, observation) -> float:
        """Accumulate step and return reward.

        Returns intermediate_reward until done, then computes trajectory score.
        """
        self._trajectory.append((action, observation))

        if getattr(observation, 'done', False):
            return self.score_trajectory(self._trajectory)
        else:
            return self.intermediate_reward

    @abstractmethod
    def score_trajectory(self, trajectory: List[Tuple[Any, Observation]]) -> float:
        """Score the complete trajectory. Return 0.0-1.0.

        Called when observation.done=True.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_step_rewards(self) -> List[float]:
        """Compute per-step rewards from the accumulated trajectory.

        Returns: List of rewards, one per step.
        Define your credit assignment strategy here.
        """
        raise NotImplementedError

    def reset(self):
        """Clear accumulated trajectory. Call on env.reset()."""
        self._trajectory = []

    @property
    def trajectory(self) -> List[Tuple[Any, Observation]]:
        """Current trajectory (read-only copy)."""
        return list(self._trajectory)
```

### ExponentialDiscountingTrajectoryRubric

Concrete implementation with standard gamma-based discounting:

```python
class ExponentialDiscountingTrajectoryRubric(TrajectoryRubric):
    """TrajectoryRubric with exponential discounting for credit assignment.

    Per-step reward: r_t = gamma^(T-1-t) * R_final

    With gamma=0.99, later steps get higher reward (they're "closer" to the outcome).
    With gamma=1.0, all steps get equal reward.

    Usage:
        rubric = ChessWinLossRubric(gamma=0.99)
        reward = rubric(action, obs)  # Returns 0.0 until done, then final score
        step_rewards = rubric.compute_step_rewards()  # Get discounted per-step rewards
    """

    def __init__(self, gamma: float = 0.99, intermediate_reward: float = 0.0):
        super().__init__(intermediate_reward=intermediate_reward)
        self.gamma = gamma

    def compute_step_rewards(self) -> List[float]:
        """Apply exponential discounting from final reward."""
        if not self._trajectory:
            return []

        final_score = self.score_trajectory(self._trajectory)
        T = len(self._trajectory)
        return [final_score * (self.gamma ** (T - 1 - t)) for t in range(T)]
```

### Composition with Existing Containers

TrajectoryRubric extends Rubric, so it composes naturally:

```python
# Terminal game with format check first
rubric = Sequential(
    FormatRubric(),           # Per-step: check action format
    ChessWinLossRubric(),     # Trajectory: score at game end
)

# Multi-game environment
rubric = RubricDict({
    "chess": ChessWinLossRubric(gamma=0.99),
    "codenames": CodenamesRubric(gamma=0.95),
})

# Weighted combination of per-step and trajectory
rubric = WeightedSum(
    [ClueQualityRubric(), GameOutcomeRubric()],
    weights=[0.3, 0.7],
)
```

### Example: Chess with Discounting

```python
class ChessWinLossRubric(ExponentialDiscountingTrajectoryRubric):
    """Score chess game based on outcome with temporal discounting."""

    def __init__(self, gamma: float = 0.99):
        super().__init__(gamma=gamma, intermediate_reward=0.0)

    def score_trajectory(self, trajectory: List[Tuple[Any, Observation]]) -> float:
        """Score based on game outcome."""
        if not trajectory:
            return 0.0

        _, final_obs = trajectory[-1]  # Unpack (action, observation)
        outcome = getattr(final_obs, 'metadata', {}).get('winner')

        if outcome == 'agent':
            return 1.0
        elif outcome == 'opponent':
            return 0.0
        else:
            return 0.5  # Draw
```

### Example: Cursor Plan Mode

```python
class PlanExecutionRubric(TrajectoryRubric):
    """Score based on whether plan led to successful execution.

    Custom credit assignment: full reward goes to the plan step,
    zero to execution steps (since execution just follows the plan).
    """

    def __init__(self):
        super().__init__(intermediate_reward=0.0)

    def score_trajectory(self, trajectory: List[Tuple[Any, Observation]]) -> float:
        """Score = execution success rate."""
        if len(trajectory) < 2:
            return 0.0

        # Find execution steps after plan
        plan_idx = None
        for i, (action, obs) in enumerate(trajectory):
            if getattr(action, 'metadata', {}).get('type') == 'plan':
                plan_idx = i

        if plan_idx is None:
            return 0.0

        # Score based on test results after plan
        execution_steps = trajectory[plan_idx + 1:]
        if not execution_steps:
            return 0.0

        _, final_obs = execution_steps[-1]
        tests_passed = final_obs.metadata.get('tests_passed', 0)
        tests_total = final_obs.metadata.get('tests_total', 1)

        return tests_passed / tests_total if tests_total > 0 else 0.0

    def compute_step_rewards(self) -> List[float]:
        """Assign full reward to plan step, zero to execution."""
        final_score = self.score_trajectory(self._trajectory)
        rewards = [0.0] * len(self._trajectory)

        for i, (action, obs) in enumerate(self._trajectory):
            if getattr(action, 'metadata', {}).get('type') == 'plan':
                rewards[i] = final_score

        return rewards
```

### Example: Codenames (Mixed Per-Step + Trajectory)

```python
class CodenamesRubric(Rubric):
    """Combine per-clue quality with game outcome."""

    def __init__(self):
        super().__init__()
        self.clue_quality = ClueQualityRubric()    # Per-step
        self.game_outcome = GameWinRubric()         # Trajectory
        self.clue_scores: List[float] = []

    def __call__(self, action, observation) -> float:
        # Score clue quality per-step
        if getattr(action, 'metadata', {}).get('type') == 'give_clue':
            score = self.clue_quality(action, observation)
            self.clue_scores.append(score)

        # Delegate to trajectory rubric (it accumulates internally)
        outcome_reward = self.game_outcome(action, observation)

        if observation.done:
            # Combine: 30% clue quality, 70% game outcome
            clue_avg = sum(self.clue_scores) / len(self.clue_scores) if self.clue_scores else 0.0
            return 0.3 * clue_avg + 0.7 * outcome_reward
        else:
            return 0.0

    def reset(self):
        self.clue_scores = []
        self.game_outcome.reset()


class GameWinRubric(ExponentialDiscountingTrajectoryRubric):
    """Score based on win/loss outcome."""

    def score_trajectory(self, trajectory: List[Tuple[Any, Observation]]) -> float:
        if not trajectory:
            return 0.0
        _, final_obs = trajectory[-1]
        if final_obs.metadata.get('outcome') == 'win':
            return 1.0
        elif final_obs.metadata.get('outcome') == 'loss':
            return 0.0
        return 0.5
```

### Environment Integration

```python
class ChessEnvironment(Environment[ChessAction, ChessObservation, ChessState]):
    """Chess environment with trajectory-based scoring."""

    def __init__(self):
        super().__init__()
        self.rubric = ChessWinLossRubric(gamma=0.99)

    def reset(self, seed=None, episode_id=None, **kwargs) -> ChessObservation:
        self._state = ChessState(episode_id=episode_id or str(uuid4()))
        self.rubric.reset()  # Clear rubric's trajectory
        return self._make_observation()

    def step(self, action: ChessAction) -> ChessObservation:
        obs = self._apply_move(action)

        # Rubric handles trajectory accumulation internally
        obs.reward = self.rubric(action, obs)

        return obs
```

**Note**: The environment just calls `self.rubric(action, obs)` on every step. The rubric handles:
- Accumulating trajectory internally
- Returning 0.0 until done
- Computing final reward when done

### Training Loop: Retrieving Per-Step Rewards

For training systems that need per-step rewards for credit assignment:

```python
obs = env.reset()
while True:
    action = agent.act(obs)
    obs = env.step(action)

    if obs.done:
        # Get per-step rewards (discounting depends on rubric implementation)
        step_rewards = env.rubric.compute_step_rewards()

        # step_rewards[i] corresponds to trajectory step i
        # For ExponentialDiscountingTrajectoryRubric with gamma=0.99:
        # later moves get higher reward (closer to outcome)
        for (action, obs), reward in zip(env.rubric.trajectory, step_rewards):
            # Use for gradient computation...
            pass

        break
```

---

## Implementation Plan

This RFC will be implemented in stacked PRs:

### PR 1: Rubric Base Class

- `Rubric` base class with `forward()`, `__call__()`, child auto-registration
- Container rubrics: `Sequential`, `Gate`, `WeightedSum`, `RubricList`
- `RubricDict` for multi-task dispatch
- `get_rubric(path)` for nested access
- Unit tests for all containers

### PR 2: Environment Integration

- Update `Environment` base class to require `rubric` attribute
- Update `step()` to wire rubric output to observation reward
- Update environment documentation

### PR 3: Migrate Existing Environments

- Migrate `textarena_env` from `RewardProvider` to `Rubric`
- Update any other environments using custom reward patterns

### PR 4: Trajectory Rubrics

- `TrajectoryRubric` base class in `src/openenv/core/rubrics/trajectory.py`
- `ExponentialDiscountingTrajectoryRubric` concrete implementation
- Unit tests for trajectory accumulation, discounting, and reset behavior

### PR 5: Trajectory Rubric Examples

- Add trajectory rubric to an existing game environment (e.g., `connect4_env` or `openspiel_env`)
- Demonstrate integration with environment reset/step cycle

### PR 6: EnvPool (Future)

- Batch orchestration across stacked environments
- `step_batch()` helper for parallel execution
- May require separate RFC depending on scope

---

## Appendix: Patterns from Literature

The design supports patterns from recent academic work on rubric-based rewards:

| Pattern | Source | Implementation |
|---------|--------|----------------|
| Hierarchical gating | Liu et al., "RLTF: Reinforcement Learning from Unit Test Feedback" (TMLR 2023) | `Sequential(Gate(...), Gate(...), ...)` |
| All-or-nothing rewards | He et al., "AdvancedIF: Rubric-Based Instruction Following" (2024) | Custom `forward()` returning 1.0 only if all criteria pass |
| Multi-dimensional rubrics | Huang et al., "Rubicon: Reinforcement Learning with Rubric Anchors" (2024) | Composite rubrics with multiple children |
| Defense rubrics | Huang et al. (2024), He et al. (2024) | Gate with anti-hacking rubrics (e.g., `NoSycophancy`) |
| Gatekeeper mechanism | Liu et al., "OpenRubrics: Scalable Synthetic Rubric Generation" (2024) | `Sequential(hard_rules..., WeightedSum(soft_principles...))` |
| Implicit aggregation | Gonzalez et al., "Rubrics as Rewards" (Scale AI, 2024) | `LLMJudge` with all criteria in prompt for holistic scoring |
