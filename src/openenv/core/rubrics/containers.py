# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Container rubrics for composing reward computations.

These containers provide common aggregation patterns for rubrics,
similar to how PyTorch provides nn.Sequential alongside nn.Module.

See RFC 004 for full design: rfcs/004-rubrics.md
"""

import asyncio
import inspect
from typing import Any, Dict, Iterator, List, Mapping, Tuple, Union

from openenv.core.rubrics.base import Rubric


def _in_async_context() -> bool:
    """Check if we're currently in an async context."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class Sequential(Rubric):
    """Run rubrics in order, fail-fast on zero.

    Runs child rubrics in order. If any returns 0, stops immediately
    and returns 0. This implements hierarchical gating patterns where
    syntax checks run before execution checks.

    Usage:
        rubric = Sequential(
            Gate(Compiles()),
            Gate(PassesTests(), threshold=0.5),
            WeightedSum([PassesTests(), StyleRubric()], weights=[0.7, 0.3])
        )
    """

    def __init__(self, *rubrics: Rubric):
        """Initialize with rubrics to run in sequence.

        Args:
            *rubrics: Rubrics to run in order. Stops and returns 0 if any
                child returns 0.
        """
        super().__init__()
        for i, rubric in enumerate(rubrics):
            setattr(self, f"rubric_{i}", rubric)
        self._rubric_list = list(rubrics)

    def forward(self, action: Any, observation: Any) -> float:
        """Run rubrics in order, return 0 if any returns 0. Sync version."""
        result = 1.0
        for rubric in self._rubric_list:
            score = rubric(action, observation)
            if score == 0.0:
                return 0.0
            result = score
        return result

    def __call__(self, action: Any, observation: Any):
        """Override to choose sync or async path based on children."""
        # Empty case - check if in async context
        if not self._rubric_list:
            if _in_async_context():
                return self._empty_async(action, observation)
            else:
                # Pre-hooks
                for hook in self._forward_pre_hooks:
                    hook(self, action, observation)
                result = 1.0
                self.last_score = result
                for hook in self._forward_hooks:
                    hook(self, action, observation, result)
                return result

        # Call first rubric to see if it's async
        first_result = self._rubric_list[0](action, observation)
        if inspect.iscoroutine(first_result):
            # At least one child is async, use async path
            return self._call_async_detected(action, observation, first_result)
        else:
            # Continue with sync path
            if first_result == 0.0:
                # Pre-hooks
                for hook in self._forward_pre_hooks:
                    hook(self, action, observation)
                self.last_score = 0.0
                for hook in self._forward_hooks:
                    hook(self, action, observation, 0.0)
                return 0.0

            final_result = first_result
            for i, rubric in enumerate(self._rubric_list[1:], start=1):
                score = rubric(action, observation)
                if inspect.iscoroutine(score):
                    # Found async mid-way, switch to async
                    # We already called rubric at index i, so pass the coroutine and remaining rubrics
                    return self._call_async_mid(
                        action,
                        observation,
                        final_result,
                        score,
                        self._rubric_list[i + 1 :],
                    )
                if score == 0.0:
                    # Pre-hooks
                    for hook in self._forward_pre_hooks:
                        hook(self, action, observation)
                    self.last_score = 0.0
                    for hook in self._forward_hooks:
                        hook(self, action, observation, 0.0)
                    return 0.0
                final_result = score

            # All sync - check if in async context
            if _in_async_context():
                return self._wrap_sync_result(action, observation, final_result)
            else:
                # Pre-hooks
                for hook in self._forward_pre_hooks:
                    hook(self, action, observation)
                self.last_score = final_result
                for hook in self._forward_hooks:
                    hook(self, action, observation, final_result)
                return final_result

    async def _empty_async(self, action, observation):
        """Async path for empty sequential."""
        for hook in self._forward_pre_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation)
            else:
                hook(self, action, observation)

        result = 1.0
        self.last_score = result

        for hook in self._forward_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation, result)
            else:
                hook(self, action, observation, result)
        return result

    async def _wrap_sync_result(self, action, observation, result):
        """Wrap sync result for async context."""
        for hook in self._forward_pre_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation)
            else:
                hook(self, action, observation)

        self.last_score = result

        for hook in self._forward_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation, result)
            else:
                hook(self, action, observation, result)
        return result

    async def _call_async_detected(self, action, observation, first_coro):
        """Async path when first child is async."""
        for hook in self._forward_pre_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation)
            else:
                hook(self, action, observation)

        result = await first_coro
        if result == 0.0:
            self.last_score = 0.0
            for hook in self._forward_hooks:
                if inspect.iscoroutinefunction(hook):
                    await hook(self, action, observation, result)
                else:
                    hook(self, action, observation, result)
            return 0.0

        for rubric in self._rubric_list[1:]:
            score = rubric(action, observation)
            if inspect.iscoroutine(score):
                score = await score
            if score == 0.0:
                self.last_score = 0.0
                for hook in self._forward_hooks:
                    if inspect.iscoroutinefunction(hook):
                        await hook(self, action, observation, 0.0)
                    else:
                        hook(self, action, observation, 0.0)
                return 0.0
            result = score

        self.last_score = result
        for hook in self._forward_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation, result)
            else:
                hook(self, action, observation, result)
        return result

    async def _call_async_mid(
        self, action, observation, current_result, first_async_coro, remaining
    ):
        """Async path when async detected mid-execution."""
        for hook in self._forward_pre_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation)
            else:
                hook(self, action, observation)

        # Await the first async rubric (already called)
        result = await first_async_coro
        if result == 0.0:
            self.last_score = 0.0
            for hook in self._forward_hooks:
                if inspect.iscoroutinefunction(hook):
                    await hook(self, action, observation, 0.0)
                else:
                    hook(self, action, observation, 0.0)
            return 0.0

        # Continue with remaining rubrics
        for rubric in remaining:
            score = rubric(action, observation)
            if inspect.iscoroutine(score):
                score = await score
            if score == 0.0:
                self.last_score = 0.0
                for hook in self._forward_hooks:
                    if inspect.iscoroutinefunction(hook):
                        await hook(self, action, observation, 0.0)
                    else:
                        hook(self, action, observation, 0.0)
                return 0.0
            result = score

        self.last_score = result
        for hook in self._forward_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation, result)
            else:
                hook(self, action, observation, result)
        return result

    def __len__(self) -> int:
        return len(self._rubric_list)

    def __getitem__(self, index: int) -> Rubric:
        return self._rubric_list[index]


class Gate(Rubric):
    """Threshold wrapper - returns 0 if child score is below threshold.

    Useful for hard constraints like "must pass 50% of tests".

    Usage:
        rubric = Gate(PassesTests(), threshold=0.5)
        # Returns PassesTests() score if >= 0.5, else 0.0
    """

    def __init__(self, rubric: Rubric, threshold: float = 1.0):
        """Initialize with a rubric and threshold.

        Args:
            rubric: The rubric to gate.
            threshold: Minimum score required. If child returns less than
                this, Gate returns 0. Default is 1.0 (must pass completely).
        """
        super().__init__()
        self.rubric = rubric
        self.threshold = threshold

    def forward(self, action: Any, observation: Any) -> float:
        """Return child score if >= threshold, else 0. Sync version."""
        score = self.rubric(action, observation)
        if score < self.threshold:
            return 0.0
        return score

    def __call__(self, action: Any, observation: Any):
        """Override to handle async child."""
        # Call child
        score = self.rubric(action, observation)

        if inspect.iscoroutine(score):
            # Child is async
            return self._call_async(action, observation, score)
        else:
            # Child is sync
            # Pre-hooks
            for hook in self._forward_pre_hooks:
                hook(self, action, observation)
            result = 0.0 if score < self.threshold else score
            self.last_score = result
            for hook in self._forward_hooks:
                hook(self, action, observation, result)
            return result

    async def _call_async(self, action, observation, score_coro):
        """Async path."""
        for hook in self._forward_pre_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation)
            else:
                hook(self, action, observation)

        score = await score_coro
        result = 0.0 if score < self.threshold else score
        self.last_score = result

        for hook in self._forward_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation, result)
            else:
                hook(self, action, observation, result)
        return result


class WeightedSum(Rubric):
    """Weighted combination of child rubrics.

    Standard aggregation pattern for multi-criteria evaluation.

    Usage:
        rubric = WeightedSum(
            [PassesTests(), StyleRubric()],
            weights=[0.7, 0.3]
        )
    """

    def __init__(self, rubrics: List[Rubric], weights: List[float]):
        """Initialize with rubrics and weights.

        Args:
            rubrics: List of rubrics to combine.
            weights: Weight for each rubric. Must sum to 1.0.

        Raises:
            ValueError: If lengths don't match or weights don't sum to 1.0.
        """
        super().__init__()
        if len(rubrics) != len(weights):
            raise ValueError(
                f"Number of rubrics ({len(rubrics)}) must match "
                f"number of weights ({len(weights)})"
            )
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

        for i, rubric in enumerate(rubrics):
            setattr(self, f"rubric_{i}", rubric)
        self._rubric_list = list(rubrics)
        self._weights = list(weights)

    def forward(self, action: Any, observation: Any) -> float:
        """Return weighted sum of child scores. Sync version."""
        total = 0.0
        for rubric, weight in zip(self._rubric_list, self._weights):
            score = rubric(action, observation)
            total += score * weight
        return total

    def __call__(self, action: Any, observation: Any):
        """Override to handle async children with parallel execution."""
        # Call all rubrics
        results = [rubric(action, observation) for rubric in self._rubric_list]

        # Check if any are async
        has_async = any(inspect.iscoroutine(r) for r in results)

        if has_async:
            # Use async path
            return self._call_async(action, observation, results)
        else:
            # Sync path
            # Pre-hooks
            for hook in self._forward_pre_hooks:
                hook(self, action, observation)
            total = 0.0
            for score, weight in zip(results, self._weights):
                total += score * weight
            self.last_score = total
            for hook in self._forward_hooks:
                hook(self, action, observation, total)
            return total

    async def _call_async(self, action, observation, results):
        """Async path with parallel execution."""
        for hook in self._forward_pre_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation)
            else:
                hook(self, action, observation)

        # Separate sync and async results
        async_tasks = []
        async_indices = []
        scores = [None] * len(results)

        for i, result in enumerate(results):
            if inspect.iscoroutine(result):
                async_tasks.append(result)
                async_indices.append(i)
            else:
                scores[i] = result

        # Await all async tasks in parallel
        if async_tasks:
            async_scores = await asyncio.gather(*async_tasks)
            for i, score in zip(async_indices, async_scores):
                scores[i] = score

        # Compute weighted sum
        total = 0.0
        for score, weight in zip(scores, self._weights):
            total += score * weight

        self.last_score = total

        for hook in self._forward_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation, total)
            else:
                hook(self, action, observation, total)
        return total

    @property
    def weights(self) -> List[float]:
        """Get the weights (read-only copy)."""
        return list(self._weights)


class RubricList(Rubric):
    """Container for dynamic lists of rubrics.

    Analogous to nn.ModuleList. Does not define aggregation - use within
    a parent rubric that implements custom logic.

    Usage:
        class MultiGameRubric(Rubric):
            def __init__(self, games: List[str]):
                super().__init__()
                self.games = RubricList([GameRubric(g) for g in games])

            def forward(self, action, obs) -> float:
                return self.games[obs.game_index](action, obs)
    """

    def __init__(self, rubrics: List[Rubric] = None):
        """Initialize with optional list of rubrics.

        Args:
            rubrics: Optional list of rubrics to start with.
        """
        super().__init__()
        self._rubrics: List[Rubric] = []
        if rubrics is not None:
            for i, rubric in enumerate(rubrics):
                self.append(rubric)

    def forward(self, action: Any, observation: Any) -> float:
        """RubricList does not define aggregation - override in parent."""
        raise NotImplementedError(
            "RubricList.forward() is not implemented. "
            "Use RubricList within a parent rubric that defines aggregation."
        )

    def append(self, rubric: Rubric) -> None:
        """Add a rubric to the list."""
        index = len(self._rubrics)
        setattr(self, f"rubric_{index}", rubric)
        self._rubrics.append(rubric)

    def extend(self, rubrics: List[Rubric]) -> None:
        """Add multiple rubrics to the list."""
        for rubric in rubrics:
            self.append(rubric)

    def __len__(self) -> int:
        return len(self._rubrics)

    def __getitem__(self, index: int) -> Rubric:
        return self._rubrics[index]

    def __iter__(self) -> Iterator[Rubric]:
        return iter(self._rubrics)


class RubricDict(Rubric):
    """Container for named rubrics with keyed access.

    Analogous to nn.ModuleDict. Enables keyed access for multi-task
    environments where different tasks require different rubrics.

    Usage:
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

        # Access: env.rubric.games["pong"]
    """

    def __init__(self, rubrics: Dict[str, Rubric] = None):
        """Initialize with optional dictionary of rubrics.

        Args:
            rubrics: Optional dictionary mapping names to rubrics.
        """
        super().__init__()
        self._rubric_dict: Dict[str, Rubric] = {}
        if rubrics is not None:
            for name, rubric in rubrics.items():
                self[name] = rubric

    def forward(self, action: Any, observation: Any) -> float:
        """RubricDict does not define aggregation - override in parent."""
        raise NotImplementedError(
            "RubricDict.forward() is not implemented. "
            "Use RubricDict within a parent rubric that defines aggregation."
        )

    def __setitem__(self, key: str, rubric: Rubric) -> None:
        """Add a rubric with the given key."""
        setattr(self, key, rubric)
        self._rubric_dict[key] = rubric

    def __getitem__(self, key: str) -> Rubric:
        """Get rubric by key."""
        return self._rubric_dict[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._rubric_dict

    def __len__(self) -> int:
        return len(self._rubric_dict)

    def __iter__(self) -> Iterator[str]:
        return iter(self._rubric_dict)

    def keys(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self._rubric_dict.keys())

    def values(self) -> Iterator[Rubric]:
        """Iterate over rubrics."""
        return iter(self._rubric_dict.values())

    def items(self) -> Iterator[Tuple[str, Rubric]]:
        """Iterate over (key, rubric) pairs."""
        return iter(self._rubric_dict.items())

    def update(self, rubrics: Union[Dict[str, Rubric], Mapping[str, Rubric]]) -> None:
        """Update with rubrics from a dictionary."""
        for name, rubric in rubrics.items():
            self[name] = rubric
