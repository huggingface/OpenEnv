# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base Rubric class for reward computation.

Rubrics compute rewards from actions and observations. The API is modeled
after PyTorch's nn.Module: users implement forward(), and the framework
handles child registration and hooks.

See RFC 004 for full design: rfcs/004-rubrics.md
"""

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


class Rubric(ABC):
    """Abstract base class for reward computation.

    A Rubric computes a reward signal from an action and observation.
    Subclasses implement forward() to define the reward logic.

    Usage:
        class MyRubric(Rubric):
            def forward(self, action, observation) -> float:
                return 1.0 if action.valid else 0.0

        rubric = MyRubric()
        reward = rubric(action, observation)

    Child rubrics are auto-registered when assigned as attributes,
    enabling hierarchical composition and introspection.
    """

    _rubric_children: Dict[str, "Rubric"]
    _forward_hooks: List[Callable]
    _forward_pre_hooks: List[Callable]
    last_score: Optional[float]

    def __init__(self):
        # Use object.__setattr__ to avoid triggering __setattr__ during init
        object.__setattr__(self, "_rubric_children", {})
        object.__setattr__(self, "_forward_hooks", [])
        object.__setattr__(self, "_forward_pre_hooks", [])
        object.__setattr__(self, "last_score", None)

    def __setattr__(self, name: str, value: Any) -> None:
        # Auto-register child rubrics when assigned as attributes
        if isinstance(value, Rubric):
            self._rubric_children[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, action: Any, observation: Any):
        """Evaluate the rubric with hooks.

        Args:
            action: The action taken by the agent.
            observation: The resulting observation.

        Returns:
            Reward value (typically 0.0 to 1.0).
        """
        # Check if forward method is async BEFORE calling it
        if inspect.iscoroutinefunction(self.forward):
            # Async path - pre-hooks will be called in _call_async
            result = self.forward(action, observation)
            return self._call_async(action, observation, result)
        else:
            # Sync path - call pre-hooks BEFORE forward()
            for hook in self._forward_pre_hooks:
                hook(self, action, observation)
            result = self.forward(action, observation)
            return self._call_sync(action, observation, result)

    def _call_sync(self, action: Any, observation: Any, result: float) -> float:
        """Synchronous call path."""
        self.last_score = result

        # Post-forward hooks
        for hook in self._forward_hooks:
            hook(self, action, observation, result)

        return result

    async def _call_async(self, action: Any, observation: Any, result_coro) -> float:
        """Asynchronous call path."""
        # Pre-forward hooks
        for hook in self._forward_pre_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation)
            else:
                hook(self, action, observation)

        # Await the forward result
        result = await result_coro
        self.last_score = result

        # Post-forward hooks
        for hook in self._forward_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation, result)
            else:
                hook(self, action, observation, result)

        return result

    @abstractmethod
    def forward(self, action: Any, observation: Any) -> float:
        """Compute the reward. Implement this in subclasses.

        Args:
            action: The action taken by the agent.
            observation: The resulting observation.

        Returns:
            Reward value (typically 0.0 to 1.0).
        """
        raise NotImplementedError

    def register_forward_hook(
        self, hook: Callable[["Rubric", Any, Any, float], None]
    ) -> None:
        """Register a hook called after forward().

        Args:
            hook: Callable with signature (rubric, action, observation, result).
        """
        self._forward_hooks.append(hook)

    def register_forward_pre_hook(
        self, hook: Callable[["Rubric", Any, Any], None]
    ) -> None:
        """Register a hook called before forward().

        Args:
            hook: Callable with signature (rubric, action, observation).
        """
        self._forward_pre_hooks.append(hook)

    def children(self) -> Iterator["Rubric"]:
        """Iterate over immediate child rubrics."""
        yield from self._rubric_children.values()

    def named_children(self) -> Iterator[Tuple[str, "Rubric"]]:
        """Iterate over immediate child rubrics with names."""
        yield from self._rubric_children.items()

    def rubrics(self) -> Iterator["Rubric"]:
        """Iterate over all descendant rubrics (depth-first)."""
        for child in self._rubric_children.values():
            yield child
            yield from child.rubrics()

    def named_rubrics(self, prefix: str = "") -> Iterator[Tuple[str, "Rubric"]]:
        """Iterate over all descendant rubrics with dot-separated names."""
        for name, child in self._rubric_children.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, child
            yield from child.named_rubrics(full_name)

    def get_rubric(self, path: str) -> "Rubric":
        """Access a nested rubric by dot-separated path.

        Args:
            path: Dot-separated path (e.g., "code.syntax").

        Returns:
            The rubric at the specified path.

        Raises:
            KeyError: If the path does not exist.
        """
        parts = path.split(".")
        current = self
        for part in parts:
            if part not in current._rubric_children:
                raise KeyError(f"Rubric path not found: {path}")
            current = current._rubric_children[part]
        return current

    def reset(self) -> None:
        """Reset any internal state. Override in subclasses if needed."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Serialize rubric configuration for checkpointing."""
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load rubric configuration from checkpoint."""
        pass
