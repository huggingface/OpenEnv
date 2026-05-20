"""Async GRPO training for Mini SWE with Pi agent.

Architecture: Pi runs in an HF Sandbox, its LLM calls are intercepted
by :class:`SWEAsyncControlPlane`, and :class:`SWERolloutWorker` forwards
them to vLLM ``/v1/completions`` to get exact token IDs and logprobs.

The worker implements TRL's ``RolloutWorkerProtocol`` and plugs into
``AsyncGRPOTrainer(rollout_worker=...)``.
"""

from .control_plane import (
    SWEAsyncControlPlane,
    SWEAsyncControlPlaneConfig,
)
from .rollout_worker import (
    RolloutSample,
    SWERolloutWorker,
    WorkerConfig,
)

__all__ = [
    "RolloutSample",
    "SWEAsyncControlPlane",
    "SWEAsyncControlPlaneConfig",
    "SWERolloutWorker",
    "WorkerConfig",
]
