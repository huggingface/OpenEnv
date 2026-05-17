"""Async GRPO control-plane primitives for Mini SWE on HF Spaces.

This package provides control-plane setup utilities:
- configure + launch the interception server on Space-exposed networking,
- keep trainer-facing rollout registration in the same process,
- expose simple runtime stats for leak detection.
"""

from .control_plane import (
    SWEAsyncControlPlane,
    SWEAsyncControlPlaneConfig,
    build_hf_space_base_url,
)
from .rollout_worker import (
    SWEAsyncRolloutSample,
    SWEAsyncRolloutWorker,
    SWEAsyncRolloutWorkerConfig,
    build_answer_tool_invoker,
    build_openai_chat_forwarder,
)

__all__ = [
    "SWEAsyncControlPlane",
    "SWEAsyncControlPlaneConfig",
    "SWEAsyncRolloutSample",
    "SWEAsyncRolloutWorker",
    "SWEAsyncRolloutWorkerConfig",
    "build_answer_tool_invoker",
    "build_hf_space_base_url",
    "build_openai_chat_forwarder",
]
