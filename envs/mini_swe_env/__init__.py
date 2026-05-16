"""Mini SWE environment — task models + environment.

Exports:
    Task models:
        SWETask, validate_swe_task, coerce_swe_task, SWETaskValidationError

    Environment + client:
        SWEEnvironment, MiniSWEEnv, SWERolloutResult, ...
"""

from .models import (
    SWECommandResult,
    SWERolloutResult,
    SWEState,
    SWETask,
    SWETaskValidationError,
    coerce_swe_task,
    validate_swe_task,
)

from .client import MiniSWEEnv

__all__ = [
    # Task models
    "SWETask",
    "SWETaskValidationError",
    "coerce_swe_task",
    "validate_swe_task",
    # Models
    "MiniSWEEnv",
    "SWECommandResult",
    "SWERolloutResult",
    "SWEState",
]
