"""Mini SWE environment — SWE-Gym task loader + grading + environment.

Exports:
    Task models:
        SWEGymTask, SWETask, validate_swe_task, coerce_swe_task, ...

    Task loader (SWE-Gym):
        load_swegym_tasks, validate_swegym_task, get_instance_image, ...

    Grading:
        grade_from_case_results, GradeResult

    Environment + client:
        SWEEnvironment, MiniSWEEnv, SWERolloutResult, ...
"""

from .models import (
    SWECommandResult,
    SWEGymTask,
    SWERolloutResult,
    SWEState,
    SWETask,
    SWETaskValidationError,
    coerce_swe_task,
    validate_swe_task,
)

from .task_loader_swegym import (
    SWEGymLoadError,
    get_instance_image,
    load_swegym_tasks,
    load_swegym_tasks_from_dicts,
    swegym_task_to_swe_task,
    validate_swegym_task,
)

from .grading import (
    GradeResult,
    GradingError,
    grade_from_case_results,
)

from .client import MiniSWEEnv

__all__ = [
    # Task models
    "SWEGymTask",
    "SWETask",
    "SWETaskValidationError",
    "coerce_swe_task",
    "validate_swe_task",
    # Task loader
    "SWEGymLoadError",
    "get_instance_image",
    "load_swegym_tasks",
    "load_swegym_tasks_from_dicts",
    "swegym_task_to_swe_task",
    "validate_swegym_task",
    # Grading
    "GradeResult",
    "GradingError",
    "grade_from_case_results",
    # Server models
    "MiniSWEEnv",
    "SWECommandResult",
    "SWERolloutResult",
    "SWEState",
]
