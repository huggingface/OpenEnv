"""Mini SWE environment — SWE-bench task adapter + environment.

Exports:
    Task adapter (Phase 2):
        SWETask, validate_swe_task, adapt_swebench_lite_row, ...

    Environment + client (Phase 3):
        SWEEnvironment, MiniSWEEnv, SWERolloutResult, ...
"""

from .task_loader_swebench_lite import (
    adapt_swebench_lite_row,
    adapt_swebench_lite_rows,
    AdaptationSkip,
    coerce_swe_task,
    deterministic_train_eval_split,
    load_task_file,
    read_jsonl_rows,
    SWEBenchLiteAdapterError,
    SWETask,
    SWETaskValidationError,
    validate_swe_task,
    write_tasks_jsonl,
)

from .models import (
    SWECommandResult,
    SWERolloutResult,
    SWEState,
)

from .client import MiniSWEEnv

__all__ = [
    # Task adapter (Phase 2)
    "AdaptationSkip",
    "SWEBenchLiteAdapterError",
    "SWETask",
    "SWETaskValidationError",
    "adapt_swebench_lite_row",
    "adapt_swebench_lite_rows",
    "coerce_swe_task",
    "deterministic_train_eval_split",
    "load_task_file",
    "read_jsonl_rows",
    "validate_swe_task",
    "write_tasks_jsonl",
    # Models (Phase 3)
    "MiniSWEEnv",
    "SWECommandResult",
    "SWERolloutResult",
    "SWEState",
]
