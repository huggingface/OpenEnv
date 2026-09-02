"""One E2B template for a whole suite, instead of one per task.

Harbor names each sandbox template `f"{environment_name}__{env_hash}"` and takes `environment_name`
from the task (`harbor/trial/trial.py:628` passes `self.task.short_name`). For a suite whose tasks all
share one image that is 2238 identical templates: measured on
`AdithyaSK/data_agent_rl_environment_train`, all 2238 Dockerfiles hash to ONE value and the
environment directories are byte-identical, and the aliases Harbor built differed only in their task
prefix — `0000_555_555434_qa_3__016e9c9f617d` and `0000_650_650548_qa_2__016e9c9f617d` carry the same
hash.

The cost of that is not just build time. Harbor decides whether to build from
`AsyncTemplate.alias_exists()`, which goes true the moment a build STARTS, so a GRPO group hitting a
task for the first time races itself: the losers skip the build and 404 with
`tag 'default' does not exist`. Sharing one alias means the template is built once, ever, and no group
can race a task's first visit.

**The hash is what makes this safe.** `env_hash` stays in the alias, so a task whose environment
genuinely differs still gets its own template automatically — this collapses identical environments,
it does not force unlike ones together. A suite with three distinct images yields three templates,
whatever the shared name is.

Opt-in through `HARBOR_SHARED_ENV_NAME`, the same variable an older Harbor honoured natively before
the knob was removed.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_applied = False


def enable_shared_templates(name: str = "") -> bool:
    """Point every E2B template at one shared `environment_name`. Idempotent.

    Args:
        name (`str`, *optional*):
            The shared name. Defaults to `HARBOR_SHARED_ENV_NAME`; when neither is set this is a no-op
            so the default behaviour is unchanged.

    Returns:
        `bool`: whether sharing is now active.
    """
    global _applied
    shared = name or os.environ.get("HARBOR_SHARED_ENV_NAME", "")
    if not shared or _applied:
        return _applied

    try:
        from harbor.environments import e2b
    except Exception as exc:  # noqa: BLE001 - a missing e2b extra must not break a docker run
        logger.warning("cannot share templates: %s", exc)
        return False

    original = e2b.E2BEnvironment.__init__

    def __init__(self, *args, **kwargs):  # noqa: N807
        # Keyword-only in practice: Harbor passes `environment_name=` from three call sites. Handled
        # defensively anyway, because silently failing to rename would reintroduce per-task templates
        # while looking like it worked.
        if "environment_name" in kwargs:
            kwargs["environment_name"] = shared
        elif len(args) >= 2:
            args = (args[0], shared, *args[2:])
        else:
            logger.warning(
                "E2BEnvironment was constructed without a recognisable environment_name; "
                "this rollout keeps its per-task template"
            )
        return original(self, *args, **kwargs)

    e2b.E2BEnvironment.__init__ = __init__
    _applied = True
    logger.info(
        "E2B templates share the name %r; the environment hash still distinguishes unlike images",
        shared,
    )
    return True
