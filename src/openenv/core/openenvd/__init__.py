# SPDX-License-Identifier: BSD-3-Clause

"""openenvd: the control plane that runs each environment as kernel-isolated zones.

openenvd is PID 1 of an environment unit. It stays in the unit's parent
namespaces and creates three zones of containers in child namespaces: the agent
zone, hidden services and privileged observers. Every call out of the agent
zone crosses a recording relay, and the trace is hash-chained and sealed.
"""

from .contract import (
    ContainerSpec,
    EnforcementSpec,
    Guarantee,
    load_manifest,
    Manifest,
    ManifestError,
    parse_manifest,
    Phase,
    Strength,
    Tier,
    validate_manifest,
    ZoneKind,
    ZoneSpec,
)
from .phases import Actor, PhaseError, PhaseMachine, Transition

__all__ = [
    "Actor",
    "ContainerSpec",
    "EnforcementSpec",
    "Guarantee",
    "Manifest",
    "ManifestError",
    "Phase",
    "PhaseError",
    "PhaseMachine",
    "Strength",
    "Tier",
    "Transition",
    "ZoneKind",
    "ZoneSpec",
    "load_manifest",
    "parse_manifest",
    "validate_manifest",
]
