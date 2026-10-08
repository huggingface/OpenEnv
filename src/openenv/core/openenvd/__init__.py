# SPDX-License-Identifier: BSD-3-Clause
"""openenvd: policy-scoped surfaces with pluggable enforcement (RFC 009).

The contract (principals, surfaces, assets, workload paths, egress, required
guarantees) is declared in `openenv.yaml` and owned by OpenEnv. Enforcement is
delegated to a registered backend: `"openshell"` (kernel-enforced via NVIDIA
OpenShell) or `"local"` (host subprocesses, no guarantees).

Exports load lazily so importing the contract never imports backend modules.
"""

from importlib import import_module

_EXPORTS = {
    "EgressPolicy": "policy",
    "EgressRule": "policy",
    "EnforcementBackend": "backends",
    "EnforcementSpec": "policy",
    "EnforcementUnavailable": "backends",
    "Guarantee": "policy",
    "IsolationError": "isolation",
    "ObservationEventType": "policy",
    "OpenEnvDConfig": "policy",
    "OpenShellConfig": "policy",
    "Principal": "policy",
    "Sandbox": "backends",
    "SurfacePolicy": "policy",
    "WorkloadPaths": "policy",
    "get_backend": "backends",
    "load_config": "policy",
    "register_backend": "backends",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
