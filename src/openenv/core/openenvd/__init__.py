# SPDX-License-Identifier: BSD-3-Clause
"""Opt-in isolated environment runtime.

Runtime exports load lazily so importing configuration or clients never imports
server implementation modules.
"""

from importlib import import_module

_EXPORTS = {
    "Collector": "observation",
    "GraderClient": "client",
    "HarnessEventSink": "harness",
    "IsolationCapabilities": "isolation",
    "IsolationError": "isolation",
    "ObservationEvent": "observation",
    "ObservationEventType": "policy",
    "OpenEnvDConfig": "policy",
    "Principal": "policy",
    "Runtime": "runtime",
    "SurfacePolicy": "policy",
    "TaskSpec": "models",
    "create_surface_app": "surfaces",
    "detect_capabilities": "isolation",
    "main": "daemon",
    "observer_stream": "client",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
