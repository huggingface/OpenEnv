# envs/finqa_env/server/__init__.py
"""Server-side components for the FinQA environment."""


def __getattr__(name):
    if name == "FinQAEnvironment":
        from .finqa_environment import FinQAEnvironment

        return FinQAEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FinQAEnvironment"]
