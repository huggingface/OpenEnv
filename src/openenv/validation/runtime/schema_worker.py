"""Disposable JSON Schema evaluator; invoked as a script to avoid package imports."""

import json
import sys

from jsonschema import Draft202012Validator
from referencing import Registry

# Quoting the 8 MiB raw trace costs at most 16 MiB. The remainder covers the
# 1 MiB schema after conservative 6x JSON normalization and 2x quoting, plus row
# metadata. Limits apply to bytes, independently of the text stream's encoding.
MAX_INPUT_BYTES = 32 * 1024 * 1024


def main():
    """Evaluate bounded inputs, printing only error locations, never subject values."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (3, 4))
        if sys.platform == "linux":
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    except (ImportError, ValueError, OSError):
        pass  # The parent always enforces the independent wall-clock deadline.
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        sys.stdout.write(json.dumps(["schema worker input exceeds its size budget"]))
        return
    payload = json.loads(raw)
    problems = []
    try:
        schema = json.loads(payload["schema_json"])
        Draft202012Validator.check_schema(schema)
        # An empty registry has no retrieval callback: unresolved references
        # cannot trigger host filesystem or network access.
        validator = Draft202012Validator(schema, registry=Registry())
        for row in payload["observations"]:
            data = json.loads(row["response_json"])["data"]
            observation = dict(data["observation"])
            observation.update(reward=data["reward"], done=data["done"])
            for error in validator.iter_errors(observation):
                location = "/".join(str(x) for x in error.absolute_path)[:160]
                problems.append(
                    f"exchange {row['index']}: schema mismatch at {location or '/'}"
                )
                if len(problems) >= 20:
                    break
            if len(problems) >= 20:
                break
    except Exception:
        problems.append("invalid or unevaluable observation schema")
    sys.stdout.write(json.dumps(problems))


if __name__ == "__main__":
    main()
