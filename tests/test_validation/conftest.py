import json
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures" / "validation"
VALID_MANIFEST_FIXTURES = [
    "served_min_pass",
    "no_oracle",
    "harbor_task_min",
    "posttrain_task_min",
]

INVALID_MANIFEST_FIXTURES = [
    "broken_manifest",
    "unpinned_judge",
]


def load_fixture_manifest(name: str) -> dict:
    return json.loads((FIXTURES / name / "normalized_manifest.json").read_text())
