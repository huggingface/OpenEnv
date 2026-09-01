"""The committed JSON Schemas must match the pydantic models (CI sync check)."""

import json
from pathlib import Path

import pytest
from openenv.validation.manifest import NormalizedManifest

SCHEMAS_DIR = (
    Path(__file__).parent.parent.parent / "src" / "openenv" / "validation" / "schemas"
)

EXPORTS = {
    "manifest.schema.json": NormalizedManifest,
}


@pytest.mark.parametrize("fname", sorted(EXPORTS))
def test_committed_schema_matches_model(fname):
    committed = (SCHEMAS_DIR / fname).read_text()
    rendered = (
        json.dumps(EXPORTS[fname].model_json_schema(), indent=2, sort_keys=True) + "\n"
    )
    assert committed == rendered, (
        f"{fname} is stale; run: PYTHONPATH=src python scripts/sync_validation_schemas.py --fix"
    )
