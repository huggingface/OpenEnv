"""Bounded Unicode evidence must survive the schema-worker transport."""

import json

from conftest import load_fixture_manifest
from openenv.validation.graders import Subject
from openenv.validation.graders.runtime import ObservationSchemaGrader
from openenv.validation.manifest import NormalizedManifest
from openenv.validation.runtime.collector import MAX_MESSAGE_BYTES, MAX_TRACE_BYTES
from openenv.validation.runtime.contracts import RuntimeEvidence, WireExchange
from openenv.validation.types import CheckStatus


def test_valid_unicode_episode_within_wire_budgets_passes_schema_worker(tmp_path):
    exchanges = []
    for index in range(7):
        response = json.dumps(
            {
                "type": "observation",
                "data": {
                    "observation": {"text": "é" * 300_000},
                    "reward": 1,
                    "done": False,
                },
            },
            ensure_ascii=False,
        )
        exchanges.append(
            WireExchange(
                operation="reset" if index == 0 else "step",
                request_json="{}",
                response_json=response,
            )
        )

    frame_sizes = [len(row.response_json.encode()) for row in exchanges]
    assert max(frame_sizes) < MAX_MESSAGE_BYTES
    assert sum(frame_sizes) < MAX_TRACE_BYTES
    subject = Subject(
        root=tmp_path,
        manifest=NormalizedManifest.model_validate(
            load_fixture_manifest("served_min_pass")
        ),
        image_ref=None,
        running=None,
        outputs_dir=tmp_path,
        runtime_evidence=RuntimeEvidence(
            exchanges=tuple(exchanges),
            observation_schema_json=json.dumps(
                {"type": "object", "properties": {"text": {"type": "string"}}}
            ),
        ),
    )

    result = ObservationSchemaGrader().run(subject)

    assert result.status is CheckStatus.PASS, result.evidence
