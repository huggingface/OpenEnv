"""UI downloads must preserve the exact supervision consumed by the trainer."""

import json
from pathlib import Path

import pytest
from harbor_env.harness import to_trace_entries
from openenv.harbor.contract import export_training_contract
from openenv.harbor.models import HarborRolloutResult, HarborTurn
from openenv.harbor.ui import _write_contract


def rollout():
    return HarborRolloutResult(
        task_name="partial-mask",
        reward=1.0,
        capture_level="tokens",
        turns=[
            HarborTurn(
                turn=0,
                node_id="agent",
                prompt_token_ids=[1, 2],
                completion_token_ids=[3, 4],
                per_token_logps=[-0.2, -0.3],
                loss_mask=[0, 0, 1, 0],
            )
        ],
    )


def test_ui_download_and_trainer_share_partial_masks():
    result = rollout()
    path = Path(_write_contract(result.model_dump()))
    try:
        document = json.loads(path.read_text())
        assert document == export_training_contract(result)
        assert document["trace_entries"] == to_trace_entries(result)
        assert document["turns"][0]["loss_mask"] == [0, 0, 1, 0]
        assert document["n_trainable_tokens"] == 1
    finally:
        path.unlink()
        path.parent.rmdir()


def test_auxiliary_and_rejected_turns_cannot_gain_supervision_in_download():
    result = rollout()
    for i, changes in enumerate(
        (dict(role="auxiliary"), dict(discarded=True), dict(trainable=False)), 1
    ):
        result.turns.append(
            result.turns[0].model_copy(update={"turn": i, "node_id": str(i), **changes})
        )
    exported = export_training_contract(result)
    assert len(exported["trace_entries"]) == 1
    assert all(t["loss_mask"] == [0, 0, 0, 0] for t in exported["turns"][1:])


def test_eval_or_fatal_results_cannot_be_exported_as_training():
    result = rollout()
    result.rollout_type = "eval"
    assert _write_contract(result.model_dump()) is None
    with pytest.raises(ValueError, match="eval-only"):
        to_trace_entries(result)
    result.rollout_type = "train"
    result.findings = ["[FATAL] invalid token provenance"]
    with pytest.raises(ValueError, match="fatal"):
        _write_contract(result.model_dump())


def test_invalid_mask_cannot_be_hidden_by_the_download_path():
    result = rollout()
    result.turns[0].loss_mask = [0, 1]
    with pytest.raises(ValueError):
        _write_contract(result.model_dump())


def test_ui_validation_keeps_the_qualified_acp_profile(tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace

    from openenv.harbor.ui import harbor_gradio_builder

    cells = []
    for provider in ["openai", "anthropic", "hf", "vllm"]:
        cell = {
            "harness": "acp",
            "provider": provider,
            "status": "eval_pass",
            "evidence": ["capture.json"],
            "configuration": {"acp_profile": "opencode-1.18.30"},
        }
        if provider == "vllm":
            cell.update(
                status="optimizer_pass",
                optimizer_validated=True,
                optimizer_evidence={
                    "matches_current_captures": True,
                    "result": "result.json",
                    "inputs": "inputs.json",
                    "scope": "diagnostic",
                    "model": "model",
                    "revision": "pinned",
                    "rows": 2,
                },
            )
        cells.append(cell)
    report = tmp_path / "matrix.json"
    report.write_text(json.dumps({"cells": cells}))
    monkeypatch.setenv("OPENENV_HARBOR_QUALIFICATION_REPORT", str(report))
    monkeypatch.setattr(
        importlib.import_module("openenv.core.harness.capture.validate_llm"),
        "validate_llm",
        lambda *args, **kwargs: SimpleNamespace(
            reachable=True,
            trainable=False,
            ok=True,
            capture_level="text",
            param_fixes=[],
            findings=[],
        ),
    )
    monkeypatch.setattr(
        importlib.import_module("openenv.harbor.capabilities"),
        "capabilities",
        lambda **kwargs: SimpleNamespace(
            available_sandboxes=["e2b"],
            sandboxes=[],
            harnesses=[
                SimpleNamespace(name="acp", dialect="openai"),
                SimpleNamespace(name="opencode", dialect="openai"),
            ],
        ),
    )
    monkeypatch.setattr("openenv.harbor.serving.HarborService.current", lambda: None)
    app = harbor_gradio_builder()
    validate = next(
        block.fn
        for block in app.fns.values()
        if getattr(block.fn, "__name__", "") == "on_validate"
    )
    _, harness, _, state, button = validate(
        "https://provider.example/v1", "model", "", "openai", "eval"
    )
    assert state["allowed_harnesses"] == ["acp"]
    assert state["harness_profiles"] == {"acp": "opencode-1.18.30"}
    assert "profile: opencode-1.18.30" in harness["choices"][0][0]
    assert button["interactive"]
    _, _, _, experimental_state, _ = validate(
        "https://provider.example/v1", "model", "", "openai", "eval", True
    )
    assert experimental_state["allowed_harnesses"] == ["acp", "opencode"]


def test_live_view_follows_its_own_session_with_concurrent_rollouts(
    tmp_path, monkeypatch
):
    import asyncio
    import threading
    from types import SimpleNamespace

    from openenv.core.harness.capture.sessions import SessionRegistry
    from openenv.harbor.ui import harbor_gradio_builder

    registry = SessionRegistry()
    release = threading.Event()
    service = SimpleNamespace(
        capture=SimpleNamespace(
            registry=registry,
            app=SimpleNamespace(
                state=SimpleNamespace(upstreams=SimpleNamespace(default=(None, "text")))
            ),
        ),
        capture_level="text",
        model="test-model",
        public_url="https://capture.example",
    )
    monkeypatch.setattr("openenv.harbor.serving.HarborService.current", lambda: service)
    monkeypatch.setattr(
        "openenv.harbor.tasks.HarborTaskProvider.task_dir", lambda *args: tmp_path
    )
    # Registry-wide discovery is not a safe way to identify the caller's run.
    monkeypatch.setattr(
        registry,
        "list_ids",
        lambda: pytest.fail("must not discover other users' sessions"),
    )

    async def run(**kwargs):
        registry.create("unrelated-private-run")
        session = registry.create("this-ui-run")
        kwargs["on_session_created"](session.session_id)
        for _ in range(1000):
            if release.is_set():
                break
            await asyncio.sleep(0.01)
        assert release.is_set(), "the live view never found its own session"
        return HarborRolloutResult(
            rollout_type="eval", capture_level="text", session_id=session.session_id
        )

    def transcript(session):
        assert session.session_id == "this-ui-run"
        release.set()
        return "this-ui-run trace"

    monkeypatch.setattr("openenv.harbor.rollout.run_rollout", run)
    monkeypatch.setattr("openenv.harbor.ui._transcript_html", transcript)
    app = harbor_gradio_builder()
    on_run = next(
        block.fn
        for block in app.fns.values()
        if getattr(block.fn, "__name__", "") == "on_run"
    )
    frames = list(
        on_run(
            {"ok": True, "allowed_harnesses": ["opencode"], "purpose": "eval"},
            "test",
            0,
            "opencode",
            "e2b",
        )
    )
    assert any(frame[1] == "this-ui-run trace" for frame in frames)
    assert all("unrelated-private-run" not in str(frame) for frame in frames)
