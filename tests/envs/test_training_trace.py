# SPDX-License-Identifier: BSD-3-Clause

import json
from unittest.mock import MagicMock

import pytest
from openenv.core.harness import TrainingTrace, TrainingTurn
from openenv.core.harness.capture.contract import to_training_trace as graph_trace
from openenv.core.harness.capture.graph import RolloutGraph, TurnNode
from openenv.core.harness.capture.upstream import training_sampling
from openenv.harbor.contract import to_training_trace
from openenv.harbor.models import HarborRolloutResult, HarborTurn


def turn(**changes):
    return TrainingTurn(
        **{
            "node_id": "agent",
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [3, 4, 5],
            "per_token_logps": [-0.1, -0.2, -0.3],
            "loss_mask": [0, 0, 1, 0, 1],
            **changes,
        }
    )


def test_partial_mask_and_zero_mask_survive_wire_round_trip():
    original = TrainingTrace(turns=[turn(), turn(node_id="context", loss_mask=[0] * 5)])
    restored = TrainingTrace.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.turns[0].loss_mask == [0, 0, 1, 0, 1]
    assert restored.turns[1].per_token_logps == [-0.1, -0.2, -0.3]


@pytest.mark.parametrize(
    "changes",
    [
        {"loss_mask": None},
        {"loss_mask": [0, 0, 1]},
        {"loss_mask": [1, 0, 1, 0, 1]},
        {"loss_mask": [0, 0, True, 0, 1]},
        {"completion_token_ids": [3, True, 5]},
        {"per_token_logps": []},
        {"per_token_logps": [-0.1, float("nan"), -0.3]},
        {"per_token_logps": [-0.1, 0.2, -0.3]},
        {"per_token_logps": [-0.1, True, -0.3]},
        {"prompt_token_ids": []},
    ],
)
def test_malformed_capture_fails_at_producer_boundary(changes):
    with pytest.raises(ValueError):
        turn(**changes)


def test_missing_mask_and_unknown_version_rejected():
    values = turn().model_dump(exclude={"loss_mask"})
    with pytest.raises(ValueError):
        TrainingTurn.model_validate(values)
    with pytest.raises(ValueError):
        TrainingTrace(schema_version=2, turns=[])


def test_duplicate_call_is_not_supervised_twice():
    with pytest.raises(ValueError, match="duplicate node_ids"):
        TrainingTrace(turns=[turn(), turn()])


def test_shared_graph_prefix_is_emitted_once_and_partial_masks_preserved():
    graph = RolloutGraph()
    for name, prompt, sampled in [
        ("root", [1], [2, 3]),
        ("left", [1, 2, 3, 8], [4]),
        ("right", [1, 2, 3, 9], [5]),
        ("aux", [90], [91]),
    ]:
        graph.add_turn(
            TurnNode(
                node_id=name,
                prompt_ids=prompt,
                sampled_ids=sampled,
                sampled_logprobs=[-0.1] * len(sampled),
            )
        )
    document = {
        "sequences": [
            {
                "role": "agent",
                "node_ids": ["root", "left"],
                "loss_mask": [0, 1, 0, 0, 1],
            },
            {
                "role": "agent",
                "node_ids": ["root", "right"],
                "loss_mask": [0, 1, 0, 0, 1],
            },
            {"role": "auxiliary", "node_ids": ["aux"], "loss_mask": [0, 1]},
        ]
    }
    trace = graph_trace(graph, document)
    assert [t.node_id for t in trace.turns] == ["root", "left", "right"]
    assert sum(sum(t.loss_mask) for t in trace.turns) == 3
    assert trace.turns[0].loss_mask == [0, 1, 0]


def test_harbor_masks_do_not_change_reward_or_drop_zero_masked_agent_calls():
    turns = [
        HarborTurn(
            turn=0, **turn().model_dump(exclude={"request", "response", "metadata"})
        ),
        HarborTurn(
            turn=1,
            **turn(node_id="context", loss_mask=[0] * 5).model_dump(
                exclude={"request", "response", "metadata"}
            ),
        ),
        HarborTurn(turn=2, role="auxiliary"),
        HarborTurn(turn=3, discarded=True),
    ]
    for reward in [None, 0.0, 1.0]:
        result = HarborRolloutResult(turns=turns, reward=reward)
        trace = to_training_trace(result)
        assert [t.node_id for t in trace.turns] == ["agent", "context"]
        assert result.reward == reward
        result.turns[0].loss_mask = None
        with pytest.raises(ValueError, match="explicit loss_mask"):
            to_training_trace(result)
        result.turns[0].loss_mask = [0, 0, 1, 0, 1]


def test_eval_and_fatal_capture_are_not_empty_successful_training():
    with pytest.raises(ValueError, match="eval-only"):
        to_training_trace(HarborRolloutResult(rollout_type="eval"))
    with pytest.raises(ValueError, match="fatal"):
        to_training_trace(HarborRolloutResult(findings=["[FATAL] bad capture"]))


def test_native_stream_keeps_engine_prompt_ids_and_normalizes_token_ids():
    from opencode_env.sandbox.interception import (
        _accumulate_stream_chunk,
        _assemble_streamed_response,
        _build_turn_record,
    )

    acc = {
        "content_by_idx": {},
        "tool_calls_by_idx": {},
        "finish_by_idx": {},
        "logprobs_by_idx": {},
    }
    _accumulate_stream_chunk(
        {
            "prompt_token_ids": [1, 2],
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "answer"},
                    "logprobs": {"content": [{"token": "token_id:3", "logprob": -0.3}]},
                }
            ],
        },
        acc,
    )
    final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    _accumulate_stream_chunk(final, acc)
    record = _build_turn_record(
        turn_idx=0,
        request_body={},
        response_json=_assemble_streamed_response(final, acc),
        latency_s=0.1,
    )
    assert record.prompt_token_ids == [1, 2]
    assert record.completion_token_ids == [3]
    assert record.per_token_logps == [-0.3]


def test_native_policy_overrides_harness_before_generation():
    from opencode_env.config import OpenCodeConfig
    from opencode_env.harness import OpenCodeSessionFactory
    from opencode_env.sandbox.interception import _prepare_forwarded_body, ProxyConfig

    policy = training_sampling({"temperature": 0.8})
    factory = OpenCodeSessionFactory(
        config=OpenCodeConfig(base_url="http://unused"),
        sandbox_backend=MagicMock(),
        mode="transparent_proxy",
        sampling=policy,
    )
    forwarded = _prepare_forwarded_body(
        {"temperature": 1.0, "top_p": 0.2, "logprobs": False},
        ProxyConfig(upstream_url="http://unused", sampling=factory.sampling),
    )
    assert all(forwarded[k] == v for k, v in policy.items())
    assert forwarded["logprobs"] is True
    assert forwarded["return_tokens_as_token_ids"] is True
    assert forwarded["return_token_ids"] is True


def test_native_session_exports_all_agent_roots_without_auxiliary_calls():
    from opencode_env.config import OpenCodeConfig
    from opencode_env.harness import OpenCodeSession
    from opencode_env.task import OpenCodeTask

    records = []
    for i, tools in enumerate([[{"type": "function"}], None, [{"type": "function"}]]):
        records.append(
            {
                "turn": i,
                "prompt_token_ids": [10 + i],
                "completion_token_ids": [20 + i],
                "per_token_logps": [-0.1],
                "request": {
                    "messages": [{"role": "user", "content": "task"}],
                    "tools": tools,
                },
                "response": {"choices": [{"message": {"content": "ok"}}]},
            }
        )
    sandbox = MagicMock(
        read_text=lambda path: "\n".join(json.dumps(r) for r in records)
    )
    session = OpenCodeSession(
        sandbox=sandbox,
        config=OpenCodeConfig(base_url="http://unused"),
        task=OpenCodeTask.coerce("task"),
        proxy_trace_path="trace",
    )
    trace = session.fetch_training_trace()
    assert [t.node_id for t in trace.turns] == ["0", "2"]
    assert all(t.loss_mask == [0, 1] for t in trace.turns)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "choice_ids, token_text, reject_pairing",
    [
        (None, "token_id:20", False),
        ([], "token_id:20", False),
        ([20], "answer", False),
        ([20], "token_id:20", False),
        ([20], "token_id:21", True),
    ],
)
def test_native_proxy_http_capture_reaches_training_contract(
    monkeypatch, tmp_path, stream, choice_ids, token_text, reject_pairing
):
    from functools import partial

    import httpx
    from fastapi.testclient import TestClient
    from opencode_env.config import OpenCodeConfig
    from opencode_env.harness import OpenCodeSession
    from opencode_env.sandbox import interception
    from opencode_env.task import OpenCodeTask

    policy = training_sampling({"temperature": 0.7})
    logprobs = {"content": [{"token": token_text, "logprob": -0.25}]}
    ids = {"token_ids": choice_ids} if choice_ids is not None else {}

    def engine(request):
        body = json.loads(request.content)
        assert all(body[key] == value for key, value in policy.items())
        assert body["logprobs"] is True
        assert body["return_tokens_as_token_ids"] is True
        assert body["return_token_ids"] is True
        if body["messages"][0]["content"] == "fail":
            return httpx.Response(400, json={"error": {"message": "rejected request"}})
        if stream:
            chunks = [
                {
                    "prompt_token_ids": [10],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "answer"},
                            "logprobs": logprobs,
                            **(ids if choice_ids else {}),
                        }
                    ],
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            if choice_ids == []:
                chunks[-1]["choices"][0]["token_ids"] = []
            content = (
                "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
                + "data: [DONE]\n\n"
            )
            return httpx.Response(
                200, text=content, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(
            200,
            json={
                "prompt_token_ids": [10],
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "answer"},
                        "logprobs": logprobs,
                        **ids,
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    monkeypatch.setattr(
        interception.httpx,
        "AsyncClient",
        partial(httpx.AsyncClient, transport=httpx.MockTransport(engine)),
    )
    path = tmp_path / "trace.jsonl"
    config = interception.ProxyConfig(
        upstream_url="http://engine", trace_path=str(path), sampling=policy
    )
    with TestClient(interception._build_app(config)) as client:
        failed = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "stream": stream,
                "messages": [{"role": "user", "content": "fail"}],
            },
        )
        assert failed.status_code == 400
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "task"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "bash", "parameters": {"type": "object"}},
                    }
                ],
                "stream": stream,
                "temperature": 1.0,
                "logprobs": False,
            },
        )
        assert response.status_code == 200
        assert "answer" in response.text
    session = OpenCodeSession(
        sandbox=MagicMock(read_text=lambda _: path.read_text()),
        task=OpenCodeTask(instruction="task"),
        config=OpenCodeConfig(base_url="http://engine"),
        verifier=None,
        proxy_trace_path=str(path),
    )
    raw = session.fetch_proxy_trace()
    assert len(raw) == 2
    if reject_pairing:
        with pytest.raises(
            ValueError, match="missing or invalid engine tokens/logprobs"
        ):
            session.fetch_training_trace()
        assert session.fetch_proxy_trace() == raw
        return
    trace = session.fetch_training_trace()
    assert len(trace.turns) == 1
    assert trace.turns[0].prompt_token_ids == [10]
    assert trace.turns[0].completion_token_ids == [20]
    assert trace.turns[0].per_token_logps == [-0.25]
    assert trace.turns[0].loss_mask == [0, 1]


@pytest.mark.parametrize("response", [{}, {"choices": []}])
def test_missing_successful_response_is_not_silently_skipped(response):
    from opencode_env.config import OpenCodeConfig
    from opencode_env.harness import OpenCodeSession
    from opencode_env.task import OpenCodeTask

    session = OpenCodeSession(
        sandbox=MagicMock(
            read_text=lambda _: json.dumps({"request": {}, "response": response})
        ),
        config=OpenCodeConfig(base_url="http://unused"),
        task=OpenCodeTask.coerce("task"),
        proxy_trace_path="trace",
    )
    with pytest.raises(ValueError, match="no response choices"):
        session.fetch_training_trace()


def test_all_failed_upstream_calls_remain_a_transport_failure():
    from opencode_env.config import OpenCodeConfig
    from opencode_env.harness import OpenCodeSession
    from opencode_env.task import OpenCodeTask

    session = OpenCodeSession(
        sandbox=MagicMock(
            read_text=lambda _: json.dumps(
                {"request": {}, "response": {"error": "unavailable"}}
            )
        ),
        config=OpenCodeConfig(base_url="http://unused"),
        task=OpenCodeTask.coerce("task"),
        proxy_trace_path="trace",
    )
    with pytest.raises(RuntimeError, match="no successful model calls"):
        session.fetch_training_trace()


def test_streamed_choice_ids_accumulate_across_chunks_without_reparsing_text():
    from opencode_env.sandbox.interception import (
        _accumulate_stream_chunk,
        _assemble_streamed_response,
        _build_turn_record,
    )

    acc = {
        "content_by_idx": {},
        "tool_calls_by_idx": {},
        "finish_by_idx": {},
        "logprobs_by_idx": {},
    }
    for token in [20, 21]:
        _accumulate_stream_chunk(
            {
                "prompt_token_ids": [10],
                "choices": [
                    {
                        "index": 0,
                        "token_ids": [token],
                        "delta": {"content": "token_id:literal"},
                        "logprobs": {
                            "content": [{"token": "token_id:literal", "logprob": -0.25}]
                        },
                    }
                ],
            },
            acc,
        )
    record = _build_turn_record(
        turn_idx=1,
        request_body={},
        response_json=_assemble_streamed_response({}, acc),
        latency_s=0.1,
    )
    assert record.completion_token_ids == [20, 21]
    assert record.per_token_logps == [-0.25, -0.25]


def test_choice_token_logprob_mismatch_is_not_truncated():
    from opencode_env.sandbox.interception import _build_turn_record

    record = _build_turn_record(
        turn_idx=1,
        request_body={},
        response_json={
            "prompt_token_ids": [10],
            "choices": [
                {
                    "token_ids": [20, 21],
                    "logprobs": {"content": [{"token": "answer", "logprob": -0.25}]},
                }
            ],
        },
        latency_s=0.1,
    )
    assert record.completion_token_ids == [20, 21]
    with pytest.raises(ValueError):
        TrainingTurn(
            node_id="call",
            prompt_token_ids=record.prompt_token_ids,
            completion_token_ids=record.completion_token_ids,
            per_token_logps=record.per_token_logps,
            loss_mask=[0, 1, 1],
        )


def test_mispaired_auxiliary_call_does_not_invalidate_agent_capture():
    from opencode_env.config import OpenCodeConfig
    from opencode_env.harness import OpenCodeSession
    from opencode_env.task import OpenCodeTask

    records = [
        {
            "turn": i,
            "prompt_token_ids": [10 + i],
            "completion_token_ids": [20],
            "per_token_logps": [-0.25],
            "request": {"messages": [], "tools": tools},
            "response": {
                "choices": [
                    {
                        "token_ids": [20],
                        "message": {"content": "answer"},
                        "logprobs": {"content": [{"token": token, "logprob": -0.25}]},
                    }
                ]
            },
        }
        for i, (tools, token) in enumerate(
            [
                (None, "token_id:21"),
                ([{"type": "function"}], "token_id:20"),
            ]
        )
    ]
    session = OpenCodeSession(
        sandbox=MagicMock(
            read_text=lambda _: "\n".join(json.dumps(r) for r in records)
        ),
        config=OpenCodeConfig(base_url="http://unused"),
        task=OpenCodeTask.coerce("task"),
        proxy_trace_path="trace",
    )
    trace = session.fetch_training_trace()
    assert [t.node_id for t in trace.turns] == ["1"]
    assert trace.turns[0].per_token_logps == [-0.25]
    assert session.fetch_proxy_trace() == records
