import sys
from pathlib import Path

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from peft import LoraConfig, TaskType, get_peft_model

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import examples.mini_swe_env.train_swe_async_grpo as train_mod
from examples.mini_swe_env.train_swe_async_grpo import (
    _chunked_logprob_backbone,
    _default_lora_target_modules,
    _lora_config_from_env,
    _model_context_limit,
    _patch_chunked_lm_head_for_wrapped_causal_lm,
    _select_task_indices,
    _sanitize_lora_merged_weight_name,
)


def test_select_task_indices_uses_offset_stride_and_repeat() -> None:
    selected = _select_task_indices(
        total_tasks=20,
        max_tasks=3,
        task_offset=2,
        task_stride=4,
        task_indices_raw="",
        repeat_tasks=2,
    )
    assert selected == [2, 6, 10, 2, 6, 10]


def test_select_task_indices_prefers_explicit_indices() -> None:
    selected = _select_task_indices(
        total_tasks=20,
        max_tasks=5,
        task_offset=0,
        task_stride=1,
        task_indices_raw="16,3,16",
        repeat_tasks=1,
    )
    assert selected == [16, 3, 16]


def test_select_task_indices_rejects_out_of_range_index() -> None:
    with pytest.raises(IndexError):
        _select_task_indices(
            total_tasks=5,
            max_tasks=2,
            task_offset=0,
            task_stride=1,
            task_indices_raw="0,5",
            repeat_tasks=1,
        )


def test_model_context_limit_prefers_first_positive_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Cfg:
        max_position_embeddings = 40960
        model_max_length = 32768

    monkeypatch.setattr(
        train_mod.AutoConfig,
        "from_pretrained",
        lambda model_name: _Cfg(),
    )

    assert _model_context_limit("Qwen/Qwen3-8B") == 40960


def test_model_context_limit_rejects_missing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Cfg:
        max_position_embeddings = None
        model_max_length = None
        max_seq_len = None
        seq_length = None

    monkeypatch.setattr(
        train_mod.AutoConfig,
        "from_pretrained",
        lambda model_name: _Cfg(),
    )

    with pytest.raises(ValueError):
        _model_context_limit("missing-context-model")


def test_default_lora_target_modules_for_qwen() -> None:
    assert _default_lora_target_modules("Qwen/Qwen3-14B") == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def test_sanitize_lora_merged_weight_name_strips_wrapper_paths() -> None:
    assert (
        _sanitize_lora_merged_weight_name(
            "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
        )
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        _sanitize_lora_merged_weight_name(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        )
        is None
    )


def test_lora_config_from_env_returns_full_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SWE_LORA", raising=False)
    config, summary = _lora_config_from_env("Qwen/Qwen3-8B")
    assert config is None
    assert summary == "full"


def test_chunked_logprob_backbone_uses_decoder_for_wrapped_causal_lm() -> None:
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    base_model = Qwen3ForCausalLM(cfg)
    peft_model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )

    assert _chunked_logprob_backbone(base_model) is base_model.model
    assert _chunked_logprob_backbone(peft_model) is peft_model.model.model


def test_patch_chunked_lm_head_for_wrapped_causal_lm_handles_peft_model() -> None:
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    peft_model = get_peft_model(
        Qwen3ForCausalLM(cfg),
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    _patch_chunked_lm_head_for_wrapped_causal_lm(
        peft_model,
        temperature=1.0,
        chunk_size=32,
    )

    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    outputs = peft_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        completion_mask=attention_mask,
        use_cache=False,
    )

    assert outputs["log_probs"].shape == (2, 7)
    assert outputs["entropy"].shape == (2, 7)
