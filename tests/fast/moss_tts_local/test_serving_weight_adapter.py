from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from miles.policies.moss_tts_local.serving_weight_adapter import MossTTSLocalServingWeightAdapter

NUM_GPUS = 0


def _args() -> Namespace:
    return Namespace(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        num_query_groups=1,
        kv_channels=4,
    )


def test_audio_export_uses_canonical_name_without_synthetic_pad_row():
    weight = torch.arange(1024 * 8).reshape(1024, 8)

    converted = MossTTSLocalServingWeightAdapter().convert_parameter(
        _args(),
        "ignored",
        "module.module.audio_embeddings.3.weight",
        weight,
    )

    assert converted[0][0] == "audio_embeddings.3.weight"
    assert converted[0][1].shape == (1024, 8)
    assert torch.equal(converted[0][1], weight)


def test_global_qwen_names_are_exported_under_transformer_prefix():
    adapter = MossTTSLocalServingWeightAdapter()

    embedding = adapter.convert_parameter(
        _args(),
        "ignored",
        "module.module.language_model.embedding.word_embeddings.weight",
        torch.zeros(16, 8),
    )
    norm = adapter.convert_parameter(
        _args(),
        "ignored",
        "module.module.language_model.decoder.final_layernorm.weight",
        torch.zeros(8),
    )

    assert embedding[0][0] == "transformer.embed_tokens.weight"
    assert norm[0][0] == "transformer.norm.weight"


def test_local_checkpoint_names_remain_stable():
    adapter = MossTTSLocalServingWeightAdapter()
    weight = torch.zeros(8, 8)

    converted = adapter.convert_parameter(
        _args(),
        "ignored",
        "module.module.local_transformer.h.0.attn.c_proj.weight",
        weight,
    )

    assert converted == [("local_transformer.h.0.attn.c_proj.weight", weight)]


def test_manifest_requires_global_local_decision_and_every_audio_table():
    names = {
        "transformer.embed_tokens.weight",
        "transformer.norm.weight",
        "transformer.layers.0.self_attn.q_proj.weight",
        "local_transformer.h.0.ln_1.weight",
        "local_text_lm_head.weight",
        *(f"audio_embeddings.{depth}.weight" for depth in range(12)),
        *(f"audio_lm_heads.{depth}.weight" for depth in range(12)),
    }
    adapter = MossTTSLocalServingWeightAdapter()

    adapter.validate_manifest(names)
    names.remove("audio_embeddings.11.weight")
    with pytest.raises(ValueError, match="audio_embeddings.11.weight"):
        adapter.validate_manifest(names)


def test_unknown_or_quantized_export_fails_closed():
    adapter = MossTTSLocalServingWeightAdapter()
    with pytest.raises(ValueError, match="Unknown MOSS"):
        adapter.convert_parameter(_args(), "ignored", "module.module.unowned.weight", torch.zeros(1))
    with pytest.raises(ValueError, match="quantized"):
        adapter.convert_parameter(
            _args(),
            "ignored",
            "module.module.local_text_lm_head.weight",
            torch.zeros(2, 8),
            {"quant_method": "something"},
        )


def test_miles_global_decoder_layer_name_is_exported():
    converted = MossTTSLocalServingWeightAdapter().convert_parameter(
        _args(), "ignored", "module.module.decoder.layers.0.input_layernorm.weight", torch.ones(8)
    )
    assert converted[0][0] == "transformer.layers.0.input_layernorm.weight"
