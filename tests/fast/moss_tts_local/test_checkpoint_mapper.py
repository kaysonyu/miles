from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from miles.policies.moss_tts_local.checkpoint_mapper import (
    MossTTSLocalCheckpointWeightMapper,
    moss_tts_local_hf_tensor,
)
from miles.policies.moss_tts_local.serving_weight_adapter import MossTTSLocalServingWeightAdapter

NUM_GPUS = 0


class Reader:
    def __init__(self, **tensors):
        self.tensors = tensors

    def __contains__(self, name):
        return name in self.tensors

    def get_tensor(self, name):
        return self.tensors[name]


def _config(**overrides):
    global_config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        tie_word_embeddings=False,
    )
    values = dict(
        model_type="moss_tts_local",
        qwen3_config=global_config,
        gpt2_config=SimpleNamespace(
            n_layer=1,
            n_head=32,
            n_inner=9728,
            rope_base=1_000_000.0,
            layer_norm_epsilon=1e-6,
        ),
        n_vq=12,
        audio_vocab_size=1024,
        audio_pad_code=1024,
        audio_start_token_id=151652,
        audio_user_slot_token_id=151654,
        audio_assistant_slot_token_id=151656,
        audio_end_token_id=151653,
        tie_audio_embeddings_and_output_weights=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _export_args():
    return SimpleNamespace(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
    )


def test_global_embedding_round_trips_through_canonical_transformer_name():
    parameter = torch.arange(16 * 8).reshape(16, 8)
    name = "module.module.language_model.embedding.word_embeddings.weight"
    exported = dict(MossTTSLocalServingWeightAdapter().convert_parameter(_export_args(), "ignored", name, parameter))

    loaded = moss_tts_local_hf_tensor(name, Reader(**exported), _config())

    assert torch.equal(loaded, parameter)


def test_global_qkv_round_trips_through_transformer_prefix():
    parameter = torch.arange(16 * 8).reshape(16, 8)
    name = "module.module.language_model.decoder.layers.0.self_attention.linear_qkv.weight"
    exported = dict(MossTTSLocalServingWeightAdapter().convert_parameter(_export_args(), "ignored", name, parameter))

    loaded = moss_tts_local_hf_tensor(name, Reader(**exported), _config())

    assert set(exported) == {
        "transformer.layers.0.self_attn.q_proj.weight",
        "transformer.layers.0.self_attn.k_proj.weight",
        "transformer.layers.0.self_attn.v_proj.weight",
    }
    assert torch.equal(loaded, parameter)


def test_global_final_norm_round_trips_through_transformer_prefix():
    parameter = torch.arange(8)
    name = "module.module.language_model.decoder.final_layernorm.weight"

    exported = dict(MossTTSLocalServingWeightAdapter().convert_parameter(_export_args(), "ignored", name, parameter))
    loaded = moss_tts_local_hf_tensor(name, Reader(**exported), _config())

    assert set(exported) == {"transformer.norm.weight"}
    assert torch.equal(loaded, parameter)


@pytest.mark.parametrize(
    ("name", "canonical_name"),
    [
        (
            "module.module.language_model.decoder.layers.0.input_layernorm.weight",
            "transformer.layers.0.input_layernorm.weight",
        ),
        (
            "module.module.language_model.decoder.layers.0.pre_mlp_layernorm.weight",
            "transformer.layers.0.post_attention_layernorm.weight",
        ),
    ],
)
def test_local_backend_norms_round_trip_through_hf_names(name, canonical_name):
    parameter = torch.arange(8)

    exported = dict(MossTTSLocalServingWeightAdapter().convert_parameter(_export_args(), "ignored", name, parameter))
    loaded = moss_tts_local_hf_tensor(name, Reader(**exported), _config())

    assert set(exported) == {canonical_name}
    assert torch.equal(loaded, parameter)


def test_audio_checkpoint_embedding_round_trips_without_a_synthetic_pad_row():
    checkpoint = torch.randn(1024, 8)
    name = "module.module.audio_embeddings.5.weight"

    loaded = moss_tts_local_hf_tensor(name, Reader(**{"audio_embeddings.5.weight": checkpoint}), _config())
    exported = MossTTSLocalServingWeightAdapter().convert_parameter(_export_args(), "ignored", name, loaded)

    assert loaded.shape == (1024, 8)
    assert loaded is checkpoint
    assert torch.equal(exported[0][1], checkpoint)


def test_independent_audio_head_round_trips_without_using_embedding_weight():
    head = torch.randn(1024, 8)
    embedding = torch.randn(1024, 8)
    name = "module.module.audio_lm_heads.5.weight"

    loaded = moss_tts_local_hf_tensor(
        name,
        Reader(
            **{
                "audio_lm_heads.5.weight": head,
                "audio_embeddings.5.weight": embedding,
            }
        ),
        _config(),
    )
    exported = MossTTSLocalServingWeightAdapter().convert_parameter(_export_args(), "ignored", name, loaded)

    assert loaded is head
    assert not torch.equal(loaded, embedding)
    assert exported[0][0] == "audio_lm_heads.5.weight"
    assert exported[0][1] is head


def test_serving_manifest_requires_every_independent_audio_head():
    adapter = MossTTSLocalServingWeightAdapter()
    names = {
        "transformer.embed_tokens.weight",
        "transformer.norm.weight",
        "transformer.layers.0.input_layernorm.weight",
        "local_transformer.ln_f.weight",
        "local_text_lm_head.weight",
        *(f"audio_embeddings.{depth}.weight" for depth in range(12)),
        *(f"audio_lm_heads.{depth}.weight" for depth in range(12)),
    }
    adapter.validate_manifest(names)

    names.remove("audio_lm_heads.7.weight")
    with pytest.raises(ValueError, match="audio_lm_heads.7.weight"):
        adapter.validate_manifest(names)


def test_local_parameters_load_without_renaming():
    parameter = torch.randn(8, 8)
    name = "module.module.local_transformer.h.0.attn.c_proj.weight"

    loaded = moss_tts_local_hf_tensor(
        name,
        Reader(**{"local_transformer.h.0.attn.c_proj.weight": parameter}),
        _config(),
    )

    assert loaded is parameter


def test_checkpoint_identity_fails_closed():
    mapper = MossTTSLocalCheckpointWeightMapper()
    mapper.validate_identity(_config())

    with pytest.raises(ValueError, match="n_vq mismatch"):
        mapper.validate_identity(_config(n_vq=8))
    with pytest.raises(ValueError, match="independent audio"):
        mapper.validate_identity(_config(tie_audio_embeddings_and_output_weights=True))
    with pytest.raises(ValueError, match="local_ffn_hidden_size mismatch"):
        config = _config()
        config.gpt2_config.n_inner = 10240
        mapper.validate_identity(config)
