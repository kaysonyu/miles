from __future__ import annotations

import pytest
import torch
from megatron.core.dist_checkpointing.mapping import ShardedTensor

from miles.policies.moss_tts_local.pretrained_checkpoint import (
    _validate_manifest_identity,
    remap_moss_tts_local_sharded_keys,
    remove_runtime_extra_state,
    validate_dcp_mismatches,
)

NUM_GPUS = 0


def _sharded(key: str) -> ShardedTensor:
    return ShardedTensor.from_rank_offsets(key, torch.zeros(2, 3))


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 2,
        "model_identity": {
            "architecture": "MOSS-TTS-Local",
            "tensor_schema": {
                "embedding_head_storage": "split_v1",
                "num_layers": 36,
                "hidden_size": 2560,
                "num_attention_heads": 32,
                "num_query_groups": 8,
                "kv_channels": 128,
                "ffn_hidden_size": 9728,
                "text_vocab_size": 151936,
                "n_vq": 12,
                "speech_vocab_size": 1024,
                "local_num_layers": 1,
                "local_hidden_size": 2560,
                "local_num_attention_heads": 32,
                "local_ffn_hidden_size": 9728,
                "qk_layernorm": True,
            },
            "forward_semantics": {
                "embedding_head_usage": "untied",
                "audio_end_token_id": 151653,
                "local_activation": "silu",
                "local_norm_epsilon": 1e-6,
                "local_rope_base": 1_000_000.0,
                "norm_epsilon": 1e-6,
                "rotary_base": 1_000_000.0,
                "rotary_interleaved": False,
            },
        },
    }


def test_native_storage_key_mapping_preserves_local_state_keys():
    state = {
        "language_model.embedding.word_embeddings.weight": _sharded("language_model.embedding.word_embeddings.weight"),
        "language_model.decoder.final_layernorm.weight": _sharded("language_model.decoder.final_layernorm.weight"),
        "audio_embeddings.3.weight": _sharded("audio_embeddings.3.weight"),
        "audio_lm_heads.3.weight": _sharded("audio_lm_heads.3.weight"),
        "local_transformer.ln_f.weight": _sharded("local_transformer.ln_f.weight"),
    }

    storage_keys = remap_moss_tts_local_sharded_keys(state)

    assert set(state) == {
        "language_model.embedding.word_embeddings.weight",
        "language_model.decoder.final_layernorm.weight",
        "audio_embeddings.3.weight",
        "audio_lm_heads.3.weight",
        "local_transformer.ln_f.weight",
    }
    assert storage_keys == {
        "text_embedding.word_embeddings.weight",
        "decoder.final_layernorm.weight",
        "audio_embeddings.3.word_embeddings.weight",
        "audio_lm_heads.3.weight",
        "local_transformer.ln_f.weight",
    }


def test_native_manifest_accepts_only_split_untied_mosslite_identity():
    manifest = _manifest()
    _validate_manifest_identity(manifest)

    manifest["model_identity"]["forward_semantics"]["embedding_head_usage"] = "tied_head_authoritative"
    with pytest.raises(ValueError, match="embedding_head_usage"):
        _validate_manifest_identity(manifest)


def test_native_manifest_rejects_old_audio_end_token():
    manifest = _manifest()
    manifest["model_identity"]["forward_semantics"]["audio_end_token_id"] = 151670

    with pytest.raises(ValueError, match="audio_end_token_id"):
        _validate_manifest_identity(manifest)


def test_runtime_extra_state_is_explicitly_excluded_from_logical_dcp():
    weight = _sharded("decoder.final_layernorm.weight")
    extra = _sharded("decoder.final_layernorm._extra_state")
    state = {
        "language_model.decoder.final_layernorm.weight": weight,
        "language_model.decoder.final_layernorm._extra_state": extra,
    }

    removed = remove_runtime_extra_state(state)

    assert removed == {"language_model.decoder.final_layernorm._extra_state"}
    assert state == {"language_model.decoder.final_layernorm.weight": weight}


def test_strict_dcp_audit_allows_only_checkpoint_runtime_objects():
    validate_dcp_mismatches(
        {
            "audio_lm_heads.0._extra_state",
            "decoder.layers.self_attention.core_attention._extra_state",
        },
        set(),
    )

    with pytest.raises(ValueError, match="checkpoint_only"):
        validate_dcp_mismatches({"audio_lm_heads.0.weight"}, set())
    with pytest.raises(ValueError, match="model_only"):
        validate_dcp_mismatches(set(), {"local_text_lm_head.weight"})
