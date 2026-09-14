"""SGLang-target checkpoint mapping for mossLite MOSS-TTS Local."""

from __future__ import annotations

import re
from typing import Any

import torch

from miles.policies.moss_tts_local.spec import MOSS_TTS_LOCAL_SPEC


def _strip_wrappers(name: str) -> str:
    while name.startswith("module."):
        name = name.removeprefix("module.")
    return name


def _global_config(config):
    return (
        getattr(config, "qwen3_config", None)
        or getattr(config, "language_config", None)
        or getattr(config, "text_config", None)
        or config
    )


def _merge_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, config) -> torch.Tensor:
    num_groups = int(config.num_key_value_heads)
    num_heads = int(config.num_attention_heads)
    head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // num_heads)
    trailing_shape = q.shape[1:]
    q = q.reshape(num_groups, num_heads // num_groups * head_dim, *trailing_shape)
    k = k.reshape(num_groups, head_dim, *trailing_shape)
    v = v.reshape(num_groups, head_dim, *trailing_shape)
    return torch.cat((q, k, v), dim=1).reshape(-1, *trailing_shape).contiguous()


def _global_hf_tensor(name: str, reader, config) -> torch.Tensor:
    direct = {
        "embedding.word_embeddings.weight": "transformer.embed_tokens.weight",
        "decoder.final_layernorm.weight": "transformer.norm.weight",
    }
    if name in direct:
        return reader.get_tensor(direct[name])

    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise KeyError(f"Unsupported MOSS-TTS Local global parameter {name!r}")
    layer, rest = match.groups()
    prefix = f"transformer.layers.{layer}"
    mapping = {
        "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
        "self_attention.linear_proj.bias": "self_attn.o_proj.bias",
        "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
        "input_layernorm.weight": "input_layernorm.weight",
        "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
        "self_attention.k_layernorm.weight": "self_attn.k_norm.weight",
        "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
        "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
        "mlp.linear_fc2.weight": "mlp.down_proj.weight",
    }
    if rest in mapping:
        return reader.get_tensor(f"{prefix}.{mapping[rest]}")
    qkv = re.fullmatch(r"self_attention\.linear_qkv\.(weight|bias)", rest)
    if qkv:
        suffix = qkv.group(1)
        return _merge_qkv(
            *(reader.get_tensor(f"{prefix}.self_attn.{projection}_proj.{suffix}") for projection in "qkv"),
            config,
        )
    if rest == "mlp.linear_fc1.weight":
        return torch.cat(
            (
                reader.get_tensor(f"{prefix}.mlp.gate_proj.weight"),
                reader.get_tensor(f"{prefix}.mlp.up_proj.weight"),
            ),
            dim=0,
        )
    raise KeyError(f"Unsupported MOSS-TTS Local global parameter {name!r}")


def moss_tts_local_hf_tensor(name: str, reader, config) -> torch.Tensor:
    normalized = _strip_wrappers(name)
    if normalized.startswith("language_model."):
        return _global_hf_tensor(normalized.removeprefix("language_model."), reader, _global_config(config))
    if normalized.startswith("audio_embeddings.") and normalized.endswith(".weight"):
        tensor = reader.get_tensor(normalized)
        if tensor.shape[0] != MOSS_TTS_LOCAL_SPEC.audio_vocab_size:
            raise ValueError(
                f"Checkpoint tensor {normalized} must have {MOSS_TTS_LOCAL_SPEC.audio_vocab_size} rows, "
                f"got {tensor.shape[0]}."
            )
        return tensor
    if normalized.startswith("audio_lm_heads.") and normalized.endswith(".weight"):
        tensor = reader.get_tensor(normalized)
        expected = (
            MOSS_TTS_LOCAL_SPEC.audio_vocab_size,
            int(_global_config(config).hidden_size),
        )
        if tuple(tensor.shape) != expected:
            raise ValueError(f"Checkpoint tensor {normalized} has shape {tuple(tensor.shape)}, expected {expected}.")
        return tensor
    if normalized.startswith("local_transformer.") or normalized == "local_text_lm_head.weight":
        return reader.get_tensor(normalized)
    raise KeyError(f"Unsupported MOSS-TTS Local Megatron parameter {name!r}")


class MossTTSLocalCheckpointWeightMapper:
    """Identity and parameter mapping used before native Megatron checkpoints exist."""

    def validate_identity(self, config: Any) -> None:
        local_config = (
            getattr(config, "gpt2_config", None)
            or getattr(config, "gpt_neox_config", None)
            or getattr(config, "local_config", None)
        )
        local_heads = getattr(
            local_config,
            "n_head",
            getattr(local_config, "num_attention_heads", None),
        )
        local_ffn = getattr(
            local_config,
            "n_inner",
            getattr(local_config, "intermediate_size", None),
        )
        local_rope = getattr(local_config, "rope_base", None)
        if local_rope is None:
            rope_parameters = getattr(local_config, "rope_parameters", None)
            if isinstance(rope_parameters, dict):
                local_rope = rope_parameters.get("rope_theta")
            else:
                local_rope = getattr(rope_parameters, "rope_theta", None)
        local_epsilon = getattr(
            local_config,
            "layer_norm_epsilon",
            getattr(local_config, "layer_norm_eps", None),
        )
        values = {
            "n_vq": getattr(config, "n_vq", None),
            "audio_vocab_size": getattr(config, "audio_vocab_size", None),
            "audio_pad_code": getattr(config, "audio_pad_code", None),
            "audio_start_token_id": getattr(config, "audio_start_token_id", None),
            "audio_user_slot_token_id": getattr(config, "audio_user_slot_token_id", None),
            "audio_assistant_slot_token_id": getattr(config, "audio_assistant_slot_token_id", None),
            "audio_end_token_id": getattr(config, "audio_end_token_id", None),
            "local_layers": getattr(
                config,
                "local_transformer_layers",
                getattr(local_config, "n_layer", getattr(local_config, "num_hidden_layers", None)),
            ),
            "local_num_attention_heads": local_heads,
            "local_ffn_hidden_size": local_ffn,
            "local_rope_base": local_rope,
            "local_layer_norm_epsilon": local_epsilon,
        }
        expected = MOSS_TTS_LOCAL_SPEC.identity()
        for name, value in values.items():
            if value is None:
                raise ValueError(f"MOSS-TTS Local checkpoint config is missing {name!r}.")
            if value != expected[name]:
                raise ValueError(f"MOSS-TTS Local checkpoint {name} mismatch: expected {expected[name]}, got {value}.")
        tie_audio = getattr(
            config,
            "tie_audio_embeddings_and_output_weights",
            getattr(config, "tie_audio_embeddings", None),
        )
        if tie_audio is not False:
            raise ValueError("mossLite MOSS-TTS Local requires independent audio embeddings and output heads.")

    def get_tensor(self, name: str, reader, config) -> torch.Tensor:
        return moss_tts_local_hf_tensor(name, reader, config)
