"""Strict model-only loader for mossLite logical Megatron checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import MutableMapping, Sequence
from pathlib import Path

import torch
from megatron.core import dist_checkpointing, mpu
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedBase
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
)
from megatron.core.dist_checkpointing.validation import StrictHandling
from miles.policies.moss_tts_local.spec import MOSS_TTS_LOCAL_SPEC

logger = logging.getLogger(__name__)


def _load_manifest(iteration_root: Path) -> dict[str, object]:
    manifest_path = iteration_root / "checkpoint.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f"MOSS-TTS Local checkpoint manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"MOSS-TTS Local checkpoint manifest is invalid: {manifest_path}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ValueError("MOSS-TTS Local pretrained checkpoint requires logical manifest " "schema_version=2.")
    return manifest


def _validate_manifest_identity(manifest: dict[str, object]) -> None:
    identity = manifest.get("model_identity")
    if not isinstance(identity, dict) or identity.get("architecture") != "MOSS-TTS-Local":
        raise ValueError("Pretrained checkpoint is not a MOSS-TTS Local checkpoint.")
    tensor_schema = identity.get("tensor_schema")
    semantics = identity.get("forward_semantics")
    if not isinstance(tensor_schema, dict) or not isinstance(semantics, dict):
        raise ValueError("MOSS-TTS Local checkpoint has no complete model identity.")

    expected_schema = {
        "embedding_head_storage": MOSS_TTS_LOCAL_SPEC.embedding_head_storage,
        "num_layers": MOSS_TTS_LOCAL_SPEC.global_layers,
        "hidden_size": MOSS_TTS_LOCAL_SPEC.hidden_size,
        "num_attention_heads": MOSS_TTS_LOCAL_SPEC.global_num_attention_heads,
        "num_query_groups": MOSS_TTS_LOCAL_SPEC.global_num_query_groups,
        "kv_channels": 128,
        "ffn_hidden_size": MOSS_TTS_LOCAL_SPEC.global_ffn_hidden_size,
        "text_vocab_size": MOSS_TTS_LOCAL_SPEC.text_vocab_size,
        "n_vq": MOSS_TTS_LOCAL_SPEC.n_vq,
        "speech_vocab_size": MOSS_TTS_LOCAL_SPEC.audio_vocab_size,
        "local_num_layers": MOSS_TTS_LOCAL_SPEC.local_layers,
        "local_hidden_size": MOSS_TTS_LOCAL_SPEC.hidden_size,
        "local_num_attention_heads": MOSS_TTS_LOCAL_SPEC.local_num_attention_heads,
        "local_ffn_hidden_size": MOSS_TTS_LOCAL_SPEC.local_ffn_hidden_size,
        "qk_layernorm": MOSS_TTS_LOCAL_SPEC.qk_layernorm,
    }
    expected_semantics = {
        "embedding_head_usage": "untied",
        "audio_end_token_id": MOSS_TTS_LOCAL_SPEC.audio_end_token_id,
        "local_activation": MOSS_TTS_LOCAL_SPEC.local_activation,
        "local_norm_epsilon": MOSS_TTS_LOCAL_SPEC.local_layer_norm_epsilon,
        "local_rope_base": MOSS_TTS_LOCAL_SPEC.local_rope_base,
        "norm_epsilon": MOSS_TTS_LOCAL_SPEC.global_layer_norm_epsilon,
        "rotary_base": MOSS_TTS_LOCAL_SPEC.global_rope_base,
        "rotary_interleaved": False,
    }
    mismatches = {
        f"tensor_schema.{name}": (expected, tensor_schema.get(name))
        for name, expected in expected_schema.items()
        if tensor_schema.get(name) != expected
    }
    mismatches.update(
        {
            f"forward_semantics.{name}": (expected, semantics.get(name))
            for name, expected in expected_semantics.items()
            if semantics.get(name) != expected
        }
    )
    if mismatches:
        raise ValueError("MOSS-TTS Local pretrained checkpoint identity mismatch: " f"{mismatches}")


def _native_key(local_key: str) -> str:
    """Map Miles module keys onto the canonical mossLite DCP tensor schema."""

    if local_key.startswith("language_model.embedding."):
        return "text_embedding." + local_key.removeprefix("language_model.embedding.")
    if local_key.startswith("language_model.decoder."):
        return "decoder." + local_key.removeprefix("language_model.decoder.")
    audio_embedding = re.fullmatch(r"audio_embeddings\.(\d+)\.weight", local_key)
    if audio_embedding:
        return f"audio_embeddings.{audio_embedding.group(1)}." "word_embeddings.weight"
    return local_key


def remap_moss_tts_local_sharded_keys(sharded_state_dict: object) -> set[str]:
    """Retarget only checkpoint storage keys, preserving local load-state keys."""

    storage_keys: set[str] = set()
    for value in nested_values(sharded_state_dict):
        if isinstance(value, ShardedBase):
            value.key = _native_key(value.key)
            storage_keys.add(value.key)
    return storage_keys


def remove_runtime_extra_state(
    sharded_state_dict: MutableMapping[str, object],
) -> set[str]:
    """Remove initialized runtime metadata that mossLite intentionally omits.

    Megatron/Transformer-Engine versions can expose ``_extra_state`` entries
    for layer norms, core attention, and even plain projections.  The mossLite
    logical model schema contains trainable tensors only.  These entries carry
    no BF16 policy weights, so preserve their freshly initialized module state
    instead of weakening DCP strictness for real tensors.
    """

    extra_state_keys = {key for key in sharded_state_dict if key.endswith("._extra_state")}
    for key in extra_state_keys:
        del sharded_state_dict[key]
    return extra_state_keys


def validate_dcp_mismatches(checkpoint_only_keys: set[str], model_only_keys: set[str]) -> None:
    """Allow checkpoint-only runtime objects, but no tensor-schema mismatch."""

    invalid_checkpoint_only = {key for key in checkpoint_only_keys if not key.endswith("._extra_state")}
    if invalid_checkpoint_only or model_only_keys:
        raise ValueError(
            "MOSS-TTS Local strict DCP schema mismatch: "
            f"checkpoint_only={sorted(checkpoint_only_keys)}, "
            f"model_only={sorted(model_only_keys)}"
        )


def _validate_native_load_state(incompatible, runtime_extra_state_keys: set[str]) -> None:
    # Only accept the exact runtime keys removed before DCP loading. Real
    # tensors and any other missing or unexpected state remain strict failures.
    missing = set(incompatible.missing_keys) - runtime_extra_state_keys
    unexpected = set(incompatible.unexpected_keys)
    if missing or unexpected:
        raise ValueError(
            "MOSS-TTS Local native load-state mismatch after strict DCP load: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
            f"excluded_runtime_extra_state={sorted(runtime_extra_state_keys)}"
        )


def load_moss_tts_local_pretrained_checkpoint(
    args: argparse.Namespace,
    model: Sequence[torch.nn.Module],
    checkpointing_context: MutableMapping[str, object],
) -> None:
    """Load the latest selected mossLite model component, excluding train state."""

    if args.ckpt_format != "torch_dist":
        raise ValueError("MOSS-TTS Local pretrained checkpoints require --ckpt-format torch_dist.")
    if len(model) != 1:
        raise ValueError("MOSS-TTS Local pretrained loading does not support virtual pipeline chunks.")

    from megatron.training.checkpointing import get_load_checkpoint_path_by_args

    iteration_root = Path(get_load_checkpoint_path_by_args(args, load_arg="pretrained_checkpoint"))
    manifest = _load_manifest(iteration_root)
    _validate_manifest_identity(manifest)
    model_checkpoint = iteration_root / "model"
    model_checkpoint_path = str(model_checkpoint)
    if not model_checkpoint.is_dir() or not (model_checkpoint / ".metadata").is_file():
        raise FileNotFoundError("MOSS-TTS Local pretrained model payload is incomplete: " f"{model_checkpoint}")
    if not dist_checkpointing.check_is_distributed_checkpoint(model_checkpoint_path):
        raise ValueError(
            "MOSS-TTS Local pretrained model payload is not a torch_dist " f"checkpoint: {model_checkpoint}"
        )

    logger.info(
        "Loading mossLite MOSS-TTS Local model-only checkpoint from %s",
        model_checkpoint,
    )
    content_metadata = dist_checkpointing.load_content_metadata(model_checkpoint_path) or {}
    local_state = model[0].sharded_state_dict(metadata=content_metadata)
    runtime_extra_state_keys = remove_runtime_extra_state(local_state)
    storage_keys = remap_moss_tts_local_sharded_keys(local_state)
    required_storage_keys = {
        "text_embedding.word_embeddings.weight",
        "decoder.final_layernorm.weight",
        "local_text_lm_head.weight",
        *(f"audio_embeddings.{index}.word_embeddings.weight" for index in range(MOSS_TTS_LOCAL_SPEC.n_vq)),
        *(f"audio_lm_heads.{index}.weight" for index in range(MOSS_TTS_LOCAL_SPEC.n_vq)),
    }
    missing = sorted(required_storage_keys - storage_keys)
    if missing:
        raise ValueError("Miles MOSS-TTS Local model cannot represent required native tensors: " f"{missing}")

    load_strategy = get_default_load_sharded_strategy(model_checkpoint_path)
    if args.ckpt_fully_parallel_load:
        load_strategy = FullyParallelLoadStrategyWrapper(
            load_strategy,
            mpu.get_data_parallel_group(with_context_parallel=True),
        )
    checkpointing_context["load_strategy"] = load_strategy
    loaded_state, checkpoint_only_keys, model_only_keys = dist_checkpointing.load(
        {"model": local_state},
        model_checkpoint_path,
        sharded_strategy=load_strategy,
        strict=StrictHandling.RETURN_ALL,
    )
    validate_dcp_mismatches(checkpoint_only_keys, model_only_keys)
    incompatible = model[0].load_state_dict(loaded_state["model"], strict=False)
    _validate_native_load_state(incompatible, runtime_extra_state_keys)


__all__ = [
    "load_moss_tts_local_pretrained_checkpoint",
    "remap_moss_tts_local_sharded_keys",
    "remove_runtime_extra_state",
    "validate_dcp_mismatches",
]
