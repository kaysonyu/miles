"""Canonical online serving export for mossLite MOSS-TTS Local."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

from miles.backends.megatron_utils.megatron_to_hf.qwen2 import convert_qwen2_to_hf
from miles.policies.moss_tts_local.spec import MOSS_TTS_LOCAL_SPEC


def _strip_wrappers(name: str) -> str:
    while name.startswith("module."):
        name = name.removeprefix("module.")
    return name


class MossTTSLocalServingWeightAdapter:
    """Export canonical checkpoint names consumed by Omni's model ``load_weights``."""

    def expected_names(self, args) -> set[str] | None:
        index = Path(args.hf_checkpoint) / "model.safetensors.index.json"
        if not index.exists():
            return None
        return set(json.loads(index.read_text())["weight_map"]) - {"text_lm_head.weight"}

    def convert_parameter(
        self,
        args,
        model_name: str,
        name: str,
        parameter: torch.Tensor,
        quantization_config: dict[str, Any] | None = None,
        *,
        transform_ue8m0: bool = True,
    ) -> list[tuple[str, torch.Tensor]]:
        del model_name, transform_ue8m0
        if quantization_config:
            raise ValueError("MOSS-TTS Local P0 does not support quantized serving weight export.")
        normalized = _strip_wrappers(name)

        audio_match = re.fullmatch(r"audio_embeddings\.(\d+)\.weight", normalized)
        if audio_match:
            depth = int(audio_match.group(1))
            if not 0 <= depth < MOSS_TTS_LOCAL_SPEC.n_vq:
                raise ValueError(f"Invalid MOSS-TTS Local audio embedding depth {depth}.")
            if parameter.shape[0] < MOSS_TTS_LOCAL_SPEC.audio_vocab_size:
                raise ValueError(f"MOSS-TTS Local audio embedding {depth} has only {parameter.shape[0]} rows.")
            return [
                (
                    f"audio_embeddings.{depth}.weight",
                    parameter[: MOSS_TTS_LOCAL_SPEC.audio_vocab_size].contiguous(),
                )
            ]
        audio_head_match = re.fullmatch(r"audio_lm_heads\.(\d+)\.weight", normalized)
        if audio_head_match:
            depth = int(audio_head_match.group(1))
            if not 0 <= depth < MOSS_TTS_LOCAL_SPEC.n_vq:
                raise ValueError(f"Invalid MOSS-TTS Local audio head depth {depth}.")
            expected = (
                MOSS_TTS_LOCAL_SPEC.audio_vocab_size,
                int(args.hidden_size),
            )
            if tuple(parameter.shape) != expected:
                raise ValueError(
                    f"MOSS-TTS Local audio head {depth} has shape {tuple(parameter.shape)}, " f"expected {expected}."
                )
            return [(f"audio_lm_heads.{depth}.weight", parameter)]
        if normalized.startswith("local_transformer."):
            return [(normalized, parameter)]
        if normalized == "local_text_lm_head.weight":
            return [(normalized, parameter)]
        if normalized.startswith("text_lm_head."):
            return []
        if normalized.startswith("_decode_input_embedding."):
            return []

        # Miles global naming removes language_model only from decoder layers.
        if normalized.startswith("decoder.layers."):
            normalized = "language_model." + normalized
        if normalized.startswith("language_model."):
            qwen_name = "module.module." + normalized.removeprefix("language_model.")
            converted = convert_qwen2_to_hf(args, qwen_name, parameter)
            canonical = []
            for converted_name, converted_parameter in converted:
                if converted_name.startswith("model."):
                    converted_name = "transformer." + converted_name.removeprefix("model.")
                if converted_name.startswith("lm_head."):
                    # The global text head is unused and tied in the public
                    # checkpoint.  Only the authoritative embedding is sent.
                    continue
                canonical.append((converted_name, converted_parameter))
            return canonical
        raise ValueError(f"Unknown MOSS-TTS Local trainer parameter {name!r}.")

    def validate_manifest(self, names: set[str]) -> None:
        required = {
            "transformer.embed_tokens.weight",
            "transformer.norm.weight",
            "local_text_lm_head.weight",
            *(f"audio_embeddings.{depth}.weight" for depth in range(MOSS_TTS_LOCAL_SPEC.n_vq)),
            *(f"audio_lm_heads.{depth}.weight" for depth in range(MOSS_TTS_LOCAL_SPEC.n_vq)),
        }
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"MOSS-TTS Local serving export is missing required tensors: {missing}")
        if not any(name.startswith("transformer.layers.") for name in names):
            raise ValueError("MOSS-TTS Local serving export contains no global transformer layers.")
        if not any(name.startswith("local_transformer.") for name in names):
            raise ValueError("MOSS-TTS Local serving export contains no local transformer tensors.")
