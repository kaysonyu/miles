"""Canonical identity for the mossLite MOSS-TTS Local policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MossTTSLocalPolicySpec:
    policy_family: str = "moss_tts_local"
    architecture: str = "MOSS-TTS-Local"
    rollout_schema_version: int = 2
    logprob_semantics: str = "temperature_scaled_full_vocab_v1"
    train_stage: str = "tts_engine"
    n_vq: int = 12
    audio_vocab_size: int = 1024
    audio_pad_code: int = 1024
    audio_start_token_id: int = 151652
    audio_user_slot_token_id: int = 151654
    audio_assistant_slot_token_id: int = 151656
    audio_end_token_id: int = 151653
    text_vocab_size: int = 151936
    hidden_size: int = 2560
    global_layers: int = 36
    global_num_attention_heads: int = 32
    global_num_query_groups: int = 8
    global_ffn_hidden_size: int = 9728
    global_rope_base: float = 1_000_000.0
    global_layer_norm_epsilon: float = 1e-6
    qk_layernorm: bool = True
    local_layers: int = 1
    local_num_attention_heads: int = 32
    local_ffn_hidden_size: int = 9728
    local_rope_base: float = 1_000_000.0
    local_layer_norm_epsilon: float = 1e-6
    local_activation: str = "silu"
    tie_audio_embeddings_and_output_weights: bool = False
    embedding_head_storage: str = "split_v1"
    sample_rate: int = 48000

    @property
    def channels(self) -> int:
        return self.n_vq + 1

    def identity(self) -> dict[str, Any]:
        policy_fields = {
            "rollout_schema_version",
            "logprob_semantics",
            "train_stage",
        }
        return {key: value for key, value in asdict(self).items() if key not in policy_fields}

    def identity_sha256(self) -> str:
        payload = json.dumps(self.identity(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def validate_identity(self, identity: dict[str, Any], *, require_hash: bool = False) -> None:
        if not isinstance(identity, dict):
            raise TypeError(f"MOSS-TTS Local model_identity must be a dict, got {type(identity).__name__}.")

        expected = self.identity()
        aliases = {
            "model_family": "policy_family",
            "n_audio_codebooks": "n_vq",
        }
        normalized = dict(identity)
        for source, target in aliases.items():
            if source in normalized and target not in normalized:
                normalized[target] = normalized[source]

        for key in expected:
            if key not in normalized:
                raise ValueError(f"MOSS-TTS Local model_identity is missing required field {key!r}.")
            if normalized[key] != expected[key]:
                raise ValueError(
                    f"MOSS-TTS Local identity mismatch for {key}: expected {expected[key]!r}, "
                    f"got {normalized[key]!r}."
                )

        config_hash = normalized.get("config_sha256")
        if require_hash and not config_hash:
            raise ValueError("MOSS-TTS Local model_identity is missing config_sha256.")
        if config_hash is not None:
            if not isinstance(config_hash, str):
                raise TypeError("MOSS-TTS Local config_sha256 must be a string when present.")
            expected_hash = self.identity_sha256()
            if config_hash != expected_hash:
                raise ValueError(
                    "MOSS-TTS Local config_sha256 mismatch: " f"expected {expected_hash}, got {config_hash}."
                )


MOSS_TTS_LOCAL_SPEC = MossTTSLocalPolicySpec()

# Kept as a source-compatibility alias for integrations compiled against the
# first Miles adapter.  Its value is the current mossLite contract, not the
# retired v1.5 release contract.
MOSS_TTS_LOCAL_V1_5_SPEC = MOSS_TTS_LOCAL_SPEC
