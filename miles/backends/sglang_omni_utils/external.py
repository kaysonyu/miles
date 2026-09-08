"""Discovery and identity validation for pre-launched SGLang-Omni pipelines."""

from __future__ import annotations

import dataclasses
import os
from argparse import Namespace
from typing import Any

from miles.backends.sglang_omni_utils.http_adapter import SGLangOmniHttpAdapter, normalize_omni_base_url
from miles.policies.moss_tts_local.spec import MOSS_TTS_LOCAL_SPEC


@dataclasses.dataclass(frozen=True)
class ExternalOmniEngineInfo:
    base_url: str
    train_stage: str
    tp_size: int
    weight_version: str
    model_identity: dict[str, Any]
    capabilities: dict[str, Any] = dataclasses.field(default_factory=dict)
    model_info: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def parallel_config(self) -> dict[str, int]:
        return {"tp_size": self.tp_size, "pp_size": 1, "ep_size": 1, "moe_dp_size": 1}

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _stage_data(model_info: dict[str, Any], train_stage: str) -> dict[str, Any]:
    for item in model_info.get("stages", model_info.get("results", [])) or []:
        if not isinstance(item, dict) or item.get("stage") != train_stage:
            continue
        data = item.get("data")
        if isinstance(data, dict) and not data.get("skipped") and not data.get("unsupported"):
            return data
    raise ValueError(f"SGLang-Omni /model_info did not return active stage {train_stage!r}: {model_info!r}")


def discover_external_omni_engines(
    endpoints: list[str],
    *,
    train_stage: str = "tts_engine",
    admin_api_key: str | None = None,
) -> list[ExternalOmniEngineInfo]:
    infos = []
    for endpoint in endpoints:
        base_url = normalize_omni_base_url(endpoint)
        adapter = SGLangOmniHttpAdapter(base_url, admin_api_key=admin_api_key)
        try:
            health = adapter.health()
            if not health.get("running", health.get("status") == "healthy"):
                raise RuntimeError(f"SGLang-Omni endpoint {base_url} is not healthy: {health!r}")
            model_info = adapter.model_info(stages=[train_stage])
        finally:
            adapter.close()

        stage_data = _stage_data(model_info, train_stage)
        identity = stage_data.get("model_identity") or {}
        MOSS_TTS_LOCAL_SPEC.validate_identity(identity, require_hash=True)
        schema_versions = stage_data.get("rollout_schema_versions") or []
        if MOSS_TTS_LOCAL_SPEC.rollout_schema_version not in schema_versions:
            raise ValueError(
                f"SGLang-Omni endpoint {base_url} does not support rollout schema "
                f"{MOSS_TTS_LOCAL_SPEC.rollout_schema_version}: {schema_versions!r}."
            )
        semantics = stage_data.get("logprob_semantics")
        if semantics != MOSS_TTS_LOCAL_SPEC.logprob_semantics:
            raise ValueError(
                f"SGLang-Omni endpoint {base_url} logprob semantics mismatch: "
                f"expected {MOSS_TTS_LOCAL_SPEC.logprob_semantics!r}, got {semantics!r}."
            )
        if not bool(stage_data.get("supports_weight_update", False)):
            raise ValueError(f"SGLang-Omni endpoint {base_url} stage {train_stage!r} cannot update weights.")
        if not bool(stage_data.get("supports_weight_checker", False)):
            raise ValueError(f"SGLang-Omni endpoint {base_url} stage {train_stage!r} has no strict weight checker.")
        weight_version = stage_data.get("weight_version", model_info.get("weight_version"))
        if weight_version is None:
            raise ValueError(f"SGLang-Omni endpoint {base_url} did not report a weight_version.")
        tp_size = int(stage_data.get("stage_tp_size") or stage_data.get("tp_size") or 1)
        if tp_size <= 0:
            raise ValueError(f"SGLang-Omni endpoint {base_url} reported invalid tp_size={tp_size}.")
        infos.append(
            ExternalOmniEngineInfo(
                base_url=base_url,
                train_stage=train_stage,
                tp_size=tp_size,
                weight_version=str(weight_version),
                model_identity=dict(identity),
                capabilities={
                    "rollout_schema_versions": list(schema_versions),
                    "logprob_semantics": semantics,
                    "supports_weight_update": bool(stage_data.get("supports_weight_update", False)),
                    "supports_weight_checker": bool(stage_data.get("supports_weight_checker", False)),
                },
                model_info=model_info,
            )
        )
    if len({info.weight_version for info in infos}) > 1:
        raise ValueError(
            f"SGLang-Omni endpoints have mixed weight versions: {[info.weight_version for info in infos]}"
        )
    if len({info.model_identity.get("config_sha256") for info in infos}) > 1:
        raise ValueError("SGLang-Omni endpoints have mixed model config hashes.")
    return infos


def apply_external_omni_info_to_args(args: Namespace, logger=None) -> None:
    endpoints = getattr(args, "sglang_omni_endpoints", None)
    if not endpoints:
        raise ValueError("--sglang-omni-endpoints is required for --rollout-backend=sglang_omni.")
    key_env = getattr(args, "sglang_omni_admin_api_key_env", None)
    admin_api_key = os.getenv(key_env) if key_env else None
    if key_env and admin_api_key is None:
        raise ValueError(f"SGLang-Omni admin key environment variable {key_env!r} is not set.")
    train_stage = getattr(args, "sglang_omni_train_stage", "tts_engine")
    infos = discover_external_omni_engines(
        endpoints,
        train_stage=train_stage,
        admin_api_key=admin_api_key,
    )
    args.sglang_omni_endpoint_infos = [info.to_dict() for info in infos]
    args.rollout_num_engines = len(infos)
    args.rollout_num_gpus = sum(info.tp_size for info in infos)
    args.rollout_external = True
    if logger is not None:
        logger.info(
            "Detected external SGLang-Omni endpoints: %s",
            [
                {
                    "base_url": info.base_url,
                    "train_stage": info.train_stage,
                    "tp_size": info.tp_size,
                    "weight_version": info.weight_version,
                }
                for info in infos
            ],
        )
