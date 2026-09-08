"""Strict decoder for SGLang-Omni MOSS-TTS Local rollout schema v2."""

from __future__ import annotations

from typing import Any

import torch

from miles.policies.moss_tts_local.spec import MOSS_TTS_LOCAL_SPEC
from miles.policies.moss_tts_local.types import MossTTSLocalTrajectoryV2


def _stream_by_name(trace: dict[str, Any], name: str) -> dict[str, Any]:
    streams = trace.get("action_streams")
    if not isinstance(streams, list):
        raise ValueError("MOSS-TTS Local omni_rollout.action_streams must be a list.")
    matches = [stream for stream in streams if isinstance(stream, dict) and stream.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"MOSS-TTS Local omni_rollout requires exactly one {name!r} action stream.")
    return matches[0]


def _required_stream_tensor(stream: dict[str, Any], key: str, *, dtype: torch.dtype) -> torch.Tensor:
    value = stream.get(key)
    if value is None:
        raise ValueError(f"MOSS-TTS Local action stream {stream.get('name')!r} is missing {key!r}.")
    return torch.as_tensor(value, dtype=dtype).detach().cpu().contiguous()


def decode_moss_tts_local_trace(
    trace: dict[str, Any],
    *,
    meta_info: dict[str, Any],
) -> MossTTSLocalTrajectoryV2:
    """Decode and validate one v2 trace from ``meta_info.omni_rollout``."""

    if not isinstance(trace, dict):
        raise TypeError(f"MOSS-TTS Local omni_rollout must be a dict, got {type(trace).__name__}.")
    replay_inputs = trace.get("replay_inputs")
    if not isinstance(replay_inputs, dict):
        raise ValueError("MOSS-TTS Local omni_rollout is missing replay_inputs.")
    if replay_inputs.get("layout") not in (None, "moss_local_rows"):
        raise ValueError(f"Unsupported MOSS-TTS Local replay layout {replay_inputs.get('layout')!r}.")
    prompt_rows = replay_inputs.get("prompt_rows")
    if prompt_rows is None:
        raise ValueError("MOSS-TTS Local replay_inputs is missing exact prompt_rows.")

    decision_stream = _stream_by_name(trace, "decision")
    code_stream = _stream_by_name(trace, "codes")
    if decision_stream.get("layout") not in (None, "time_1d"):
        raise ValueError(f"Unsupported decision layout {decision_stream.get('layout')!r}.")
    if code_stream.get("layout") not in (None, "time_depth_autoregressive"):
        raise ValueError(f"Unsupported codes layout {code_stream.get('layout')!r}.")
    if int(decision_stream.get("vocab_size", 0)) != 2:
        raise ValueError("MOSS-TTS Local decision stream must declare vocab_size=2.")
    if int(code_stream.get("vocab_size", 0)) != MOSS_TTS_LOCAL_SPEC.audio_vocab_size:
        raise ValueError(f"MOSS-TTS Local code stream must declare vocab_size={MOSS_TTS_LOCAL_SPEC.audio_vocab_size}.")

    decisions = _required_stream_tensor(decision_stream, "actions", dtype=torch.long).reshape(-1)
    decision_logprobs = _required_stream_tensor(decision_stream, "logprobs", dtype=torch.float32).reshape(-1)
    decision_mask = _required_stream_tensor(decision_stream, "action_mask", dtype=torch.bool).reshape(-1)
    codes = _required_stream_tensor(code_stream, "actions", dtype=torch.long).reshape(-1, MOSS_TTS_LOCAL_SPEC.n_vq)
    code_logprobs = _required_stream_tensor(code_stream, "logprobs", dtype=torch.float32).reshape(
        -1, MOSS_TTS_LOCAL_SPEC.n_vq
    )
    code_mask = _required_stream_tensor(code_stream, "action_mask", dtype=torch.bool).reshape(
        -1, MOSS_TTS_LOCAL_SPEC.n_vq
    )

    finish = trace.get("finish_reason")
    meta_finish = meta_info.get("finish_reason")
    if isinstance(meta_finish, dict):
        meta_finish = meta_finish.get("type")
    if finish is None:
        finish = meta_finish
    if finish != meta_finish:
        raise ValueError(f"MOSS-TTS Local trace/meta finish reason mismatch: trace={finish!r}, meta={meta_finish!r}.")

    trace_version = trace.get("admission_weight_version")
    meta_version = meta_info.get("weight_version")
    if trace_version is None or meta_version is None:
        raise ValueError("MOSS-TTS Local trace and meta_info must both report a weight version.")
    if str(trace_version) != str(meta_version):
        raise ValueError(
            f"MOSS-TTS Local trace/meta weight version mismatch: trace={trace_version!r}, meta={meta_version!r}."
        )

    declared_actions = trace.get("total_action_count")
    actual_actions = int(decision_mask.sum().item() + code_mask.sum().item())
    if declared_actions is not None and int(declared_actions) != actual_actions:
        raise ValueError(
            f"MOSS-TTS Local total_action_count mismatch: declared={declared_actions}, actual={actual_actions}."
        )

    request_metadata = meta_info.get("request_metadata")
    request_id = trace.get("request_id")
    if request_id is None and isinstance(request_metadata, dict):
        request_id = request_metadata.get("request_id")

    sampling = dict(trace.get("sampling") or {})
    sampling.setdefault("logprob_semantics", trace.get("logprob_semantics"))
    trajectory = MossTTSLocalTrajectoryV2(
        version=int(trace.get("version", -1)),
        model_family=str(trace.get("model_family", "")),
        prompt_rows=torch.as_tensor(prompt_rows, dtype=torch.long).detach().cpu().contiguous(),
        decisions=decisions,
        decision_logprobs=decision_logprobs,
        decision_mask=decision_mask,
        codes=codes,
        code_logprobs=code_logprobs,
        code_mask=code_mask,
        finish_reason=str(finish),
        sampling=sampling,
        model_identity=dict(trace.get("model_identity") or {}),
        weight_version=str(trace_version),
        request_id=str(request_id or ""),
    )
    trajectory.validate()
    return trajectory
