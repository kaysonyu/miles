"""Validated structured trajectory and audio artifact types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from miles.policies.media import MediaArtifact as MediaArtifact
from miles.policies.moss_tts_local.spec import MOSS_TTS_LOCAL_SPEC, MossTTSLocalPolicySpec


def _cpu_tensor(value: Any, *, dtype: torch.dtype, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} cannot be converted to {dtype}: {exc}") from exc
    return tensor.detach().cpu().contiguous()


def _clone_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous().clone()


@dataclass
class MossTTSLocalTrajectoryV2:
    version: Literal[2]
    model_family: Literal["moss_tts_local"]
    prompt_rows: torch.Tensor
    decisions: torch.Tensor
    decision_logprobs: torch.Tensor
    decision_mask: torch.Tensor
    codes: torch.Tensor
    code_logprobs: torch.Tensor
    code_mask: torch.Tensor
    finish_reason: Literal["stop", "length"]
    sampling: dict[str, Any]
    model_identity: dict[str, Any]
    weight_version: str
    request_id: str

    def validate(self, spec: MossTTSLocalPolicySpec = MOSS_TTS_LOCAL_SPEC) -> None:
        if self.version != spec.rollout_schema_version:
            raise ValueError(
                f"Unsupported MOSS-TTS Local trajectory version {self.version}; "
                f"expected {spec.rollout_schema_version}."
            )
        if self.model_family != spec.policy_family:
            raise ValueError(
                f"MOSS-TTS Local model_family mismatch: expected {spec.policy_family!r}, "
                f"got {self.model_family!r}."
            )
        if self.prompt_rows.dtype != torch.long or self.prompt_rows.ndim != 2:
            raise ValueError(
                "MOSS-TTS Local prompt_rows must be int64 [P, channels], "
                f"got dtype={self.prompt_rows.dtype}, shape={tuple(self.prompt_rows.shape)}."
            )
        if self.prompt_rows.shape[0] <= 0 or self.prompt_rows.shape[1] != spec.channels:
            raise ValueError(
                f"MOSS-TTS Local prompt_rows must have shape [P>0, {spec.channels}], "
                f"got {tuple(self.prompt_rows.shape)}."
            )
        text_vocab_size = int(self.model_identity.get("text_vocab_size", 0) or 0)
        if text_vocab_size and not bool(
            ((self.prompt_rows[:, 0] >= 0) & (self.prompt_rows[:, 0] < text_vocab_size)).all()
        ):
            raise ValueError(f"MOSS-TTS Local prompt text ids must be in [0, {text_vocab_size - 1}].")
        prompt_audio = self.prompt_rows[:, 1:]
        if prompt_audio.numel() and not bool(((prompt_audio >= 0) & (prompt_audio <= spec.audio_pad_code)).all()):
            raise ValueError(f"MOSS-TTS Local prompt audio ids must be in [0, {spec.audio_pad_code}].")
        if self.decisions.dtype != torch.long or self.decisions.ndim != 1:
            raise ValueError("MOSS-TTS Local decisions must be a one-dimensional int64 tensor.")
        if self.decision_logprobs.dtype != torch.float32 or self.decision_logprobs.shape != self.decisions.shape:
            raise ValueError("decision_logprobs must be fp32 and shape-aligned with decisions.")
        if self.decision_mask.dtype != torch.bool or self.decision_mask.shape != self.decisions.shape:
            raise ValueError("decision_mask must be bool and shape-aligned with decisions.")
        if self.codes.dtype != torch.long or self.codes.ndim != 2 or self.codes.shape[1] != spec.n_vq:
            raise ValueError(f"MOSS-TTS Local codes must be int64 [T, {spec.n_vq}].")
        if self.code_logprobs.dtype != torch.float32 or self.code_logprobs.shape != self.codes.shape:
            raise ValueError("code_logprobs must be fp32 and shape-aligned with codes.")
        if self.code_mask.dtype != torch.bool or self.code_mask.shape != self.codes.shape:
            raise ValueError("code_mask must be bool and shape-aligned with codes.")

        if not bool(self.decision_mask.all()):
            raise ValueError("MOSS-TTS Local v2 requires every returned decision to be trainable.")
        if self.code_mask.numel() and not bool(self.code_mask.all()):
            raise ValueError("MOSS-TTS Local v2 requires every emitted code to be trainable.")
        if self.decisions.numel() and not bool(((self.decisions == 0) | (self.decisions == 1)).all()):
            raise ValueError("MOSS-TTS Local decisions must contain only 0=continue or 1=stop.")
        if self.codes.numel() and not bool(((self.codes >= 0) & (self.codes < spec.audio_vocab_size)).all()):
            raise ValueError(f"MOSS-TTS Local codes must be in [0, {spec.audio_vocab_size - 1}].")
        if not bool(torch.isfinite(self.decision_logprobs[self.decision_mask]).all()):
            raise ValueError("MOSS-TTS Local decision_logprobs contain non-finite values.")
        if self.code_logprobs.numel() and not bool(torch.isfinite(self.code_logprobs[self.code_mask]).all()):
            raise ValueError("MOSS-TTS Local code_logprobs contain non-finite values.")

        num_frames = int(self.codes.shape[0])
        num_decisions = int(self.decisions.shape[0])
        if self.finish_reason == "stop":
            if num_decisions != num_frames + 1:
                raise ValueError(f"Natural stop requires R=T+1, got decisions={num_decisions}, frames={num_frames}.")
            if num_decisions == 0 or int(self.decisions[-1]) != 1:
                raise ValueError("Natural stop requires a final stop decision (1).")
            if num_frames and not bool((self.decisions[:-1] == 0).all()):
                raise ValueError("Every emitted MOSS-TTS Local frame must have a continue decision (0).")
        elif self.finish_reason == "length":
            if num_decisions != num_frames:
                raise ValueError(
                    f"Length truncation requires R=T, got decisions={num_decisions}, frames={num_frames}."
                )
            if num_decisions and not bool((self.decisions == 0).all()):
                raise ValueError("Length-truncated trajectories cannot contain a stop decision.")
        else:
            raise ValueError(f"Unsupported MOSS-TTS Local finish_reason {self.finish_reason!r}.")

        if not isinstance(self.sampling, dict):
            raise TypeError("MOSS-TTS Local sampling metadata must be a dict.")
        semantics = self.sampling.get("logprob_semantics")
        if semantics is not None and semantics != spec.logprob_semantics:
            raise ValueError(
                f"MOSS-TTS Local logprob semantics mismatch: expected {spec.logprob_semantics!r}, "
                f"got {semantics!r}."
            )
        spec.validate_identity(self.model_identity)
        if not isinstance(self.weight_version, str) or not self.weight_version:
            raise ValueError("MOSS-TTS Local trajectory requires a non-empty string weight_version.")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("MOSS-TTS Local trajectory requires a non-empty request_id.")

    @property
    def num_frames(self) -> int:
        return int(self.codes.shape[0])

    @property
    def num_decisions(self) -> int:
        return int(self.decisions.shape[0])

    @property
    def num_actions(self) -> int:
        return int(self.decision_mask.sum().item() + self.code_mask.sum().item())

    def server_joint_logprobs(self) -> torch.Tensor:
        self.validate()
        joint = self.decision_logprobs.clone()
        if self.num_frames:
            joint[: self.num_frames] += (self.code_logprobs * self.code_mask).sum(dim=-1)
        return joint

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "version": self.version,
            "model_family": self.model_family,
            "prompt_rows": _clone_cpu(self.prompt_rows),
            "decisions": _clone_cpu(self.decisions),
            "decision_logprobs": _clone_cpu(self.decision_logprobs),
            "decision_mask": _clone_cpu(self.decision_mask),
            "codes": _clone_cpu(self.codes),
            "code_logprobs": _clone_cpu(self.code_logprobs),
            "code_mask": _clone_cpu(self.code_mask),
            "finish_reason": self.finish_reason,
            "sampling": dict(self.sampling),
            "model_identity": dict(self.model_identity),
            "weight_version": self.weight_version,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MossTTSLocalTrajectoryV2:
        trajectory = cls(
            version=int(data["version"]),
            model_family=str(data["model_family"]),
            prompt_rows=_cpu_tensor(data["prompt_rows"], dtype=torch.long, name="prompt_rows"),
            decisions=_cpu_tensor(data["decisions"], dtype=torch.long, name="decisions").reshape(-1),
            decision_logprobs=_cpu_tensor(
                data["decision_logprobs"], dtype=torch.float32, name="decision_logprobs"
            ).reshape(-1),
            decision_mask=_cpu_tensor(data["decision_mask"], dtype=torch.bool, name="decision_mask").reshape(-1),
            codes=_cpu_tensor(data["codes"], dtype=torch.long, name="codes").reshape(-1, MOSS_TTS_LOCAL_SPEC.n_vq),
            code_logprobs=_cpu_tensor(data["code_logprobs"], dtype=torch.float32, name="code_logprobs").reshape(
                -1, MOSS_TTS_LOCAL_SPEC.n_vq
            ),
            code_mask=_cpu_tensor(data["code_mask"], dtype=torch.bool, name="code_mask").reshape(
                -1, MOSS_TTS_LOCAL_SPEC.n_vq
            ),
            finish_reason=str(data["finish_reason"]),
            sampling=dict(data.get("sampling") or {}),
            model_identity=dict(data.get("model_identity") or {}),
            weight_version=str(data["weight_version"]),
            request_id=str(data["request_id"]),
        )
        trajectory.validate()
        return trajectory
