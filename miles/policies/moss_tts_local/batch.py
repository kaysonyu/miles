"""MOSS-TTS Local micro-batch construction and causal alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from megatron.core.packed_seq_params import PackedSeqParams

from miles.policies.moss_tts_local.spec import MOSS_TTS_LOCAL_SPEC

MOSS_TTS_LOCAL_BATCH_KEYS = (
    "prompt_rows",
    "decisions",
    "decision_masks",
    "codes",
    "code_masks",
    "rollout_decision_logprobs",
    "rollout_code_logprobs",
    "rewards",
    "rollout_ids",
    "weight_versions",
)


@dataclass
class MossTTSLocalPolicyBatch:
    global_rows: torch.Tensor
    global_position_ids: torch.Tensor
    global_lengths: torch.Tensor
    packed_seq_params: PackedSeqParams | None
    prediction_positions: torch.Tensor
    decisions: torch.Tensor
    decision_mask: torch.Tensor
    codes: torch.Tensor
    code_mask: torch.Tensor
    rollout_decision_logprobs: torch.Tensor
    rollout_code_logprobs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor | None
    old_joint_logprobs: torch.Tensor | None
    decision_offsets: torch.Tensor
    frame_offsets: torch.Tensor
    rollout_ids: torch.Tensor
    weight_versions: list[str]

    @property
    def batch_size(self) -> int:
        return int(self.decision_offsets.numel() - 1)

    def server_joint_logprobs(self) -> torch.Tensor:
        joint = self.rollout_decision_logprobs.clone()
        if self.batch_size == 1:
            num_frames = int(self.rollout_code_logprobs.shape[0])
            if num_frames:
                joint[:num_frames] += (self.rollout_code_logprobs * self.code_mask).sum(dim=-1)
            return joint
        for batch_index in range(self.batch_size):
            decision_start = int(self.decision_offsets[batch_index])
            frame_start = int(self.frame_offsets[batch_index])
            frame_end = int(self.frame_offsets[batch_index + 1])
            num_frames = frame_end - frame_start
            if num_frames:
                code_joint = (
                    self.rollout_code_logprobs[frame_start:frame_end] * self.code_mask[frame_start:frame_end]
                ).sum(dim=-1)
                joint[decision_start : decision_start + num_frames] += code_joint
        return joint


def _require_list(raw_batch: dict[str, Any], key: str) -> list[Any]:
    values = raw_batch.get(key)
    if not isinstance(values, list):
        raise TypeError(f"MOSS-TTS Local micro-batch field {key!r} must be a list, got {type(values).__name__}.")
    return values


def _build_packed_seq_params(global_lengths: list[int], *, device: torch.device) -> PackedSeqParams:
    """Describe the concatenated global rows as independent causal THD sequences."""

    cu_seqlens = torch.zeros(len(global_lengths) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.tensor(global_lengths, dtype=torch.int32, device=device).cumsum(dim=0)
    max_seqlen = max(global_lengths)
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )


def collate_moss_tts_local_batch(
    raw_batch: dict[str, Any],
    *,
    packed_thd: bool = False,
) -> MossTTSLocalPolicyBatch:
    """Pack ragged samples while preserving global-time and local-depth geometry.

    ``global_rows`` remains one physical ``[1, total_rows, channels]`` stream so
    Megatron can execute variable-size micro-batches.  When ``packed_thd`` is
    enabled, ``packed_seq_params`` is the authoritative sample-boundary
    description used by both RoPE and attention.  Without it, only singleton
    batches are safe to execute.
    """

    fields = {key: _require_list(raw_batch, key) for key in MOSS_TTS_LOCAL_BATCH_KEYS}
    batch_size = len(fields["prompt_rows"])
    if batch_size == 0:
        raise ValueError("MOSS-TTS Local micro-batch cannot be empty.")
    if any(len(values) != batch_size for values in fields.values()):
        raise ValueError("MOSS-TTS Local micro-batch fields have inconsistent sample counts.")

    global_rows = []
    position_ids = []
    prediction_positions = []
    decisions = []
    decision_masks = []
    codes = []
    code_masks = []
    rollout_decision_logprobs = []
    rollout_code_logprobs = []
    decision_offsets = [0]
    frame_offsets = [0]
    global_lengths = []

    global_offset = 0
    for sample_index in range(batch_size):
        prompt = fields["prompt_rows"][sample_index]
        sample_codes = fields["codes"][sample_index]
        sample_decisions = fields["decisions"][sample_index]
        if prompt.ndim != 2 or tuple(prompt.shape[1:]) != (MOSS_TTS_LOCAL_SPEC.channels,):
            raise ValueError(
                f"MOSS-TTS Local prompt_rows[{sample_index}] must be [P, {MOSS_TTS_LOCAL_SPEC.channels}]."
            )
        if sample_codes.ndim != 2 or tuple(sample_codes.shape[1:]) != (MOSS_TTS_LOCAL_SPEC.n_vq,):
            raise ValueError(f"MOSS-TTS Local codes[{sample_index}] must be [T, {MOSS_TTS_LOCAL_SPEC.n_vq}].")
        num_prompt_rows = int(prompt.shape[0])
        num_frames = int(sample_codes.shape[0])
        num_decisions = int(sample_decisions.numel())
        if num_prompt_rows <= 0:
            raise ValueError("MOSS-TTS Local prompt must contain at least one row.")
        if num_decisions not in (num_frames, num_frames + 1):
            raise ValueError(
                f"MOSS-TTS Local sample {sample_index} requires R in {{T,T+1}}, "
                f"got R={num_decisions}, T={num_frames}."
            )

        emitted_rows = torch.empty(
            (num_frames, MOSS_TTS_LOCAL_SPEC.channels),
            dtype=torch.long,
            device=prompt.device,
        )
        if num_frames:
            emitted_rows[:, 0] = MOSS_TTS_LOCAL_SPEC.audio_assistant_slot_token_id
            emitted_rows[:, 1:] = sample_codes
        sample_global_rows = torch.cat([prompt, emitted_rows], dim=0)
        global_rows.append(sample_global_rows)
        global_lengths.append(int(sample_global_rows.shape[0]))
        position_ids.append(torch.arange(sample_global_rows.shape[0], device=prompt.device, dtype=torch.long))

        sample_prediction_positions = global_offset + torch.arange(
            num_prompt_rows - 1,
            num_prompt_rows - 1 + num_decisions,
            device=prompt.device,
            dtype=torch.long,
        )
        if sample_prediction_positions.numel() and int(sample_prediction_positions[-1]) >= (
            global_offset + sample_global_rows.shape[0]
        ):
            raise ValueError("MOSS-TTS Local decision prediction position exceeds the global row sequence.")
        prediction_positions.append(sample_prediction_positions)
        global_offset += int(sample_global_rows.shape[0])

        decisions.append(sample_decisions.reshape(-1))
        decision_masks.append(fields["decision_masks"][sample_index].reshape(-1))
        codes.append(sample_codes)
        code_masks.append(fields["code_masks"][sample_index])
        rollout_decision_logprobs.append(fields["rollout_decision_logprobs"][sample_index].reshape(-1))
        rollout_code_logprobs.append(fields["rollout_code_logprobs"][sample_index])
        decision_offsets.append(decision_offsets[-1] + num_decisions)
        frame_offsets.append(frame_offsets[-1] + num_frames)

    device = global_rows[0].device
    advantages = raw_batch.get("advantages")
    old_joint_logprobs = raw_batch.get("old_joint_logprobs")
    if advantages is not None:
        if not isinstance(advantages, list) or len(advantages) != batch_size:
            raise ValueError("MOSS-TTS Local advantages must contain one tensor per sample.")
        advantages = torch.cat([value.reshape(-1) for value in advantages], dim=0).float()
    if old_joint_logprobs is not None:
        if not isinstance(old_joint_logprobs, list) or len(old_joint_logprobs) != batch_size:
            raise ValueError("MOSS-TTS Local old_joint_logprobs must contain one tensor per sample.")
        old_joint_logprobs = torch.cat([value.reshape(-1) for value in old_joint_logprobs], dim=0).float()
    packed_seq_params = _build_packed_seq_params(global_lengths, device=device) if packed_thd else None
    return MossTTSLocalPolicyBatch(
        global_rows=torch.cat(global_rows, dim=0).unsqueeze(0),
        global_position_ids=torch.cat(position_ids, dim=0).unsqueeze(0),
        global_lengths=torch.tensor(global_lengths, dtype=torch.int32, device=device),
        packed_seq_params=packed_seq_params,
        prediction_positions=torch.cat(prediction_positions, dim=0),
        decisions=torch.cat(decisions, dim=0).long(),
        decision_mask=torch.cat(decision_masks, dim=0).bool(),
        codes=torch.cat(codes, dim=0).long(),
        code_mask=torch.cat(code_masks, dim=0).bool(),
        rollout_decision_logprobs=torch.cat(rollout_decision_logprobs, dim=0).float(),
        rollout_code_logprobs=torch.cat(rollout_code_logprobs, dim=0).float(),
        rewards=torch.as_tensor(fields["rewards"], dtype=torch.float32, device=device),
        advantages=advantages,
        old_joint_logprobs=old_joint_logprobs,
        decision_offsets=torch.tensor(decision_offsets, dtype=torch.long, device=device),
        frame_offsets=torch.tensor(frame_offsets, dtype=torch.long, device=device),
        rollout_ids=torch.as_tensor(fields["rollout_ids"], dtype=torch.long, device=device),
        weight_versions=[str(version) for version in fields["weight_versions"]],
    )
