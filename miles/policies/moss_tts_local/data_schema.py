"""One tensor schema for Local rollout transport and training-device preparation.

The wire representation stays a Miles RolloutBatch. Required action fields and
optional paired scores use the same names, dtypes and geometry on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from miles.policies.moss_tts_local.spec import MOSS_TTS_LOCAL_SPEC


@dataclass(frozen=True)
class TensorField:
    name: str
    dtype: torch.dtype
    required: bool = True
    code_scores: bool = False


ACTION_FIELDS = (
    TensorField("prompt_rows", torch.long),
    TensorField("decisions", torch.long),
    TensorField("decision_masks", torch.bool),
    TensorField("codes", torch.long),
    TensorField("code_masks", torch.bool),
    TensorField("rollout_decision_logprobs", torch.float32),
    TensorField("rollout_code_logprobs", torch.float32),
)
PAIRED_SCORE_FIELDS = (
    TensorField("teacher_decision_logprobs", torch.float32, required=False),
    TensorField("teacher_code_logprobs", torch.float32, required=False, code_scores=True),
    TensorField("student_prefill_decision_logprobs", torch.float32, required=False),
    TensorField("student_prefill_code_logprobs", torch.float32, required=False, code_scores=True),
)
BATCH_KEYS = tuple(field.name for field in ACTION_FIELDS) + ("rewards", "rollout_ids", "weight_versions")
PAIRED_SCORE_KEYS = tuple(field.name for field in PAIRED_SCORE_FIELDS)


def tensorize_shard(data) -> None:
    for field in (*ACTION_FIELDS, *PAIRED_SCORE_FIELDS):
        if field.name not in data:
            continue
        values = [torch.as_tensor(value, dtype=field.dtype).detach().cpu().contiguous() for value in data[field.name]]
        if field.code_scores:
            values = [value.reshape(-1, MOSS_TTS_LOCAL_SPEC.n_vq) for value in values]
        data[field.name] = values
    if "rollout_event_mask_sums" in data:
        data["rollout_event_mask_sums"] = (
            torch.as_tensor(data["rollout_event_mask_sums"], dtype=torch.float32).detach().cpu().contiguous()
        )


def prepare_shard(data, *, device):
    for field in (*ACTION_FIELDS, *PAIRED_SCORE_FIELDS):
        if field.name not in data:
            if field.required:
                raise ValueError(f"MOSS-TTS Local rollout shard is missing required field {field.name!r}.")
            continue
        data[field.name] = [
            value.to(device=device, dtype=field.dtype, non_blocking=True) for value in data[field.name]
        ]
    versions = {str(version) for version in data.get("weight_versions", [])}
    if len(versions) != 1:
        raise ValueError(f"MOSS-TTS Local rollout shard has mixed weight versions: {sorted(versions)}")
    return data
