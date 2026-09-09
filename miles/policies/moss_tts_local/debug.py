"""Dependency-light helpers for MOSS-TTS Local train-side debug dumps."""

from __future__ import annotations

from typing import Any

import torch

from miles.utils.types import RolloutBatch

_DEBUG_PER_SAMPLE_KEYS = (
    "sample_indices",
    "rollout_ids",
    "weight_versions",
    "policy_lags",
    "behavior_snapshot_versions",
    "rewards",
    "prompt_lengths",
    "frame_lengths",
    "decision_lengths",
    "decision_masks",
    "advantages",
    "code_advantages",
    "teacher_decision_logprobs",
    "teacher_code_logprobs",
    "teacher_versions",
    "teacher_domains",
    "student_prefill_decision_logprobs",
    "student_prefill_code_logprobs",
    "teacher_weight_digests",
    "server_joint_logprobs",
    "old_joint_logprobs",
    "trainer_decision_logprobs",
    "trainer_code_logprobs",
    "trainer_joint_logprobs",
    "source_names",
)


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    return value


def build_moss_tts_local_debug_shard(
    rollout_data: RolloutBatch,
    *,
    rank: int,
    data_parallel_rank: int,
) -> dict[str, Any]:
    """Build a compact train-side dump that joins to the rollout dump by position."""

    sample_count = len(rollout_data["rewards"])
    partition = [int(index) for index in rollout_data["partition"]]
    if len(partition) != sample_count:
        raise ValueError("MOSS-TTS Local debug partition does not match the local sample count.")
    samples = []
    for local_index, rollout_position in enumerate(partition):
        sample: dict[str, Any] = {
            "rollout_position": rollout_position,
            "data_parallel_rank": data_parallel_rank,
        }
        for key in _DEBUG_PER_SAMPLE_KEYS:
            values = rollout_data.get(key)
            if values is None:
                continue
            if isinstance(values, (list, tuple)) and len(values) == sample_count:
                sample[key] = _to_cpu(values[local_index])
            elif torch.is_tensor(values) and values.ndim >= 1 and values.shape[0] == sample_count:
                sample[key] = _to_cpu(values[local_index])
        samples.append(sample)
    return {
        "rank": rank,
        "data_parallel_rank": data_parallel_rank,
        "samples": samples,
        "layout": {
            "partition": partition,
            "micro_batch_indices": _to_cpu(rollout_data["micro_batch_indices"]),
            "num_microbatches": _to_cpu(rollout_data["num_microbatches"]),
            "num_rollouts": _to_cpu(rollout_data["num_rollouts"]),
        },
    }


__all__ = ["build_moss_tts_local_debug_shard"]
