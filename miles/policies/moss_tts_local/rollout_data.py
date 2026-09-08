"""MOSS-TTS Local sample conversion and DP partitioning."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

import torch

from miles.policies.moss_tts_local.types import MossTTSLocalTrajectoryV2
from miles.utils.dp_schedule import build_dp_schedule
from miles.utils.types import RolloutBatch, Sample

_TENSOR_DTYPES: dict[str, torch.dtype] = {
    "prompt_rows": torch.long,
    "decisions": torch.long,
    "decision_masks": torch.bool,
    "codes": torch.long,
    "code_masks": torch.bool,
    "rollout_decision_logprobs": torch.float32,
    "rollout_code_logprobs": torch.float32,
}


def _cpu_tensor(value, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype).detach().cpu().contiguous()


def _tensorize_moss_rollout_data(rollout_data: RolloutBatch) -> None:
    for key, dtype in _TENSOR_DTYPES.items():
        if key in rollout_data:
            rollout_data[key] = [_cpu_tensor(value, dtype) for value in rollout_data[key]]
    for key in ("rollout_event_mask_sums",):
        if key in rollout_data:
            rollout_data[key] = _cpu_tensor(rollout_data[key], torch.float32)


class MossTTSLocalRolloutDataAdapter:
    """Own the structured trajectory-to-training representation."""

    def dispose(self) -> None:
        from miles.policies.moss_tts_local.rollout import dispose_rollout_state

        dispose_rollout_state()

    def samples_to_train_data(
        self,
        args: Namespace,
        samples: list[Sample],
        *,
        reward_postprocess: Callable[[list[Sample]], tuple[list[float], list[float]]],
    ) -> RolloutBatch:
        if not samples:
            raise ValueError("MOSS-TTS Local rollout conversion requires at least one sample.")
        if getattr(args, "custom_convert_samples_to_train_data_path", None) is not None:
            raise ValueError(
                "MOSS-TTS Local owns structured rollout conversion; "
                "--custom-convert-samples-to-train-data-path is not supported."
            )

        trajectories: list[MossTTSLocalTrajectoryV2] = []
        for sample_index, sample in enumerate(samples):
            trajectory = sample.structured_trajectory
            if not isinstance(trajectory, MossTTSLocalTrajectoryV2):
                raise TypeError(f"MOSS-TTS Local sample {sample_index} is missing a MossTTSLocalTrajectoryV2.")
            trajectory.validate()
            if sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
                raise ValueError(
                    f"MOSS-TTS Local sample {sample_index} has non-trainable status {sample.status.value!r}."
                )
            expected_status = (
                Sample.Status.COMPLETED if trajectory.finish_reason == "stop" else Sample.Status.TRUNCATED
            )
            if sample.status != expected_status:
                raise ValueError(
                    f"MOSS-TTS Local sample {sample_index} finish/status mismatch: "
                    f"finish_reason={trajectory.finish_reason!r}, status={sample.status.value!r}."
                )
            trajectories.append(trajectory)

        raw_rewards, rewards = reward_postprocess(samples)
        if len(raw_rewards) != len(samples) or len(rewards) != len(samples):
            raise ValueError("Reward post-processing must return one raw and normalized reward per sample.")
        rollout_ids = [sample.rollout_id if sample.rollout_id is not None else sample.index for sample in samples]

        decision_masks = []
        code_masks = []
        for sample, trajectory in zip(samples, trajectories, strict=True):
            decision_mask = trajectory.decision_mask.clone()
            code_mask = trajectory.code_mask.clone()
            if sample.remove_sample:
                decision_mask.zero_()
                code_mask.zero_()
            decision_masks.append(decision_mask)
            code_masks.append(code_mask)

        event_mask_sums_per_sample = [int(mask.sum().item()) for mask in decision_masks]
        rollout_total_events: dict[int, int] = {}
        for rollout_id, event_count in zip(rollout_ids, event_mask_sums_per_sample, strict=True):
            rollout_total_events[rollout_id] = rollout_total_events.get(rollout_id, 0) + event_count

        train_data: RolloutBatch = {
            "prompt_rows": [trajectory.prompt_rows for trajectory in trajectories],
            "decisions": [trajectory.decisions for trajectory in trajectories],
            "decision_masks": decision_masks,
            "codes": [trajectory.codes for trajectory in trajectories],
            "code_masks": code_masks,
            "rollout_decision_logprobs": [trajectory.decision_logprobs for trajectory in trajectories],
            "rollout_code_logprobs": [trajectory.code_logprobs for trajectory in trajectories],
            "prompt_lengths": [int(trajectory.prompt_rows.shape[0]) for trajectory in trajectories],
            "frame_lengths": [trajectory.num_frames for trajectory in trajectories],
            "decision_lengths": [trajectory.num_decisions for trajectory in trajectories],
            "total_lengths": [
                int(trajectory.prompt_rows.shape[0]) + trajectory.num_frames for trajectory in trajectories
            ],
            "action_counts": [trajectory.num_actions for trajectory in trajectories],
            "event_counts": event_mask_sums_per_sample,
            "rollout_event_mask_sums": [rollout_total_events[rollout_id] for rollout_id in rollout_ids],
            "weight_versions": [trajectory.weight_version for trajectory in trajectories],
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if trajectory.finish_reason == "length" else 0 for trajectory in trajectories],
            "sample_indices": [sample.index for sample in samples],
            "rollout_ids": rollout_ids,
            "source_names": [(sample.metadata or {}).get("source_name", "unknown") for sample in samples],
        }
        if any(sample.metadata and "raw_reward" in sample.metadata for sample in samples):
            train_data["raw_reward"] = [
                sample.metadata["raw_reward"] if sample.metadata and "raw_reward" in sample.metadata else sample.reward
                for sample in samples
            ]
        return train_data


def split_raw(args, data, train_parallel_config):
    """Schedule structured samples, preserving their real global sequence lengths."""
    lengths = [int(value) for value in data["total_lengths"]]
    partitions, micro_batches, microbatch_counts, rollout_counts = build_dp_schedule(
        args,
        train_parallel_config,
        lengths,
        global_batch_size=args.global_batch_size,
        rollout_indices=data["rollout_ids"],
    )
    result = []
    for rank, partition in enumerate(partitions):
        shard = {
            key: [values[index] for index in partition]
            for key, values in data.items()
            if key not in {"total_lengths", "raw_reward"}
        }
        shard.update(
            partition=list(partition),
            total_lengths=lengths,
            raw_reward=data["raw_reward"],
            micro_batch_indices=micro_batches[rank],
            num_microbatches=microbatch_counts,
            num_rollouts=rollout_counts,
        )
        _tensorize_moss_rollout_data(shard)
        result.append(shard)
    return result
