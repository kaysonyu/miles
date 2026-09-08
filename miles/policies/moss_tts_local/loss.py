"""Frame-joint GRPO objective for MOSS-TTS Local."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class FrameJointLossOutput:
    loss: torch.Tensor
    pg_loss: torch.Tensor
    clip_fraction: torch.Tensor
    approx_kl: torch.Tensor
    ratio_mean: torch.Tensor


def expand_sample_advantages(
    sample_advantages: torch.Tensor,
    decision_offsets: torch.Tensor,
) -> torch.Tensor:
    if sample_advantages.ndim != 1 or sample_advantages.numel() != decision_offsets.numel() - 1:
        raise ValueError("Sample advantages and decision offsets are inconsistent.")
    return torch.cat(
        [
            sample_advantages[sample_index].expand(
                int(decision_offsets[sample_index + 1] - decision_offsets[sample_index])
            )
            for sample_index in range(sample_advantages.numel())
        ],
        dim=0,
    )


def _reduce_sample_means(
    values: torch.Tensor,
    mask: torch.Tensor,
    offsets: torch.Tensor,
    *,
    reduction: Literal["mean", "sum"],
) -> torch.Tensor:
    if reduction not in ("mean", "sum"):
        raise ValueError(f"Unknown sample reduction {reduction!r}.")
    sample_count = offsets.numel() - 1
    sample_ids = torch.repeat_interleave(
        torch.arange(sample_count, device=values.device),
        offsets[1:] - offsets[:-1],
        output_size=values.numel(),
    )
    weights = mask.to(dtype=values.dtype)
    sample_sums = torch.zeros(sample_count, dtype=values.dtype, device=values.device)
    sample_counts = torch.zeros_like(sample_sums)
    sample_sums.scatter_add_(0, sample_ids, values * weights)
    sample_counts.scatter_add_(0, sample_ids, weights)
    sample_means = sample_sums / sample_counts.clamp_min(1)
    return sample_means.mean() if reduction == "mean" else sample_means.sum()


def frame_joint_grpo_loss(
    current_joint_logprobs: torch.Tensor,
    old_joint_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    event_mask: torch.Tensor,
    decision_offsets: torch.Tensor,
    *,
    eps_clip: float,
    eps_clip_high: float | None = None,
    sample_reduction: Literal["mean", "sum"] = "mean",
) -> FrameJointLossOutput:
    """Compute clipped policy loss with equal weight per rollout sample.

    The public default is a batch mean.  Megatron training requests ``sum``:
    its pipeline schedule already divides accumulated micro-batches, while
    Miles's outer reducer divides by the step-global sample count.  Supplying
    sample sums therefore keeps every sample's gradient weight invariant when
    dynamic packing changes the number of samples in each micro-batch.
    """

    tensors = (current_joint_logprobs, old_joint_logprobs, advantages, event_mask)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("MOSS-TTS Local joint logprobs, advantages, and event mask must be one-dimensional.")
    if len({tensor.numel() for tensor in tensors}) != 1:
        raise ValueError("MOSS-TTS Local joint logprobs, advantages, and event mask must be shape-aligned.")
    if event_mask.dtype != torch.bool:
        raise ValueError("MOSS-TTS Local event_mask must be boolean.")
    if int(decision_offsets[-1]) != current_joint_logprobs.numel():
        raise ValueError("MOSS-TTS Local decision offsets do not cover all events.")
    eps_high = float(eps_clip if eps_clip_high is None else eps_clip_high)

    log_ratio = current_joint_logprobs - old_joint_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - float(eps_clip), 1.0 + eps_high)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    event_pg_loss = -torch.minimum(unclipped, clipped)
    event_clip = (ratio != clipped_ratio).float()
    event_kl = old_joint_logprobs - current_joint_logprobs

    def reduce(values: torch.Tensor) -> torch.Tensor:
        return _reduce_sample_means(
            values,
            event_mask,
            decision_offsets,
            reduction=sample_reduction,
        )

    pg_loss = reduce(event_pg_loss)
    clip_fraction = reduce(event_clip)
    approx_kl = reduce(event_kl)
    ratio_mean = reduce(ratio)
    return FrameJointLossOutput(
        loss=pg_loss,
        pg_loss=pg_loss.detach(),
        clip_fraction=clip_fraction.detach(),
        approx_kl=approx_kl.detach(),
        ratio_mean=ratio_mean.detach(),
    )
