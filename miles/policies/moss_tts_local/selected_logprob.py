"""Selected-action probability helpers for the hierarchical policy."""

from __future__ import annotations

import torch


def selected_logprobs(
    logits: torch.Tensor,
    actions: torch.Tensor,
    *,
    temperature: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Return fp32 full-vocabulary selected logprobs after temperature scaling."""

    if logits.ndim < 2:
        raise ValueError(f"selected_logprobs expects logits [..., vocab], got shape {tuple(logits.shape)}.")
    if tuple(actions.shape) != tuple(logits.shape[:-1]):
        raise ValueError(
            f"selected_logprobs action shape {tuple(actions.shape)} does not match logits prefix {tuple(logits.shape[:-1])}."
        )
    if actions.numel() and not bool(((actions >= 0) & (actions < logits.shape[-1])).all()):
        raise ValueError("selected_logprobs actions contain an out-of-vocabulary index.")
    temperature_tensor = torch.as_tensor(temperature, dtype=torch.float32, device=logits.device)
    if bool((temperature_tensor <= 0).any()):
        raise ValueError("selected_logprobs requires strictly positive temperature.")
    while temperature_tensor.ndim < logits.ndim:
        temperature_tensor = temperature_tensor.unsqueeze(-1)
    log_probs = torch.log_softmax(logits.float() / temperature_tensor, dim=-1)
    return log_probs.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)


def join_moss_action_logprobs(
    decision_logprobs: torch.Tensor,
    code_logprobs: torch.Tensor,
    *,
    decision_offsets: torch.Tensor,
    frame_offsets: torch.Tensor,
    code_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build one joint logprob per continue frame or terminal stop event."""

    if decision_logprobs.ndim != 1:
        raise ValueError("MOSS-TTS Local decision_logprobs must be one-dimensional.")
    if code_logprobs.ndim != 2:
        raise ValueError("MOSS-TTS Local code_logprobs must be [sum_frames, n_vq].")
    if code_mask is None:
        code_mask = torch.ones_like(code_logprobs, dtype=torch.bool)
    if code_mask.shape != code_logprobs.shape:
        raise ValueError("MOSS-TTS Local code_mask must be shape-aligned with code_logprobs.")
    if decision_offsets.ndim != 1 or frame_offsets.ndim != 1 or decision_offsets.shape != frame_offsets.shape:
        raise ValueError("MOSS-TTS Local decision/frame offsets must be aligned one-dimensional tensors.")
    if int(decision_offsets[-1]) != decision_logprobs.numel():
        raise ValueError("MOSS-TTS Local decision offsets do not cover decision_logprobs.")
    if int(frame_offsets[-1]) != code_logprobs.shape[0]:
        raise ValueError("MOSS-TTS Local frame offsets do not cover code_logprobs.")

    joint = decision_logprobs.clone()
    for sample_index in range(decision_offsets.numel() - 1):
        decision_start = int(decision_offsets[sample_index])
        decision_end = int(decision_offsets[sample_index + 1])
        frame_start = int(frame_offsets[sample_index])
        frame_end = int(frame_offsets[sample_index + 1])
        num_frames = frame_end - frame_start
        num_decisions = decision_end - decision_start
        if num_decisions not in (num_frames, num_frames + 1):
            raise ValueError(
                f"MOSS-TTS Local sample {sample_index} requires R in {{T,T+1}}, "
                f"got R={num_decisions}, T={num_frames}."
            )
        if num_frames:
            joint[decision_start : decision_start + num_frames] += (
                code_logprobs[frame_start:frame_end] * code_mask[frame_start:frame_end]
            ).sum(dim=-1)
    return joint
