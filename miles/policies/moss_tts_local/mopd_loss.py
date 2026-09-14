"""Per-action Local MOPD, retaining decision/code masks and sample weighting."""

from __future__ import annotations

import torch

from miles.policies.moss_tts_local.loss import FrameJointLossOutput


def _terms(current, old, teacher, behavior, mask, *, clip, eps_low, eps_high):
    if not (current.shape == old.shape == teacher.shape == behavior.shape == mask.shape):
        raise ValueError("MOPD scores/masks must be shape-aligned")
    current = torch.where(mask, current, 0)
    old = torch.where(mask, old.detach(), 0)
    teacher = torch.where(mask, teacher.detach(), 0)
    behavior = torch.where(mask, behavior.detach(), 0)
    advantage = (teacher - behavior).clamp(-clip, clip)
    ratio = (current - old).exp()
    clipped = ratio.clamp(1 - eps_low, 1 + eps_high)
    pg = -torch.minimum(ratio * advantage, clipped * advantage)
    return torch.stack((pg, (ratio != clipped).float(), old - current, ratio, advantage.abs()), dim=-1)


def action_mopd_loss(
    output,
    batch,
    teacher_decisions,
    teacher_codes,
    *,
    advantage_clip,
    eps_clip,
    eps_clip_high,
    ratio_source="trainer_preupdate",
    student_scores=None,
):
    if ratio_source == "trainer_preupdate":
        old_d, old_c = output.decision_logprobs.detach(), output.code_logprobs.detach()
    elif ratio_source == "server_behavior":
        old_d, old_c = batch.rollout_decision_logprobs, batch.rollout_code_logprobs
    else:
        raise ValueError(f"Unsupported MOPD ratio source {ratio_source}")
    options = dict(clip=advantage_clip, eps_low=eps_clip, eps_high=eps_clip_high)
    scoring_d, scoring_c = student_scores or (batch.rollout_decision_logprobs, batch.rollout_code_logprobs)
    decisions = _terms(
        output.decision_logprobs,
        old_d,
        teacher_decisions,
        scoring_d,
        batch.decision_mask,
        **options,
    )
    codes = _terms(output.code_logprobs, old_c, teacher_codes, scoring_c, batch.code_mask, **options)
    # Equal weight per sample, normalized by its actual number of active actions.
    totals = decisions.new_zeros(5)
    for i in range(batch.batch_size):
        ds, de = int(batch.decision_offsets[i]), int(batch.decision_offsets[i + 1])
        fs, fe = int(batch.frame_offsets[i]), int(batch.frame_offsets[i + 1])
        dm, cm = batch.decision_mask[ds:de], batch.code_mask[fs:fe]
        count = (dm.sum() + cm.sum()).clamp_min(1)
        totals = (
            totals + ((decisions[ds:de] * dm[:, None]).sum(0) + (codes[fs:fe] * cm[..., None]).sum((0, 1))) / count
        )
    return (
        FrameJointLossOutput(
            loss=totals[0],
            pg_loss=totals[0].detach(),
            clip_fraction=totals[1].detach(),
            approx_kl=totals[2].detach(),
            ratio_mean=totals[3].detach(),
        ),
        totals[4].detach(),
    )
