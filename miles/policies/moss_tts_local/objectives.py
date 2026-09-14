"""Local objective selection and score requirements, independent of Megatron.

The existing numerical loss functions retain their reductions. This adapter
selects their inputs and exposes the historical metric names to the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from miles.policies.moss_tts_local import domain_distillation
from miles.policies.moss_tts_local.loss import frame_joint_grpo_loss
from miles.policies.moss_tts_local.mopd_loss import action_mopd_loss


@dataclass(frozen=True)
class LocalObjective:
    kind: str

    @classmethod
    def from_args(cls, args) -> LocalObjective:
        family = getattr(args, "moss_local_objective", "wer_grpo")
        kind = getattr(args, "moss_local_mopd_estimator", "sampled") if family == "mopd" else family
        if kind not in {"wer_grpo", "sampled", "sampled_native", "dense_reverse", "dense_forward"}:
            raise ValueError(f"Unknown Local objective {kind!r}")
        return cls(kind)

    @property
    def is_distillation(self) -> bool:
        return self.kind != "wer_grpo"

    @property
    def uses_native_teacher(self) -> bool:
        return self.kind in {"sampled_native", "dense_reverse", "dense_forward"}

    @property
    def is_dense(self) -> bool:
        return self.kind in {"dense_reverse", "dense_forward"}


@dataclass(frozen=True)
class ObjectiveTargets:
    teacher_scores: tuple[torch.Tensor, torch.Tensor] | None = None
    student_scores: tuple[torch.Tensor, torch.Tensor] | None = None
    teacher_distributions: tuple[torch.Tensor, torch.Tensor] | None = None
    teacher_domains: list[str] | None = None


@dataclass(frozen=True)
class ObjectiveLoss:
    sample_sum: torch.Tensor
    metrics: dict[str, torch.Tensor]


def compute_objective_loss(args, output, batch, old_joint_logprobs, targets: ObjectiveTargets) -> ObjectiveLoss:
    objective = LocalObjective.from_args(args)
    if objective.is_dense:
        if targets.teacher_distributions is None:
            raise ValueError("Dense MOPD requires native teacher distributions")
        value = domain_distillation.action_kl(
            args, output, batch, *targets.teacher_distributions, targets.teacher_domains
        )
        zero = value.detach().new_zeros(())
        return ObjectiveLoss(
            value,
            dict(
                loss=value.detach(),
                pg_loss=zero,
                pg_clipfrac=zero,
                ppo_kl=zero,
                ratio_mean=zero + batch.batch_size,
                mopd_dense_kl=value.detach(),
            ),
        )
    advantage_abs = None
    if objective.is_distillation:
        teacher, student = targets.teacher_scores, targets.student_scores
        if objective.uses_native_teacher:
            if targets.teacher_distributions is None:
                raise ValueError("Native MOPD requires teacher distributions")
            decisions, codes = targets.teacher_distributions
            temperature = float(args.rollout_temperature)
            teacher = (
                (decisions / temperature).log_softmax(-1).gather(-1, batch.decisions[:, None]).squeeze(-1),
                (codes / temperature).log_softmax(-1).gather(-1, batch.codes[..., None]).squeeze(-1),
            )
            student = output.decision_logprobs.detach(), output.code_logprobs.detach()
        if teacher is None:
            raise ValueError("MOPD requires teacher scores for every microbatch")
        result, advantage_abs = action_mopd_loss(
            output,
            batch,
            *teacher,
            advantage_clip=args.moss_local_mopd_advantage_clip,
            eps_clip=float(args.eps_clip),
            eps_clip_high=float(args.eps_clip_high),
            ratio_source=args.moss_local_old_policy_source,
            student_scores=student,
        )
    else:
        result = frame_joint_grpo_loss(
            output.joint_logprobs,
            old_joint_logprobs,
            batch.advantages,
            batch.decision_mask,
            batch.decision_offsets,
            eps_clip=float(args.eps_clip),
            eps_clip_high=float(args.eps_clip_high),
            sample_reduction="sum",
        )
    metrics = dict(
        loss=result.loss.detach(),
        pg_loss=result.pg_loss,
        pg_clipfrac=result.clip_fraction,
        ppo_kl=result.approx_kl,
        ratio_mean=result.ratio_mean,
    )
    if advantage_abs is not None:
        metrics["mopd_advantage_abs"] = advantage_abs
    return ObjectiveLoss(result.loss, metrics)
