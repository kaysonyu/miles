"""Selected-score collection and Megatron loss adaptation for Local policies."""

from __future__ import annotations

import torch
from megatron.core import mpu

from miles.policies.moss_tts_local.batch import MossTTSLocalPolicyBatch
from miles.policies.moss_tts_local.dense_mopd import native_action_entropies
from miles.policies.moss_tts_local.objectives import LocalObjective, ObjectiveTargets, compute_objective_loss
from miles.policies.moss_tts_local.policy import MossTTSLocalPolicyOutput
from miles.policies.moss_tts_local.selected_logprob import join_moss_action_logprobs


def _collect_policy_output(
    output: MossTTSLocalPolicyOutput,
    *,
    batch: MossTTSLocalPolicyBatch,
    non_loss_data: bool = True,
) -> dict[str, list[torch.Tensor]]:
    if not non_loss_data:
        raise ValueError("MOSS-TTS Local forward-only collection requires non_loss_data=True.")
    if batch.batch_size == 1:
        result: dict[str, list[torch.Tensor]] = {
            "decision_logprobs": [output.decision_logprobs.detach()],
            "code_logprobs": [output.code_logprobs.detach()],
            "joint_logprobs": [output.joint_logprobs.detach()],
        }
        if output.decision_entropy is not None:
            result["decision_entropy"] = [output.decision_entropy.detach()]
        if output.code_entropy is not None:
            result["code_entropy"] = [output.code_entropy.detach()]
        return result
    decision_values = []
    code_values = []
    joint_values = []
    for sample_index in range(batch.batch_size):
        decision_start = int(batch.decision_offsets[sample_index])
        decision_end = int(batch.decision_offsets[sample_index + 1])
        frame_start = int(batch.frame_offsets[sample_index])
        frame_end = int(batch.frame_offsets[sample_index + 1])
        decision_values.append(output.decision_logprobs[decision_start:decision_end].detach())
        code_values.append(output.code_logprobs[frame_start:frame_end].detach())
        joint_values.append(output.joint_logprobs[decision_start:decision_end].detach())
    result = {
        "decision_logprobs": decision_values,
        "code_logprobs": code_values,
        "joint_logprobs": joint_values,
    }
    if output.decision_entropy is not None:
        result["decision_entropy"] = [
            output.decision_entropy[
                int(batch.decision_offsets[index]) : int(batch.decision_offsets[index + 1])
            ].detach()
            for index in range(batch.batch_size)
        ]
    if output.code_entropy is not None:
        result["code_entropy"] = [
            output.code_entropy[int(batch.frame_offsets[index]) : int(batch.frame_offsets[index + 1])].detach()
            for index in range(batch.batch_size)
        ]
    return result


def _moss_loss_closure(
    args,
    batch: MossTTSLocalPolicyBatch,
    num_microbatches: int,
    step_global_batch_size: int,
    output: MossTTSLocalPolicyOutput,
    *,
    trainer_output_collector: dict[str, list[torch.Tensor]] | None = None,
    parity_values: list[torch.Tensor] | None = None,
    teacher_scores: tuple[torch.Tensor, torch.Tensor] | None = None,
    student_scores: tuple[torch.Tensor, torch.Tensor] | None = None,
    teacher_distributions: tuple[torch.Tensor, torch.Tensor] | None = None,
    teacher_domains: list[str] | None = None,
):
    objective = LocalObjective.from_args(args)
    if not objective.is_dense and batch.advantages is None:
        raise ValueError("MOSS-TTS Local train batch is missing advantages.")
    old_joint_logprobs = batch.old_joint_logprobs
    if old_joint_logprobs is None:
        if not (
            args.moss_local_old_policy_source == "trainer_preupdate"
            and getattr(args, "moss_local_reuse_train_forward", True)
        ):
            raise ValueError("MOSS-TTS Local train batch is missing old_joint_logprobs.")
        old_joint_logprobs = output.joint_logprobs.detach()
        if trainer_output_collector is None or parity_values is None:
            raise RuntimeError("MOSS-TTS Local same-forward replay requires output and parity collectors.")
    if trainer_output_collector is not None:
        if parity_values is None:
            raise RuntimeError("Local train output collection requires parity storage")
        captured = _collect_policy_output(output, batch=batch)
        for key, values in captured.items():
            trainer_output_collector.setdefault(f"trainer_{key}", []).extend(values)
        comparison_logprobs = batch.server_joint_logprobs()
        if teacher_distributions is not None and student_scores is not None:
            # Native KL can include teacher-origin replay: compare against the
            # current student's supplied-action scores, not teacher behavior.
            comparison_logprobs = join_moss_action_logprobs(
                *student_scores,
                decision_offsets=batch.decision_offsets,
                frame_offsets=batch.frame_offsets,
                code_mask=batch.code_mask,
            )
        active_parity = (output.joint_logprobs.detach() - comparison_logprobs)[batch.decision_mask]
        parity_values.append(
            active_parity.abs().max() if active_parity.numel() else output.joint_logprobs.detach().sum() * 0.0
        )
    loss_output = compute_objective_loss(
        args,
        output,
        batch,
        old_joint_logprobs,
        ObjectiveTargets(teacher_scores, student_scores, teacher_distributions, teacher_domains),
    )
    # ``loss_output`` contains per-sample means summed across this physical
    # micro-batch.  Megatron divides each micro-batch by ``num_microbatches``;
    # the inverse factor below plus the step-global divisor therefore yields
    # one equal-weight mean over samples, independent of dynamic packing.
    sample_sum_loss = loss_output.sample_sum
    scaled_loss = (
        sample_sum_loss
        * num_microbatches
        / step_global_batch_size
        * mpu.get_data_parallel_world_size(with_context_parallel=True)
    )
    metric_keys = list(loss_output.metrics)
    metric_values = torch.stack(
        [
            torch.zeros((), device=sample_sum_loss.device),
            *loss_output.metrics.values(),
        ]
    )
    if teacher_distributions is not None:
        entropies = native_action_entropies(
            output, batch, *teacher_distributions, temperature=float(args.rollout_temperature)
        )
        metric_keys.extend(entropies)
        metric_values = torch.cat((metric_values, torch.stack(list(entropies.values()))))
    return (
        scaled_loss,
        torch.tensor(1, device=sample_sum_loss.device),
        {
            "keys": metric_keys,
            "values": metric_values,
        },
    )
