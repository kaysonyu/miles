"""Packing-invariant gradients across the Local objective integration seam."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from miles.policies.moss_tts_local.selected_logprob import join_moss_action_logprobs
from miles.policies.moss_tts_local.training_loss import _moss_loss_closure


def loss_case(kind, *, source="trainer_preupdate"):
    generator = torch.Generator().manual_seed(7)
    decisions = torch.randn(4, 2, generator=generator, requires_grad=True)
    codes = torch.randn(2, 12, 5, generator=generator, requires_grad=True)
    teacher_decisions = torch.randn(4, 2, generator=generator)
    teacher_codes = torch.randn(2, 12, 5, generator=generator)
    args = SimpleNamespace(
        moss_local_objective="wer_grpo" if kind == "wer_grpo" else "mopd",
        moss_local_mopd_estimator=kind,
        moss_local_old_policy_source=source,
        moss_local_mopd_advantage_clip=5.0,
        rollout_temperature=0.7,
        eps_clip=0.2,
        eps_clip_high=0.2,
    )

    def sample(indices):
        # Sample 0 stops immediately; sample 1 emits two frames and a stop.
        ds, de = (0, 4) if len(indices) == 2 else ((0, 1) if indices == [0] else (1, 4))
        fs, fe = (0, 0) if indices == [0] else (0, 2)
        d, c = decisions[ds:de], codes[fs:fe]
        da = torch.tensor([1, 0, 0, 1])[ds:de]
        ca = torch.arange(24).reshape(2, 12)[fs:fe] % 5
        dlog = (d / 0.7).log_softmax(-1).gather(-1, da[:, None]).squeeze(-1)
        clog = (c / 0.7).log_softmax(-1).gather(-1, ca[..., None]).squeeze(-1)
        offsets = dict(
            decision_offsets=torch.tensor([0, 1, 4] if len(indices) == 2 else [0, de - ds]),
            frame_offsets=torch.tensor([0, 0, 2] if len(indices) == 2 else [0, fe - fs]),
        )
        mask = torch.ones_like(ca, dtype=torch.bool)
        joint = join_moss_action_logprobs(dlog, clog, code_mask=mask, **offsets)
        batch = SimpleNamespace(
            batch_size=len(indices),
            decisions=da,
            codes=ca,
            decision_mask=torch.ones_like(da, dtype=torch.bool),
            code_mask=mask,
            advantages=torch.tensor([0.5, -0.2, -0.2, -0.2])[ds:de],
            old_joint_logprobs=joint.detach() - 0.1,
            rollout_decision_logprobs=dlog.detach() - 0.1,
            rollout_code_logprobs=clog.detach() - 0.1,
            **offsets,
        )
        output = SimpleNamespace(
            decision_logits=d,
            code_logits=c,
            decision_logprobs=dlog,
            code_logprobs=clog,
            joint_logprobs=joint,
        )
        teacher_d, teacher_c = teacher_decisions[ds:de], teacher_codes[fs:fe]
        kwargs = {}
        if kind != "wer_grpo":
            kwargs["teacher_scores"] = (
                (teacher_d / 0.7).log_softmax(-1).gather(-1, da[:, None]).squeeze(-1),
                (teacher_c / 0.7).log_softmax(-1).gather(-1, ca[..., None]).squeeze(-1),
            )
            kwargs["student_scores"] = dlog.detach(), clog.detach()
        if kind in {"sampled_native", "dense_reverse", "dense_forward"}:
            kwargs["teacher_distributions"] = teacher_d, teacher_c
        return batch, output, kwargs

    return args, (decisions, codes), sample


@pytest.mark.parametrize("kind", ["wer_grpo", "sampled", "sampled_native", "dense_reverse", "dense_forward"])
def test_megatron_loss_scaling_keeps_gradients_invariant_to_packing(monkeypatch, kind):
    from miles.policies.moss_tts_local import training_loss

    monkeypatch.setattr(training_loss.mpu, "get_data_parallel_world_size", lambda **kwargs: 1)
    args, parameters, sample = loss_case(kind)
    batch, output, targets = sample([0, 1])
    packed, _, _ = _moss_loss_closure(args, batch, 1, 2, output, **targets)
    packed_grad = torch.autograd.grad(packed, parameters)
    unpacked = parameters[0].new_zeros(())
    for indices in ([0], [1]):
        batch, output, targets = sample(indices)
        value, _, _ = _moss_loss_closure(args, batch, 2, 2, output, **targets)
        unpacked = unpacked + value / 2  # Megatron's microbatch accumulation divisor.
    unpacked_grad = torch.autograd.grad(unpacked, parameters)
    torch.testing.assert_close(packed, unpacked)
    for packed_value, unpacked_value in zip(packed_grad, unpacked_grad, strict=True):
        torch.testing.assert_close(packed_value, unpacked_value)


@pytest.mark.parametrize("kind", ["wer_grpo", "sampled", "sampled_native", "dense_reverse", "dense_forward"])
def test_workflow_runs_through_the_explicit_runtime_and_forward_builder(monkeypatch, kind):
    from miles.backends.megatron_utils import model as backend
    from miles.backends.training_utils.data import DataIterator
    from miles.policies.base import TrainingContext
    from miles.policies.moss_tts_local import workflow

    args, parameters, sample = loss_case(kind)
    batch, output, targets = sample([0, 1])
    args.moss_local_packed_thd = True
    args.moss_local_reuse_train_forward = True
    args.debug_train_only = False
    args.num_rollout = 1
    args.save_debug_train_data = None
    output.decision_entropy = output.code_entropy = None
    events = []
    data = dict(
        prompt_rows=[torch.zeros(1, 13, dtype=torch.long)] * 2,
        decisions=[batch.decisions[:1], batch.decisions[1:]],
        decision_masks=[batch.decision_mask[:1], batch.decision_mask[1:]],
        codes=[batch.codes[:0], batch.codes],
        code_masks=[batch.code_mask[:0], batch.code_mask],
        rollout_decision_logprobs=[batch.rollout_decision_logprobs[:1], batch.rollout_decision_logprobs[1:]],
        rollout_code_logprobs=[batch.rollout_code_logprobs[:0], batch.rollout_code_logprobs],
        rewards=[0.5, -0.2],
        rollout_ids=[0, 0],
        weight_versions=["7", "7"],
        micro_batch_indices=[[0, 1]],
        num_microbatches=[1],
        num_rollouts=[2],
    )
    if kind != "wer_grpo":
        for prefix, pair in (("teacher", targets["teacher_scores"]), ("student_prefill", targets["student_scores"])):
            data[prefix + "_decision_logprobs"] = [pair[0][:1], pair[0][1:]]
            data[prefix + "_code_logprobs"] = [pair[1][:0], pair[1]]
        data["teacher_domains"] = ["one", "one"]
    runtime = SimpleNamespace(
        published_version="7",
        rollout_data_postprocess=None,
        backup_required=True,
        finish_rollout=lambda rollout_id: events.append(("finish", rollout_id)),
        pop_weight_metrics=lambda: {},
    )
    context = TrainingContext(args, [lambda **kwargs: output], None, None, runtime, rollout_id=0)
    session = workflow.MossTTSLocalTrainingWorkflow()
    pool = SimpleNamespace(
        score=lambda *args: targets["teacher_distributions"], verify_frozen=lambda: events.append("audit")
    )
    native = kind in {"sampled_native", "dense_reverse", "dense_forward"}
    monkeypatch.setattr(session, "_get_native_teachers", lambda args: pool if native else None)
    monkeypatch.setattr(
        workflow, "get_data_iterator", lambda *args: ([DataIterator(data, micro_batch_indices=[[0, 1]])], [1])
    )
    monkeypatch.setattr(workflow, "_log_packing", lambda *args: None)
    monkeypatch.setattr(workflow, "_log_training_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow.train_metric_utils, "log_perf_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow, "inverse_timer", lambda *args: nullcontext())
    monkeypatch.setattr(workflow, "timer", lambda *args: nullcontext())
    monkeypatch.setattr(workflow.mpu, "get_data_parallel_world_size", lambda **kwargs: 1)

    def train(rollout_id, models, optimizer, scheduler, iterators, microbatches, batch_sizes, **kwargs):
        assert microbatches == [1] and batch_sizes == [2]
        forward = kwargs["custom_forward_step_builder"](args, 1, 2)
        model_output, loss = forward(iterators[0], models[0])
        value, _, _ = loss(model_output)
        value.backward()
        events.append("train")
        return workflow.TrainStepOutcome.NORMAL

    monkeypatch.setattr(backend, "train", train)
    session.train_actor(context, 0, data)
    assert events == ["train", ("finish", 0)] + (["audit"] if native else [])
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert len(data["old_joint_logprobs"]) == len(data["trainer_joint_logprobs"]) == 2
    assert data["policy_lags"] == [0, 0]
