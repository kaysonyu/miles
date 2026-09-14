"""Megatron training workflow for MOSS-TTS Local structured actions."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import mpu

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.backends.training_utils import log_utils as train_metric_utils
from miles.backends.training_utils.data import get_data_iterator
from miles.policies.base import TrainingContext
from miles.policies.moss_tts_local.async_policy import policy_lag
from miles.policies.moss_tts_local.batch import (
    MOSS_TTS_LOCAL_BATCH_KEYS,
    collate_moss_tts_local_batch,
)
from miles.policies.moss_tts_local.data_schema import PAIRED_SCORE_KEYS, prepare_shard
from miles.policies.moss_tts_local.debug import build_moss_tts_local_debug_shard
from miles.policies.moss_tts_local.dense_mopd import NativeTeacherPool
from miles.policies.moss_tts_local.objectives import LocalObjective
from miles.policies.moss_tts_local.selected_logprob import join_moss_action_logprobs
from miles.policies.moss_tts_local.training_loss import _collect_policy_output as _collect_policy_output
from miles.policies.moss_tts_local.training_loss import _moss_loss_closure as _moss_loss_closure
from miles.utils.timer import inverse_timer, timer
from miles.utils.tracking_utils import tracking as logging_utils
from miles.utils.types import RolloutBatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ReplayPreparation:
    source: str
    reuse_train_forward: bool
    parity_max_abs: torch.Tensor | None


def _save_debug_train_data(args, *, rollout_id: int, rollout_data: RolloutBatch) -> None:
    """Persist MOSS structured replay outputs without assuming text-token fields."""

    path_template = getattr(args, "save_debug_train_data", None)
    if path_template is None:
        return
    if not mpu.is_pipeline_last_stage(ignore_virtual=True) or mpu.get_tensor_model_parallel_rank() != 0:
        return

    rank = dist.get_rank()
    dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
    local_shard = build_moss_tts_local_debug_shard(rollout_data, rank=rank, data_parallel_rank=dp_rank)
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    writer_rank = mpu.get_data_parallel_src_rank(with_context_parallel=False)
    if dp_size == 1:
        shards = [local_shard]
    else:
        shards = [None] * dp_size if rank == writer_rank else None
        dist.gather_object(
            local_shard,
            shards,
            dst=writer_rank,
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=False),
        )
    if rank != writer_rank:
        return
    assert shards is not None
    samples = [sample for shard in shards for sample in shard["samples"]]
    samples.sort(key=lambda sample: sample["rollout_position"])
    path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "policy_family": "moss_tts_local",
            "rollout_id": rollout_id,
            "rank": rank,
            "samples": samples,
            "dp_shards": [
                shard["layout"] | {"rank": shard["rank"], "data_parallel_rank": shard["data_parallel_rank"]}
                for shard in shards
            ],
        },
        path,
    )
    logger.info("Saved MOSS-TTS Local debug train data from %d DP shard(s) to %s", dp_size, path)


def _make_train_forward_step(
    args,
    num_microbatches: int,
    step_global_batch_size: int,
    *,
    trainer_output_collector: dict[str, list[torch.Tensor]] | None = None,
    parity_values: list[torch.Tensor] | None = None,
    native_teachers: NativeTeacherPool | None = None,
):
    def forward_step(data_iterator, model, return_schedule_plan: bool = False):
        if return_schedule_plan:
            raise ValueError("MOSS-TTS Local P0 does not support combined 1f1b schedule plans.")
        batch_keys = [*MOSS_TTS_LOCAL_BATCH_KEYS, "advantages"]
        if not (
            args.moss_local_old_policy_source == "trainer_preupdate"
            and getattr(args, "moss_local_reuse_train_forward", True)
            and trainer_output_collector is not None
        ):
            batch_keys.append("old_joint_logprobs")
        is_mopd = LocalObjective.from_args(args).is_distillation
        if is_mopd:
            batch_keys.extend(PAIRED_SCORE_KEYS)
        if native_teachers is not None:
            batch_keys.append("teacher_domains")
        raw_batch = data_iterator.get_next(batch_keys)
        teacher_scores = student_scores = None
        if is_mopd:
            teacher_scores = (
                torch.cat(raw_batch["teacher_decision_logprobs"]),
                torch.cat([value.reshape(-1, 12) for value in raw_batch["teacher_code_logprobs"]]),
            )
            student_scores = (
                torch.cat(raw_batch["student_prefill_decision_logprobs"]),
                torch.cat([value.reshape(-1, 12) for value in raw_batch["student_prefill_code_logprobs"]]),
            )
        batch = collate_moss_tts_local_batch(
            raw_batch,
            packed_thd=bool(getattr(args, "moss_local_packed_thd", False)),
        )
        distributions = (
            native_teachers.score(batch, raw_batch["teacher_domains"]) if native_teachers is not None else None
        )
        output = (
            model(policy_batch=batch, with_entropy=False, with_logits=True)
            if native_teachers is not None
            else model(policy_batch=batch, with_entropy=False)
        )
        return output, partial(
            _moss_loss_closure,
            args,
            batch,
            num_microbatches,
            step_global_batch_size,
            trainer_output_collector=trainer_output_collector,
            parity_values=parity_values,
            teacher_scores=teacher_scores,
            student_scores=student_scores,
            teacher_distributions=distributions,
            teacher_domains=raw_batch.get("teacher_domains"),
        )

    return forward_step


def _log_packing(context, rollout_data):
    local_lengths = [
        int(prompt.shape[0]) + int(codes.shape[0])
        for prompt, codes in zip(rollout_data["prompt_rows"], rollout_data["codes"], strict=True)
    ]
    micro_batch_sizes = [len(indices) for indices in rollout_data["micro_batch_indices"]]
    micro_batch_rows = [
        sum(local_lengths[index] for index in indices) for indices in rollout_data["micro_batch_indices"]
    ]
    if mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0:
        logger.info(
            "MOSS-TTS Local packing: packed_thd=%s dynamic=%s microbatches_per_rank=%d "
            "samples_per_microbatch(mean/max)=%.3f/%d rows_per_microbatch(mean/max)=%.1f/%d",
            bool(getattr(context.args, "moss_local_packed_thd", False)),
            bool(getattr(context.args, "use_dynamic_batch_size", False)),
            len(micro_batch_sizes),
            sum(micro_batch_sizes) / len(micro_batch_sizes),
            max(micro_batch_sizes),
            sum(micro_batch_rows) / len(micro_batch_rows),
            max(micro_batch_rows),
        )


def _log_training_metrics(
    context, rollout_id, rollout_data, *, parity_max_abs, source, cpu_backup_required, num_microbatches
):
    if mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0:
        assert parity_max_abs is not None
        step = rollout_id
        logging_utils.log(
            context.args,
            {
                "train/moss_logprob_parity_max_abs": float(parity_max_abs),
                "train/moss_old_policy_is_trainer_surrogate": float(source == "trainer_preupdate"),
                "train/moss_old_policy_is_behavior_snapshot": float(source == "trainer_behavior"),
                "train/moss_policy_lag": float(rollout_data.get("policy_lags", [0])[0]),
                "train/moss_behavior_replay_seconds": float(rollout_data.get("behavior_replay_seconds", 0.0)),
                "train/moss_actor_cpu_backup_skipped": float(not cpu_backup_required),
                "train/moss_packed_thd": float(bool(getattr(context.args, "moss_local_packed_thd", False))),
                "train/moss_microbatches_per_rank": float(sum(num_microbatches)),
                "train/moss_samples_per_microbatch_mean_local": float(
                    len(rollout_data["rewards"]) / max(sum(num_microbatches), 1)
                ),
                "train/moss_samples_per_microbatch_max_local": float(
                    max(len(indices) for indices in rollout_data["micro_batch_indices"])
                ),
                "train/step": step,
            },
            step_key="train/step",
        )


class MossTTSLocalTrainingWorkflow:
    needs_tokenizer = False

    def __init__(self):
        self._native_teachers = None

    @staticmethod
    def uses_behavior_snapshots(args) -> bool:
        return args.moss_local_old_policy_source == "trainer_behavior"

    def prepare_rollout_data(self, data: RolloutBatch, *, device: torch.device | int) -> RolloutBatch:
        return prepare_shard(data, device=device)

    def _get_native_teachers(self, args):
        if not LocalObjective.from_args(args).uses_native_teacher:
            return None
        if self._native_teachers is None:
            self._native_teachers = NativeTeacherPool(args)
        return self._native_teachers

    @staticmethod
    def _forward_step(data_iterator, model, return_schedule_plan: bool = False, *, packed_thd: bool = False):
        if return_schedule_plan:
            raise ValueError("MOSS-TTS Local forward-only does not use schedule plans.")
        raw_batch = data_iterator.get_next(MOSS_TTS_LOCAL_BATCH_KEYS)
        batch = collate_moss_tts_local_batch(raw_batch, packed_thd=packed_thd)
        output = model(policy_batch=batch, with_entropy=False)
        return output, partial(_collect_policy_output, batch=batch)

    def compute_log_probs(
        self,
        context: TrainingContext,
        data_iterator,
        num_microbatches: list[int],
        *,
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:
        # Megatron execution is optional when importing policy math on CPU.
        from miles.backends.megatron_utils.model import forward_only

        return forward_only(
            None,
            context.args,
            context.model,
            data_iterator,
            num_microbatches,
            store_prefix=store_prefix,
            rollout_id=context.rollout_id,
            custom_forward_step=partial(
                self._forward_step,
                packed_thd=bool(getattr(context.args, "moss_local_packed_thd", False)),
            ),
        )

    @staticmethod
    def _server_joint_logprobs(rollout_data: RolloutBatch) -> list[torch.Tensor]:
        result = []
        for decision_logprobs, code_logprobs, code_mask in zip(
            rollout_data["rollout_decision_logprobs"],
            rollout_data["rollout_code_logprobs"],
            rollout_data["code_masks"],
            strict=True,
        ):
            result.append(
                join_moss_action_logprobs(
                    decision_logprobs,
                    code_logprobs,
                    decision_offsets=torch.tensor([0, decision_logprobs.numel()], device=decision_logprobs.device),
                    frame_offsets=torch.tensor([0, code_logprobs.shape[0]], device=decision_logprobs.device),
                    code_mask=code_mask,
                ).detach()
            )
        return result

    @staticmethod
    def _parity_max_abs(
        trainer_joint: list[torch.Tensor],
        server_joint: list[torch.Tensor],
        masks: list[torch.Tensor],
    ) -> torch.Tensor:
        device = trainer_joint[0].device
        max_abs = torch.zeros((), dtype=torch.float32, device=device)
        for trainer, server, mask in zip(trainer_joint, server_joint, masks, strict=True):
            if bool(mask.any()):
                max_abs = torch.maximum(max_abs, (trainer[mask] - server[mask]).abs().max())
        if dist.is_initialized():
            dist.all_reduce(
                max_abs, op=dist.ReduceOp.MAX, group=mpu.get_data_parallel_group(with_context_parallel=True)
            )
        return max_abs

    @staticmethod
    def _collected_parity_max_abs(parity_values: list[torch.Tensor]) -> torch.Tensor:
        if not parity_values:
            raise RuntimeError("MOSS-TTS Local same-forward replay did not collect parity values.")
        max_abs = torch.stack(parity_values).max()
        if dist.is_initialized():
            dist.all_reduce(
                max_abs,
                op=dist.ReduceOp.MAX,
                group=mpu.get_data_parallel_group(with_context_parallel=True),
            )
        return max_abs

    def _prepare_replay(self, context, rollout_data, data_iterator, num_microbatches) -> _ReplayPreparation:
        runtime = context.runtime
        async_mode = bool(getattr(context.args, "moss_local_async", False))
        rollout_versions = {str(version) for version in rollout_data["weight_versions"]}
        expected_version = runtime.published_version
        lag = (
            0
            if context.args.debug_train_only
            else policy_lag(rollout_versions, expected_version, allow_one_step=async_mode)
        )
        rollout_data["policy_lags"] = [lag] * len(rollout_data["rewards"])

        source = context.args.moss_local_old_policy_source
        reuse_train_forward = bool(
            source in {"trainer_preupdate", "trainer_behavior"}
            and getattr(context.args, "moss_local_reuse_train_forward", True)
            and runtime.rollout_data_postprocess is None
        )
        server_joint = self._server_joint_logprobs(rollout_data)
        rollout_data["server_joint_logprobs"] = server_joint
        if source == "trainer_behavior":
            started = time.perf_counter()
            behavior_version = next(iter(rollout_versions))
            with inverse_timer("train_wait"), timer("train"), timer("behavior_replay"):
                with runtime.behavior_snapshot(behavior_version):
                    behavior = self.compute_log_probs(
                        context, data_iterator, num_microbatches, store_prefix="behavior_"
                    )
            rollout_data["old_joint_logprobs"] = [value.detach() for value in behavior["behavior_joint_logprobs"]]
            rollout_data["behavior_snapshot_versions"] = [behavior_version] * len(server_joint)
            rollout_data["behavior_replay_seconds"] = time.perf_counter() - started
        parity_max_abs = None
        if not reuse_train_forward:
            trainer_outputs = self.compute_log_probs(
                context,
                data_iterator,
                num_microbatches,
                store_prefix="trainer_",
            )
            trainer_joint = trainer_outputs["trainer_joint_logprobs"]
            parity_max_abs = self._parity_max_abs(
                trainer_joint,
                server_joint,
                rollout_data["decision_masks"],
            )
            rollout_data.update(trainer_outputs)
            rollout_data["logprob_parity_max_abs"] = parity_max_abs.detach()

        if source == "server_behavior":
            assert parity_max_abs is not None
            threshold = float(context.args.moss_local_logprob_parity_max_abs)
            if float(parity_max_abs) > threshold:
                raise RuntimeError(
                    f"MOSS-TTS Local server/trainer logprob parity failed: max_abs={float(parity_max_abs):.6g} "
                    f"> threshold={threshold:.6g}. Use an explained trainer_preupdate surrogate or fix parity."
                )
            rollout_data["old_joint_logprobs"] = [value.detach() for value in server_joint]
        elif source == "trainer_preupdate":
            if not reuse_train_forward:
                rollout_data["old_joint_logprobs"] = [value.detach() for value in trainer_joint]
        elif source == "trainer_behavior":
            pass  # The matching version was replayed before restoring current actor weights.
        else:
            raise ValueError(f"Unknown MOSS-TTS Local old-policy source {source!r}.")

        return _ReplayPreparation(source, reuse_train_forward, parity_max_abs)

    @staticmethod
    def _prepare_advantages(context, rollout_data) -> None:
        runtime = context.runtime
        rollout_data["advantages"] = [
            torch.full(
                (decision_mask.numel(),),
                float(reward),
                dtype=torch.float32,
                device=decision_mask.device,
            )
            for reward, decision_mask in zip(rollout_data["rewards"], rollout_data["decision_masks"], strict=True)
        ]
        if getattr(context.args, "moss_local_objective", "wer_grpo") == "mopd":
            clip = context.args.moss_local_mopd_advantage_clip
            rollout_data["advantages"] = [
                (teacher - student).clamp(-clip, clip)
                for teacher, student in zip(
                    rollout_data["teacher_decision_logprobs"],
                    rollout_data["student_prefill_decision_logprobs"],
                    strict=True,
                )
            ]
            rollout_data["code_advantages"] = [
                (teacher.reshape(-1, 12) - student).clamp(-clip, clip)
                for teacher, student in zip(
                    rollout_data["teacher_code_logprobs"], rollout_data["student_prefill_code_logprobs"], strict=True
                )
            ]
        if runtime.rollout_data_postprocess is not None:
            runtime.rollout_data_postprocess(context.args)

    def train_actor(
        self,
        context: TrainingContext,
        rollout_id: int,
        rollout_data: RolloutBatch,
        *,
        external_data=None,
    ) -> None:
        if external_data is not None:
            raise ValueError("MOSS-TTS Local P0 does not accept critic/external actor data.")
        from miles.backends.megatron_utils.model import train

        runtime = context.runtime
        native_teachers = self._get_native_teachers(context.args)
        async_mode = bool(getattr(context.args, "moss_local_async", False))
        if async_mode:
            logger.info("MOSS async train start: rollout=%d timestamp=%.6f", rollout_id, time.time())
        data_iterator, num_microbatches = get_data_iterator(context.args, context.model, rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["num_rollouts"]
        _log_packing(context, rollout_data)
        replay = self._prepare_replay(context, rollout_data, data_iterator, num_microbatches)
        self._prepare_advantages(context, rollout_data)
        source, reuse_train_forward = replay.source, replay.reuse_train_forward
        parity_max_abs = replay.parity_max_abs

        trainer_output_collector: dict[str, list[torch.Tensor]] | None = {} if reuse_train_forward else None
        parity_values: list[torch.Tensor] | None = [] if reuse_train_forward else None
        with inverse_timer("train_wait"), timer("train"):
            outcome = train(
                rollout_id,
                context.model,
                context.optimizer,
                context.opt_param_scheduler,
                data_iterator,
                num_microbatches,
                global_batch_sizes,
                witness_info=None,
                attempt=0,
                custom_forward_step_builder=partial(
                    _make_train_forward_step,
                    trainer_output_collector=trainer_output_collector,
                    parity_values=parity_values,
                    native_teachers=native_teachers,
                ),
            )
        if outcome != TrainStepOutcome.NORMAL:
            raise RuntimeError(f"MOSS optimizer step did not complete: {outcome}")
        if reuse_train_forward:
            assert trainer_output_collector is not None
            assert parity_values is not None
            expected_samples = len(rollout_data["rewards"])
            trainer_joint = trainer_output_collector.get("trainer_joint_logprobs", [])
            if len(trainer_joint) != expected_samples:
                raise RuntimeError(
                    "MOSS-TTS Local same-forward replay output count mismatch: "
                    f"expected {expected_samples}, got {len(trainer_joint)}."
                )
            rollout_data.update(trainer_output_collector)
            if source == "trainer_preupdate":
                rollout_data["old_joint_logprobs"] = [value.detach() for value in trainer_joint]
            parity_max_abs = self._collected_parity_max_abs(parity_values)
            rollout_data["logprob_parity_max_abs"] = parity_max_abs.detach()
        _save_debug_train_data(context.args, rollout_id=rollout_id, rollout_data=rollout_data)
        cpu_backup_required = runtime.backup_required
        runtime.finish_rollout(rollout_id)
        if async_mode:
            logger.info("MOSS async train end: rollout=%d timestamp=%.6f", rollout_id, time.time())

        _log_training_metrics(
            context,
            rollout_id,
            rollout_data,
            parity_max_abs=parity_max_abs,
            source=source,
            cpu_backup_required=cpu_backup_required,
            num_microbatches=num_microbatches,
        )
        train_metric_utils.log_perf_data(
            rollout_id,
            context.args,
            extra_metrics=runtime.pop_weight_metrics(),
        )
        if native_teachers is not None and rollout_id == context.args.num_rollout - 1:
            native_teachers.verify_frozen()
