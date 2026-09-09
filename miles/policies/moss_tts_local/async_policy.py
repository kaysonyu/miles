"""Explicit bounds and versioned trainer snapshots for Local asynchronous RL."""

from __future__ import annotations


def validate_configuration(args):
    enabled = getattr(args, "moss_local_async", False)
    source = getattr(args, "moss_local_old_policy_source", "trainer_preupdate")
    if getattr(args, "fully_async", False):
        raise ValueError("Local supports the one-batch-ahead driver, not the text --fully-async worker")
    if not enabled and source != "trainer_behavior":
        return
    if source != "trainer_behavior" or not getattr(args, "keep_old_actor", False):
        raise ValueError("Local async requires trainer_behavior and --keep-old-actor")
    if getattr(args, "update_weights_interval", 1) != 1:
        raise ValueError("Local versioned behavior replay requires update_weights_interval=1")
    batch = getattr(args, "global_batch_size", None)
    prompts = getattr(args, "rollout_batch_size", None)
    samples = getattr(args, "n_samples_per_prompt", None)
    if all(value is not None for value in (batch, prompts, samples)) and batch != prompts * samples:
        raise ValueError("Local async requires exactly one optimizer step per rollout batch")
    if getattr(args, "moss_local_objective", "wer_grpo") != "wer_grpo":
        raise ValueError("Local async currently validates WER only; MOPD needs separate scorer scheduling")
    if not getattr(args, "moss_local_reuse_train_forward", True):
        raise ValueError("Local behavior replay requires --moss-local-reuse-train-forward")
    for option in ("offload_train", "rematerialize_param_from_master_weight", "overlap_param_gather"):
        if getattr(args, option, False):
            raise ValueError(f"Local versioned snapshot replay does not support {option}")
    if getattr(args, "rollout_data_postprocess_path", None):
        raise ValueError("Local versioned snapshot replay requires the standard rollout postprocessing")
    if getattr(args, "debug_train_only", False) or getattr(args, "debug_skip_weight_update", False):
        raise ValueError("Local versioned behavior replay requires live weight publication")
    if enabled:
        total, interval = getattr(args, "num_rollout", None), getattr(args, "save_interval", None)
        if total is None or total < 1:
            raise ValueError("Local async requires an explicit positive --num-rollout")
        if interval is not None and interval < total:
            raise ValueError("Local async currently checkpoints only after the final batch is drained")
        if getattr(args, "save_trigger_sentinel", None) or getattr(args, "debug_exit_after_rollout", None):
            raise ValueError("Local async cannot checkpoint/exit with an unpersisted prefetched batch")


def policy_lag(versions, published_version, *, allow_one_step):
    versions = {str(value) for value in versions}
    if len(versions) != 1 or not all(value.isdecimal() for value in versions):
        raise ValueError(f"Local batch needs one numeric behavior version, got {versions}")
    version = int(next(iter(versions)))
    lag = int(published_version) - version
    if version < 1 or lag < 0 or lag > int(allow_one_step):
        raise ValueError(f"Local policy version mismatch: behavior={version}, published={published_version}, lag={lag}")
    return lag


def record_snapshot_versions(actor, published_version):
    """Called after the standard rollout_actor -> old_actor snapshot rotation."""
    current = int(published_version)
    labels = getattr(actor, "_moss_snapshot_versions", {})
    previous = labels.get("rollout_actor", current)
    if current < 1 or (labels and current != previous + 1):
        raise ValueError("Local behavior snapshots require consecutive published versions")
    actor._moss_snapshot_versions = {"old_actor": previous, "rollout_actor": current}


def behavior_snapshot_tag(actor, behavior_version):
    version = int(behavior_version)
    for tag, value in getattr(actor, "_moss_snapshot_versions", {}).items():
        if value == version:
            return tag
    raise ValueError(f"No retained trainer snapshot matches behavior version {version}")


def replay_behavior(workflow, context, data_iterator, num_microbatches, behavior_version):
    """Replay old probabilities and restore current weights before any backward."""
    actor = context.actor
    tag = behavior_snapshot_tag(actor, behavior_version)
    try:
        actor._switch_model(tag)
        return workflow.compute_log_probs(context, data_iterator, num_microbatches, store_prefix="behavior_")
    finally:
        actor._switch_model("actor")
