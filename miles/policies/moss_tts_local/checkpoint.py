"""Model-only pretrained load, distinct from resuming a Miles checkpoint."""

from pathlib import Path

import torch.distributed as dist
from megatron.core.utils import unwrap_model

from miles.utils.function_registry import load_function


def has_resume_checkpoint(path):
    if not path:
        return False
    root = Path(path)
    return (root / "latest_checkpointed_iteration.txt").exists() or (root / "common.pt").exists()


def validate_optimizer_backend_resume(saved_impl, current_impl, sharding_type):
    """Bucket-based optimizer state cannot follow a different parameter layout."""
    if saved_impl == current_impl:
        return
    if sharding_type not in {"fully_reshardable", "fully_sharded_model_space"}:
        raise ValueError(
            f"Cannot resume MOSS optimizer from {saved_impl!r} into {current_impl!r} "
            f"with {sharding_type!r} checkpoint layout. Reload with the original "
            "transformer implementation and save with --dist-ckpt-optim-fully-reshardable "
            "before switching, or explicitly start fresh optimizer state with --finetune."
        )


def validate_resume_implementation(args):
    if getattr(args, "policy_family", None) != "moss_tts_local":
        return
    if args.finetune or args.no_load_optim or not args.use_distributed_optimizer:
        return
    # Checkpoint readers are needed only for a distributed optimizer resume.
    from megatron.core import dist_checkpointing
    from megatron.training.checkpointing import get_load_checkpoint_path_by_args

    checkpoint_root = get_load_checkpoint_path_by_args(args, load_arg="load")
    common = dist_checkpointing.load_common_state_dict(checkpoint_root)
    saved_args = common.get("args")
    if getattr(saved_args, "no_save_optim", False):
        return
    metadata = dist_checkpointing.load_content_metadata(preloaded_state_dict=common) or {}
    validate_optimizer_backend_resume(
        getattr(saved_args, "transformer_impl", None),
        args.transformer_impl,
        metadata.get("distrib_optim_sharding_type"),
    )


def load_pretrained(args, model, optimizer, checkpointing_context, *, skip_load_to_model_and_opt):
    if skip_load_to_model_and_opt or getattr(args, "load_main_params_from_ckpt", False):
        raise ValueError("MOSS native loader does not support FSDP in-place or master-parameter loading")
    loader = load_function(args.custom_pretrained_checkpoint_loader_path)
    previous_finetune = args.finetune
    args.finetune = True
    try:
        result = loader(args, unwrap_model(model), {} if checkpointing_context is None else checkpointing_context)
        if result is not None:
            raise TypeError("A pretrained loader must return None")
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()
        if dist.is_initialized():
            dist.barrier()
    except BaseException:
        args.finetune = previous_finetune
        raise
    return 0, 0
