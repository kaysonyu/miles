"""Local optimizer resume checks and compatibility imports for model-only loading."""

from miles.backends.megatron_utils.pretrained_loader import has_resume_checkpoint as has_resume_checkpoint
from miles.backends.megatron_utils.pretrained_loader import load_pretrained as load_pretrained


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
