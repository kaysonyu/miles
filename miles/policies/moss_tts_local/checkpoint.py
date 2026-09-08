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
