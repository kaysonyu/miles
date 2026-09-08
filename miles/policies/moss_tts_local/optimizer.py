"""Use the validated stateful Torch Adam with Megatron's optimizer wrapper."""

from contextlib import contextmanager

from miles.backends.megatron_utils.stateful_torch_adam import StatefulTorchAdam


@contextmanager
def torch_adam_context(enabled):
    if not enabled:
        yield
        return
    # Megatron resolves these implementation classes when constructing the wrapper.
    import megatron.core.optimizer as optimizer_module
    import megatron.core.optimizer.distrib_optimizer as distributed_module

    saved = []
    for module in (optimizer_module, distributed_module):
        for name in ("Adam", "CPUAdam"):
            if hasattr(module, name):
                saved.append((module, name, getattr(module, name)))
                setattr(module, name, StatefulTorchAdam)
    try:
        yield
    finally:
        for module, name, value in reversed(saved):
            setattr(module, name, value)
