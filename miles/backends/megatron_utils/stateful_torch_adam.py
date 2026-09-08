"""Stateful Torch Adam/AdamW matching the validated MOSS training recipe."""

from __future__ import annotations

import math
from typing import Any

import torch


class StatefulTorchAdam(torch.optim.Optimizer):
    """Pure PyTorch Adam compatible with Megatron's Transformer-Engine constructor.

    This preserves the optimizer used by the validated Slime MOSS recipe
    while accepting Megatron's Transformer-Engine constructor keywords.
    It retains Adam first/second moments and supports Megatron's distributed
    checkpoint convention of storing the update step on parameter groups.

    Master parameters remain owned by Megatron's optimizer wrapper.  Moment
    tensors default to the parameter dtype unless Megatron explicitly requests
    ``exp_avg_dtype`` / ``exp_avg_sq_dtype`` (normally fp32).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        *,
        bias_correction: bool = True,
        adam_w_mode: bool = True,
        maximize: bool = False,
        use_decoupled_grad: bool = False,
        master_weights: bool = False,
        exp_avg_dtype: torch.dtype | None = None,
        exp_avg_sq_dtype: torch.dtype | None = None,
        set_grad_none: bool | None = None,
        **_: Any,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if amsgrad:
            raise NotImplementedError("StatefulTorchAdam does not support amsgrad.")
        if master_weights:
            raise NotImplementedError("StatefulTorchAdam relies on Megatron's optimizer wrapper for master weights.")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            bias_correction=bias_correction,
            adam_w_mode=adam_w_mode,
            maximize=maximize,
            use_decoupled_grad=use_decoupled_grad,
            exp_avg_dtype=exp_avg_dtype,
            exp_avg_sq_dtype=exp_avg_sq_dtype,
            set_grad_none=True if set_grad_none is None else set_grad_none,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            group.setdefault("step", 0)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Megatron's TE-compatible distributed checkpoint stores the step on
        # groups and restores only the per-parameter moment tensors.
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if state and "step" not in state:
                    state["step"] = int(group.get("step", 0))

    @staticmethod
    def _state_zeros_like(param: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
        return torch.zeros_like(param, dtype=dtype or param.dtype, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            adam_w_mode = group.get("adam_w_mode", True)
            bias_correction = group.get("bias_correction", True)
            maximize = group.get("maximize", False)
            use_decoupled_grad = group.get("use_decoupled_grad", False)
            exp_avg_dtype = group.get("exp_avg_dtype")
            exp_avg_sq_dtype = group.get("exp_avg_sq_dtype")
            group["step"] = int(group.get("step", 0)) + 1

            for param in group["params"]:
                grad = getattr(param, "decoupled_grad", None) if use_decoupled_grad else param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("StatefulTorchAdam does not support sparse gradients.")

                grad_for_update = grad.neg() if maximize else grad
                if weight_decay != 0 and adam_w_mode:
                    param.mul_(1.0 - lr * weight_decay)
                elif weight_decay != 0:
                    grad_for_update = grad_for_update.add(param, alpha=weight_decay)

                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = self._state_zeros_like(param, exp_avg_dtype)
                    state["exp_avg_sq"] = self._state_zeros_like(param, exp_avg_sq_dtype)
                state["step"] = int(state["step"]) + 1

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state_grad = grad_for_update.to(dtype=exp_avg.dtype)
                exp_avg.mul_(beta1).add_(state_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    state_grad.to(dtype=exp_avg_sq.dtype),
                    state_grad.to(dtype=exp_avg_sq.dtype),
                    value=1.0 - beta2,
                )

                if bias_correction:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = lr / bias_correction1
                    denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                else:
                    step_size = lr
                    denom = exp_avg_sq.sqrt().add_(eps)
                param.addcdiv_(
                    exp_avg.to(dtype=param.dtype),
                    denom.to(dtype=param.dtype),
                    value=-step_size,
                )

        return loss
