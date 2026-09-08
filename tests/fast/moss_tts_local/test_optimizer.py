from __future__ import annotations

import copy

import pytest
import torch

from miles.backends.megatron_utils.stateful_torch_adam import StatefulTorchAdam

NUM_GPUS = 0


@pytest.mark.parametrize("adam_w_mode", [True, False])
def test_stateful_torch_adam_matches_torch_over_multiple_steps(adam_w_mode):
    torch.manual_seed(11)
    initial = torch.randn(17, dtype=torch.float64)
    grads = [torch.randn_like(initial) for _ in range(5)]
    actual = initial.clone()
    expected = initial.clone()
    optimizer = StatefulTorchAdam(
        [actual],
        lr=0.03,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.1,
        adam_w_mode=adam_w_mode,
    )
    expected_optimizer_cls = torch.optim.AdamW if adam_w_mode else torch.optim.Adam
    expected_optimizer = expected_optimizer_cls(
        [expected],
        lr=0.03,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.1,
    )

    for grad in grads:
        actual.grad = grad.clone()
        expected.grad = grad.clone()
        optimizer.step()
        expected_optimizer.step()
        optimizer.zero_grad()
        expected_optimizer.zero_grad()

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_stateful_torch_adam_persists_and_round_trips_moments():
    param = torch.tensor([1.0, -2.0], dtype=torch.float64)
    optimizer = StatefulTorchAdam([param], lr=0.01)
    param.grad = torch.tensor([0.3, -0.4], dtype=torch.float64)
    optimizer.step()

    state = optimizer.state[param]
    assert state["step"] == 1
    assert torch.count_nonzero(state["exp_avg"]) == 2
    assert torch.count_nonzero(state["exp_avg_sq"]) == 2

    saved_param = param.detach().clone()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    restored_param = saved_param.clone()
    restored = StatefulTorchAdam([restored_param], lr=0.01)
    restored.load_state_dict(saved_optimizer)
    next_grad = torch.tensor([-0.2, 0.7], dtype=torch.float64)
    param.grad = next_grad.clone()
    restored_param.grad = next_grad.clone()
    optimizer.step()
    restored.step()

    torch.testing.assert_close(restored_param, param)
    assert restored.state[restored_param]["step"] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def test_megatron_group_step_resume_matches_uninterrupted_update():
    parameter = torch.tensor([1.0, 2.0], dtype=torch.float64)
    optimizer = StatefulTorchAdam([parameter], lr=0.01)
    for _ in range(3):
        parameter.grad = torch.tensor([0.3, -0.4], dtype=torch.float64)
        optimizer.step()
    checkpoint = copy.deepcopy(optimizer.state_dict())
    for state in checkpoint["state"].values():
        del state["step"]
    resumed_parameter = parameter.detach().clone()
    resumed = StatefulTorchAdam([resumed_parameter], lr=0.01)
    resumed.load_state_dict(checkpoint)
    resumed_parameter.grad = parameter.grad.clone()
    optimizer.step()
    resumed.step()
    torch.testing.assert_close(resumed_parameter, parameter, rtol=0, atol=0)
    assert resumed.state[resumed_parameter]["step"] == 4
