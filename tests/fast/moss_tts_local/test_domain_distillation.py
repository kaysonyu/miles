import json
from types import SimpleNamespace

import pytest
import torch
from miles.policies.moss_tts_local.dense_mopd import categorical_kl, dense_action_kl

from miles.policies.moss_tts_local import domain_distillation as domain


def inputs():
    output = SimpleNamespace(
        decision_logits=torch.tensor([[0.3, -0.3], [0.8, -0.8], [-0.2, 0.2]], requires_grad=True),
        code_logits=torch.zeros((1, 12, 3), requires_grad=True),
    )
    teacher_d = torch.tensor([[1.0, -1.0]] * 3, requires_grad=True)
    teacher_c = torch.tensor([[[0.0, 1.0, -1.0]] * 12], requires_grad=True)
    batch = SimpleNamespace(
        batch_size=2,
        decision_offsets=torch.tensor([0, 2, 3]),
        frame_offsets=torch.tensor([0, 1, 1]),
        decision_mask=torch.ones(3, dtype=torch.bool),
        code_mask=torch.ones((1, 12), dtype=torch.bool),
    )
    return output, batch, teacher_d, teacher_c


def test_no_config_is_exact_existing_loss_and_gradient():
    output, batch, td, tc = inputs()
    args = SimpleNamespace(rollout_temperature=0.7, moss_local_mopd_estimator="dense_reverse")
    a = dense_action_kl(output, batch, td, tc, temperature=0.7, direction="dense_reverse")
    b = domain.action_kl(args, output, batch, td, tc, None)
    assert torch.equal(a, b)
    ga = torch.autograd.grad(a, (output.decision_logits, output.code_logits), retain_graph=True)
    gb = torch.autograd.grad(b, (output.decision_logits, output.code_logits), retain_graph=True)
    assert all(torch.equal(x, y) for x, y in zip(ga, gb))


def test_domain_direction_gradient_and_terminal_only_sample(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "dialect": {"estimator": "dense_reverse", "codebook_weights": [4] * 3 + [1] * 9},
                "instruction": {"estimator": "dense_forward"},
            }
        )
    )
    args = SimpleNamespace(
        rollout_temperature=1.0,
        moss_local_mopd_estimator="dense_reverse",
        moss_local_domain_loss_specs=domain.load_specs(path),
    )
    output, batch, td, tc = inputs()
    loss = domain.action_kl(args, output, batch, td, tc, ["dialect", "instruction"])
    gradient = torch.autograd.grad(loss, output.decision_logits, retain_graph=True)[0]
    p = output.decision_logits.detach().softmax(-1)
    q = td.detach().softmax(-1)
    torch.testing.assert_close(gradient[2], p[2] - q[2])
    log_ratio = output.decision_logits.detach().log_softmax(-1) - td.detach().log_softmax(-1)
    expected = p[:2] * (log_ratio[:2] - (p[:2] * log_ratio[:2]).sum(-1, keepdim=True)) / 14
    torch.testing.assert_close(gradient[:2], expected)
    assert td.grad is None and tc.grad is None
    with pytest.raises(ValueError, match="matching"):
        domain.action_kl(args, output, batch, td, tc, ["dialect", "missing"])
    with pytest.raises(ValueError, match="matching"):
        domain.action_kl(args, output, batch, td, tc, None)


@pytest.mark.parametrize(
    "bad",
    [
        {"estimator": "sampled"},
        {"estimator": "dense_reverse", "typo": 4},
        {"estimator": "dense_reverse", "decision_weight": 0},
        {"estimator": "dense_reverse", "codebook_weights": [1] * 11},
        {"estimator": "dense_reverse", "prefix_frames": -1},
    ],
)
def test_invalid_settings_are_rejected(tmp_path, bad):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"dialect": bad}))
    with pytest.raises(ValueError):
        domain.load_specs(path)
