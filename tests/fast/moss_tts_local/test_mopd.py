from types import SimpleNamespace

import pytest
import torch

from miles.policies.moss_tts_local.mopd_client import payload_hash, teacher_routes
from miles.policies.moss_tts_local.mopd_loss import action_mopd_loss


def fixture():
    d = torch.tensor([-2.0, -2.0], requires_grad=True)
    c = torch.full((1, 12), -2.0, requires_grad=True)
    output = SimpleNamespace(decision_logprobs=d, code_logprobs=c)
    batch = SimpleNamespace(
        batch_size=1,
        decision_offsets=torch.tensor([0, 2]),
        frame_offsets=torch.tensor([0, 1]),
        decision_mask=torch.ones(2, dtype=torch.bool),
        code_mask=torch.ones((1, 12), dtype=torch.bool),
        rollout_decision_logprobs=d.detach().clone(),
        rollout_code_logprobs=c.detach().clone(),
    )
    return output, batch


def loss(output, batch, td, tc):
    return action_mopd_loss(output, batch, td, tc, advantage_clip=5.0, eps_clip=0.2, eps_clip_high=0.2)[0]


def test_identical_scores_zero_gradient():
    output, batch = fixture()
    result = loss(output, batch, output.decision_logprobs.detach(), output.code_logprobs.detach())
    result.loss.backward()
    assert torch.count_nonzero(output.decision_logprobs.grad) == 0
    assert torch.count_nonzero(output.code_logprobs.grad) == 0


def test_opposing_channels_keep_nonzero_gradients_and_teacher_frozen():
    output, batch = fixture()
    batch.decision_mask.zero_()
    batch.code_mask.zero_()
    batch.code_mask[0, :2] = True
    td = output.decision_logprobs.detach().clone().requires_grad_()
    tc = output.code_logprobs.detach().clone()
    tc[0, 0] += 1
    tc[0, 1] -= 1
    tc.requires_grad_()
    result = loss(output, batch, td, tc)
    result.loss.backward()
    assert output.code_logprobs.grad[0, 0] < 0 and output.code_logprobs.grad[0, 1] > 0
    assert tc.grad is None and td.grad is None
    assert torch.count_nonzero(output.code_logprobs.grad[0, 2:]) == 0


def test_inactive_nan_scores_are_masked_before_nonlinear_ops():
    output, batch = fixture()
    batch.code_mask.zero_()
    result = loss(output, batch, output.decision_logprobs.detach(), torch.full((1, 12), float("nan")))
    result.loss.backward()
    assert torch.isfinite(output.code_logprobs.grad).all()


def test_teacher_routes_are_explicit_and_unique():
    assert teacher_routes(["en=http://a/score_actions", "zh=http://b/score_actions"])["en"] == "http://a/score_actions"
    with pytest.raises(ValueError):
        teacher_routes(["en=http://a", "en=http://b"])


def test_payload_identity_tracks_actions():
    a = {"sample_id": "1", "codes": [[1]], "temperature": 1.0}
    assert payload_hash(a) == payload_hash(a | {"sample_id": "2"})
    assert payload_hash(a) != payload_hash(a | {"codes": [[2]]})


def test_matched_prefill_scores_do_not_train_on_behavior_path_noise():
    output, batch = fixture()
    teacher_d = output.decision_logprobs.detach() + .1
    teacher_c = output.code_logprobs.detach() + .2
    result, _ = action_mopd_loss(
        output, batch, teacher_d, teacher_c, advantage_clip=5.,
        eps_clip=.2, eps_clip_high=.2, student_scores=(teacher_d.clone(), teacher_c.clone()),
    )
    result.loss.backward()
    assert torch.count_nonzero(output.code_logprobs.grad) == 0
    assert torch.count_nonzero(output.decision_logprobs.grad) == 0
