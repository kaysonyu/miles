from types import SimpleNamespace

import pytest
import torch

from miles.policies.moss_tts_local.dense_mopd import (
    NativeTeacherPool,
    categorical_kl,
    dense_action_kl,
    native_action_entropies,
    native_teacher_sources,
)


@pytest.mark.parametrize("direction", ["dense_reverse", "dense_forward"])
def test_identical_distributions_have_zero_loss_and_gradient(direction):
    student = torch.tensor([[1.0, -1.0, 0.0]], requires_grad=True)
    teacher = student.detach().clone().requires_grad_()
    loss = categorical_kl(student, teacher, temperature=1.0, direction=direction).sum()
    loss.backward()
    assert loss.item() == 0
    assert student.grad.abs().max() < 1e-7
    assert teacher.grad is None


@pytest.mark.parametrize("direction", ["dense_reverse", "dense_forward"])
def test_dense_gradient_matches_complete_action_expectation(direction):
    student = torch.tensor([[0.8, -0.3, 0.1]], requires_grad=True)
    teacher = torch.tensor([[-0.2, 1.4, 0.3]], requires_grad=True)
    loss = categorical_kl(student, teacher, temperature=1.0, direction=direction).sum()
    gradient = torch.autograd.grad(loss, student)[0]
    p, q = student.softmax(-1).detach(), teacher.softmax(-1).detach()
    if direction == "dense_reverse":
        log_ratio = student.log_softmax(-1).detach() - teacher.log_softmax(-1).detach()
        expected = p * (log_ratio - (p * log_ratio).sum(-1, keepdim=True))
    else:
        expected = p - q
    torch.testing.assert_close(gradient, expected, atol=1e-7, rtol=1e-6)
    assert gradient[0, 1] < 0  # Improve the teacher-preferred alternative, even if it was not sampled.


def test_sample_weighting_terminal_stop_and_inactive_nan_masks():
    d = torch.zeros((4, 2), requires_grad=True)
    c = torch.zeros((2, 12, 3), requires_grad=True)
    td = torch.tensor([[1.0, -1.0]] * 4, requires_grad=True)
    tc = torch.full((2, 12, 3), float("nan"), requires_grad=True)
    batch = SimpleNamespace(
        batch_size=2,
        decision_offsets=torch.tensor([0, 1, 4]),
        frame_offsets=torch.tensor([0, 0, 2]),
        decision_mask=torch.ones(4, dtype=torch.bool),
        code_mask=torch.zeros((2, 12), dtype=torch.bool),
    )
    output = SimpleNamespace(decision_logits=d, code_logits=c)
    value = dense_action_kl(output, batch, td, tc, temperature=1.0, direction="dense_reverse")
    one = categorical_kl(d[:1], td[:1], temperature=1.0, direction="dense_reverse").sum()
    torch.testing.assert_close(value, 2 * one)
    value.backward()
    assert torch.isfinite(d.grad).all() and torch.count_nonzero(c.grad) == 0
    assert td.grad is None and tc.grad is None


def test_native_teacher_routes_reject_duplicates_and_relative_paths():
    assert set(native_teacher_sources(["dialect=/a", "instruction=/b"])) == {"dialect", "instruction"}
    for entries in [[], ["dialect=relative"], ["dialect=/a", "dialect=/b"]]:
        with pytest.raises(ValueError):
            native_teacher_sources(entries)


def test_entropy_observation_uses_action_weighting_without_gradients():
    decisions = torch.zeros((3, 2), requires_grad=True)
    codes = torch.zeros((1, 12, 3), requires_grad=True)
    output = SimpleNamespace(decision_logits=decisions, code_logits=codes)
    batch = SimpleNamespace(
        batch_size=2,
        decision_offsets=torch.tensor([0, 1, 3]),
        frame_offsets=torch.tensor([0, 0, 1]),
        decision_mask=torch.ones(3, dtype=torch.bool),
        code_mask=torch.ones((1, 12), dtype=torch.bool),
    )
    values = native_action_entropies(output, batch, decisions, codes, temperature=0.7)
    log2, log3 = torch.tensor(2.0).log(), torch.tensor(3.0).log()
    for label in ["student", "teacher"]:
        torch.testing.assert_close(values[f"mopd_{label}_action_entropy"], log2 + (2 * log2 + 12 * log3) / 14)
        torch.testing.assert_close(values[f"mopd_{label}_rvq1_entropy"], log3)
        assert not values[f"mopd_{label}_action_entropy"].requires_grad
    batch.code_mask.zero_()
    masked = SimpleNamespace(decision_logits=decisions, code_logits=torch.full_like(codes, float("nan")))
    values = native_action_entropies(masked, batch, decisions, masked.code_logits, temperature=1.0)
    torch.testing.assert_close(values["mopd_student_action_entropy"], 2 * log2)
    assert values["mopd_teacher_rvq1_entropy"] == 0


def test_mixed_domain_scores_preserve_packed_sample_routes_and_stop_only_sample():
    batch = SimpleNamespace(
        batch_size=3,
        decision_offsets=torch.tensor([0, 1, 4, 6]),
        frame_offsets=torch.tensor([0, 0, 2, 3]),
    )
    decisions = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    codes = torch.arange(108, dtype=torch.float32).reshape(3, 12, 3)
    calls = []

    def teacher(domain, offset):
        def score(*, policy_batch, with_logits):
            assert policy_batch is batch and with_logits and not torch.is_grad_enabled()
            calls.append(domain)
            return SimpleNamespace(decision_logits=decisions + offset, code_logits=codes + offset)

        return score

    pool = SimpleNamespace(models={"dialect": teacher("dialect", 10), "instruction": teacher("instruction", 1000)})
    routed_decisions, routed_codes = NativeTeacherPool.score(pool, batch, ["dialect", "instruction", "dialect"])
    expected_decisions, expected_codes = decisions + 10, codes + 10
    expected_decisions[1:4] += 990
    expected_codes[:2] += 990
    torch.testing.assert_close(routed_decisions, expected_decisions)
    torch.testing.assert_close(routed_codes, expected_codes)
    assert calls == ["dialect", "instruction"]
    for domains in [["dialect"], ["dialect", "missing", "dialect"]]:
        with pytest.raises(ValueError):
            NativeTeacherPool.score(pool, batch, domains)


def test_teacher_models_are_reused_across_rollouts(monkeypatch):
    from miles.policies.moss_tts_local import dense_mopd

    created = []

    def build(args):
        pool = object()
        created.append(pool)
        return pool

    monkeypatch.setattr(dense_mopd, "NativeTeacherPool", build)
    actor = SimpleNamespace()
    args = SimpleNamespace(moss_local_mopd_estimator="dense_reverse")
    assert dense_mopd.get_native_teacher_pool(args, actor) is dense_mopd.get_native_teacher_pool(args, actor)
    assert len(created) == 1
    assert dense_mopd.get_native_teacher_pool(SimpleNamespace(moss_local_mopd_estimator="sampled"), actor) is None


def test_prefix_weight_preserves_decision_gradient_and_resets_each_sample():
    d = torch.zeros((8, 2), requires_grad=True)
    c = torch.zeros((6, 12, 3), requires_grad=True)
    td = torch.tensor([[1.0, -1.0]] * 8)
    tc = torch.tensor([[[1.0, -1.0, 0.0]] * 12] * 6)
    batch = SimpleNamespace(
        batch_size=2,
        decision_offsets=torch.tensor([0, 4, 8]),
        frame_offsets=torch.tensor([0, 3, 6]),
        decision_mask=torch.ones(8, dtype=torch.bool),
        code_mask=torch.ones((6, 12), dtype=torch.bool),
    )
    output = SimpleNamespace(decision_logits=d, code_logits=c)
    kwargs = dict(temperature=0.7, direction="dense_reverse")
    original = dense_action_kl(output, batch, td, tc, **kwargs)
    original_d, original_c = torch.autograd.grad(original, (d, c), retain_graph=True)
    weighted = dense_action_kl(output, batch, td, tc, prefix_frames=1, prefix_weight=4, **kwargs)
    weighted_d, weighted_c = torch.autograd.grad(weighted, (d, c), retain_graph=True)
    torch.testing.assert_close(weighted, original)
    torch.testing.assert_close(weighted_d, original_d)
    torch.testing.assert_close(weighted_c[[0, 3]], original_c[[0, 3]] * 2)
    torch.testing.assert_close(weighted_c[[1, 2, 4, 5]], original_c[[1, 2, 4, 5]] / 2)
    torch.testing.assert_close(weighted_c.sum(0), original_c.sum(0))
    for frames, weight in [(0, 4), (8, 1)]:
        same = dense_action_kl(output, batch, td, tc, prefix_frames=frames, prefix_weight=weight, **kwargs)
        assert torch.equal(same, original)
        for before, after in zip((original_d, original_c), torch.autograd.grad(same, (d, c), retain_graph=True)):
            assert torch.equal(before, after)
    batch.code_mask.zero_()
    masked = dense_action_kl(output, batch, td, tc * float("nan"), prefix_frames=8, prefix_weight=4, **kwargs)
    assert torch.isfinite(masked)


@pytest.mark.parametrize("books", [[4.0] + [1.0] * 11, [4.0] * 3 + [1.0] * 9])
def test_codebook_weighting_preserves_total_rvq_and_decision_gradients(books):
    d = torch.zeros((4, 2), requires_grad=True)
    c = torch.zeros((3, 12, 2), requires_grad=True)
    td = torch.tensor([[1.0, -1.0]] * 4)
    tc = torch.tensor([[[1.0, -1.0]] * 12] * 3)
    batch = SimpleNamespace(
        batch_size=1,
        decision_offsets=torch.tensor([0, 4]),
        frame_offsets=torch.tensor([0, 3]),
        decision_mask=torch.ones(4, dtype=torch.bool),
        code_mask=torch.ones((3, 12), dtype=torch.bool),
    )
    output = SimpleNamespace(decision_logits=d, code_logits=c)
    kwargs = dict(temperature=0.7, direction="dense_reverse")
    old = dense_action_kl(output, batch, td, tc, **kwargs)
    old_d, old_c = torch.autograd.grad(old, (d, c), retain_graph=True)
    new = dense_action_kl(output, batch, td, tc, codebook_weights=books, **kwargs)
    new_d, new_c = torch.autograd.grad(new, (d, c), retain_graph=True)
    torch.testing.assert_close(old, new)
    torch.testing.assert_close(old_d, new_d)
    expected = torch.tensor(books).reshape(1, 12, 1) * (12 / sum(books))
    torch.testing.assert_close(new_c, old_c * expected)
    torch.testing.assert_close(new_c.sum(1), old_c.sum(1))
    same = dense_action_kl(output, batch, td, tc, codebook_weights=[1.0] * 12, decision_weight=1, **kwargs)
    assert torch.equal(old, same)


def test_decision_weight_changes_relative_gradient_with_unit_total_coefficient():
    d = torch.zeros((4, 2), requires_grad=True)
    c = torch.zeros((3, 12, 2), requires_grad=True)
    td = torch.tensor([[1.0, -1.0]] * 4)
    tc = torch.tensor([[[1.0, -1.0]] * 12] * 3)
    batch = SimpleNamespace(
        batch_size=1,
        decision_offsets=torch.tensor([0, 4]),
        frame_offsets=torch.tensor([0, 3]),
        decision_mask=torch.ones(4, dtype=torch.bool),
        code_mask=torch.ones((3, 12), dtype=torch.bool),
    )
    output = SimpleNamespace(decision_logits=d, code_logits=c)
    kwargs = dict(temperature=1, direction="dense_forward")
    old = dense_action_kl(output, batch, td, tc, **kwargs)
    old_d, old_c = torch.autograd.grad(old, (d, c), retain_graph=True)
    weighted = dense_action_kl(output, batch, td, tc, decision_weight=4, **kwargs)
    new_d, new_c = torch.autograd.grad(weighted, (d, c), retain_graph=True)
    torch.testing.assert_close(weighted, old)
    torch.testing.assert_close(new_d, old_d * (40 / 13))
    torch.testing.assert_close(new_c, old_c * (10 / 13))
    batch.code_mask.zero_()
    masked = dense_action_kl(
        output, batch, td, tc * float("nan"), codebook_weights=[4] + [1] * 11, decision_weight=4, **kwargs
    )
    assert torch.isfinite(masked)
    assert torch.count_nonzero(torch.autograd.grad(masked, c)[0]) == 0


def test_small_positive_weights_retain_normalized_scale_on_terminal_only_and_codes():
    d = torch.zeros((3, 2), requires_grad=True)
    c = torch.zeros((1, 12, 2), requires_grad=True)
    td = torch.tensor([[1.0, -1.0]] * 3)
    tc = torch.tensor([[[1.0, -1.0]] * 12])
    batch = SimpleNamespace(
        batch_size=2,
        decision_offsets=torch.tensor([0, 1, 3]),
        frame_offsets=torch.tensor([0, 0, 1]),
        decision_mask=torch.ones(3, dtype=torch.bool),
        code_mask=torch.ones((1, 12), dtype=torch.bool),
    )
    output = SimpleNamespace(decision_logits=d, code_logits=c)
    old = dense_action_kl(output, batch, td, tc, temperature=1, direction="dense_reverse")
    new = dense_action_kl(
        output,
        batch,
        td,
        tc,
        temperature=1,
        direction="dense_reverse",
        codebook_weights=[0.01] * 12,
        decision_weight=0.1,
    )
    torch.testing.assert_close(old, new)
    assert torch.isfinite(torch.autograd.grad(new, d)[0]).all()
