from types import SimpleNamespace

import pytest

from miles.policies.moss_tts_local.async_policy import (
    behavior_snapshot_tag,
    policy_lag,
    record_snapshot_versions,
    replay_behavior,
    validate_configuration,
)


def config(**overrides):
    return SimpleNamespace(**({
        "moss_local_async": True,
        "moss_local_old_policy_source": "trainer_behavior",
        "keep_old_actor": True,
        "update_weights_interval": 1,
        "moss_local_objective": "wer_grpo",
        "num_rollout": 32,
        "save_interval": 32,
    } | overrides))


def test_async_accepts_only_bounded_explicit_behavior_configuration():
    validate_configuration(config())
    for override in [
        {"keep_old_actor": False},
        {"moss_local_old_policy_source": "trainer_preupdate"},
        {"update_weights_interval": 2},
        {"fully_async": True},
        {"moss_local_objective": "mopd"},
        {"save_interval": 8},
        {"save_trigger_sentinel": "save-now"},
        {"debug_exit_after_rollout": 2},
        {"offload_train": True},
        {"rematerialize_param_from_master_weight": True},
        {"overlap_param_gather": True},
        {"global_batch_size": 32, "rollout_batch_size": 16, "n_samples_per_prompt": 4},
    ]:
        with pytest.raises(ValueError):
            validate_configuration(config(**override))


def test_default_sync_does_not_need_async_options():
    validate_configuration(SimpleNamespace())
    assert policy_lag(["3", "3"], "3", allow_one_step=False) == 0
    with pytest.raises(ValueError):
        policy_lag(["2"], "3", allow_one_step=False)


@pytest.mark.parametrize("versions,current", [(["1", "2"], 2), (["1"], 3), (["3"], 2), (["default"], 2), (["0"], 1)])
def test_async_rejects_mixed_excessive_future_or_unpublished_versions(versions, current):
    with pytest.raises(ValueError):
        policy_lag(versions, current, allow_one_step=True)


def test_snapshot_rotation_matches_one_ahead_behavior_versions():
    actor = SimpleNamespace()
    record_snapshot_versions(actor, 1)
    assert behavior_snapshot_tag(actor, 1) == "old_actor"
    record_snapshot_versions(actor, 2)
    assert actor._moss_snapshot_versions == {"old_actor": 1, "rollout_actor": 2}
    assert policy_lag(["1"], 2, allow_one_step=True) == 1
    assert behavior_snapshot_tag(actor, 1) == "old_actor"
    assert behavior_snapshot_tag(actor, 2) == "rollout_actor"
    record_snapshot_versions(actor, 3)
    with pytest.raises(ValueError):
        behavior_snapshot_tag(actor, 1)
    with pytest.raises(ValueError):
        record_snapshot_versions(actor, 5)


def test_replay_restores_actor_even_when_old_forward_fails():
    switches = []
    actor = SimpleNamespace(_moss_snapshot_versions={"old_actor": 1, "rollout_actor": 2}, _switch_model=switches.append)

    def fail(*args, **kwargs):
        assert kwargs["store_prefix"] == "behavior_"
        raise RuntimeError("probe failure")

    with pytest.raises(RuntimeError, match="probe failure"):
        replay_behavior(SimpleNamespace(compute_log_probs=fail), SimpleNamespace(actor=actor), None, None, 1)
    assert switches == ["old_actor", "actor"]


def test_behavior_denominator_survives_current_output_collection(monkeypatch):
    import torch

    from miles.policies.moss_tts_local import workflow

    monkeypatch.setattr(workflow.mpu, "get_data_parallel_world_size", lambda **kwargs: 1)
    current = torch.tensor([-0.9], requires_grad=True)
    old = torch.tensor([-1.0])
    batch = SimpleNamespace(
        batch_size=1,
        advantages=torch.ones(1),
        old_joint_logprobs=old,
        decision_mask=torch.ones(1, dtype=torch.bool),
        decision_offsets=torch.tensor([0, 1]),
        server_joint_logprobs=lambda: old,
    )
    output = SimpleNamespace(
        joint_logprobs=current,
        decision_logprobs=current,
        code_logprobs=torch.empty(0, 12),
        decision_entropy=None,
        code_entropy=None,
    )
    collector = {}
    loss, _, metrics = workflow._moss_loss_closure(
        SimpleNamespace(moss_local_old_policy_source="trainer_behavior", eps_clip=.2, eps_clip_high=.2),
        batch, 1, 1, output, trainer_output_collector=collector, parity_values=[],
    )
    loss.backward()
    assert metrics["values"][-1].item() == pytest.approx(torch.exp(torch.tensor(.1)).item())
    assert current.grad.item() == pytest.approx(-torch.exp(torch.tensor(.1)).item())
    assert batch.old_joint_logprobs.item() == -1.0
    assert collector["trainer_joint_logprobs"][0].item() == pytest.approx(-.9)
