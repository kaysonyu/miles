from argparse import Namespace

import pytest

from miles.backends.sglang_omni_utils.api_client import SGLangOmniApiClient
from miles.policies.moss_tts_local.checkpoint import (
    has_resume_checkpoint,
    validate_optimizer_backend_resume,
    validate_resume_implementation,
)
from miles.policies.moss_tts_local.rollout_data import split_raw
from miles.utils.seqlen_balancing import first_fit_decreasing_pack


def test_sample_cap_is_independent_of_token_budget():
    bins = first_fit_decreasing_pack([8, 8, 8, 8, 8], 1000, max_samples_per_bin=2)
    assert sorted(i for batch in bins for i in batch) == list(range(5))
    assert all(len(batch) <= 2 for batch in bins)


def test_structured_schedule_uses_global_frame_lengths_without_text_tokens():
    args = Namespace(
        global_batch_size=4,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=1024,
        max_samples_per_microbatch=2,
        balance_by_flops=False,
        balance_data=False,
    )
    topology = dict(dp_size=2, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1)
    data = dict(
        total_lengths=[10, 20, 30, 40],
        rollout_ids=[0, 1, 2, 3],
        rewards=[0.0, 1.0, 2.0, 3.0],
        raw_reward=[0.0, 1.0, 2.0, 3.0],
        sample_indices=[0, 1, 2, 3],
    )
    shards = split_raw(args, data, topology)
    assert len(shards) == 2
    assert sorted(i for shard in shards for i in shard["sample_indices"]) == [0, 1, 2, 3]
    assert all(shard["num_rollouts"] == [4] for shard in shards)
    assert all(len(batch) <= 2 for shard in shards for batch in shard["micro_batch_indices"])


def test_pretrained_load_is_not_used_for_a_resume_checkpoint(tmp_path):
    assert not has_resume_checkpoint(tmp_path)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("3")
    assert has_resume_checkpoint(tmp_path)


@pytest.mark.parametrize("layout", ["dp_reshardable", "dp_zero_gather_scatter", None])
@pytest.mark.parametrize("saved,current", [("local", "transformer_engine"), ("transformer_engine", "local")])
def test_moss_rejects_bucket_optimizer_resume_across_implementations(layout, saved, current):
    with pytest.raises(ValueError, match="dist-ckpt-optim-fully-reshardable"):
        validate_optimizer_backend_resume(saved, current, layout)


@pytest.mark.parametrize("impl", ["local", "transformer_engine"])
def test_moss_allows_same_backend_bucket_resume(impl):
    validate_optimizer_backend_resume(impl, impl, "dp_reshardable")


@pytest.mark.parametrize("layout", ["fully_reshardable", "fully_sharded_model_space"])
def test_moss_allows_parameter_based_optimizer_resume_across_implementations(layout):
    validate_optimizer_backend_resume("local", "transformer_engine", layout)


def test_resume_backend_guard_does_not_apply_to_other_policies():
    validate_resume_implementation(Namespace(policy_family="text"))


@pytest.mark.parametrize(
    "finetune,no_load_optim,distributed", [(True, False, True), (False, True, True), (False, False, False)]
)
def test_resume_backend_guard_skips_without_distributed_optimizer_restore(finetune, no_load_optim, distributed):
    validate_resume_implementation(
        Namespace(
            policy_family="moss_tts_local",
            finetune=finetune,
            no_load_optim=no_load_optim,
            use_distributed_optimizer=distributed,
        )
    )


def test_megatron_resume_calls_moss_guard_without_a_pretrained_path(monkeypatch, tmp_path):
    from miles.backends.megatron_utils import checkpoint as backend
    from miles.policies.moss_tts_local import checkpoint as policy

    (tmp_path / "marker").write_text("checkpoint")
    args = Namespace(policy_family="moss_tts_local", load=str(tmp_path), custom_pretrained_checkpoint_loader_path=None)
    monkeypatch.setattr(backend, "get_args", lambda: args)
    monkeypatch.setattr(backend, "_is_megatron_checkpoint", lambda path: True)
    monkeypatch.setattr(backend, "is_dsv4_model", lambda args: False)

    def reject(args):
        raise ValueError("resume guard reached before optimizer load")

    monkeypatch.setattr(policy, "validate_resume_implementation", reject)
    with pytest.raises(ValueError, match="resume guard reached"):
        backend.load_checkpoint([], None, None, {}, False)


@pytest.mark.asyncio
async def test_omni_bucket_keeps_stage_paused_and_stamps_version(monkeypatch):
    captured = {}

    async def admin(self, endpoint, payload=None):
        captured.update(endpoint=endpoint, payload=payload)
        return {"success": True}

    monkeypatch.setattr(SGLangOmniApiClient, "_admin", admin)
    client = SGLangOmniApiClient("http://localhost:1", "tts_engine")
    await client.update_weights_from_distributed(
        ["audio_lm_heads.0.weight"], ["bfloat16"], [(2, 2)], "weights", weight_version="7"
    )
    assert captured["payload"]["keep_pause"] is True
    assert captured["payload"]["weight_version"] == "7"
    assert captured["payload"]["shapes"] == [[2, 2]]


@pytest.mark.asyncio
async def test_omni_controller_weight_update_lock_lifecycle():
    from miles.backends.sglang_omni_utils.controller import OmniInferenceController

    controller = OmniInferenceController(Namespace(debug_train_only=True))
    await controller.init()
    info = await controller.start_update_weights()
    assert info.rollout_engines == []
    await controller.end_update_weights(info.snapshot_cell_id_to_hashes)
    await controller.dispose()
