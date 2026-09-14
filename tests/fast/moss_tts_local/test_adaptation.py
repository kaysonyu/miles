"""Contracts at the policy, transport and trainer-lifecycle interfaces."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.policy_runtime import MegatronPolicyRuntime
from miles.policies.base import RolloutDataAdapter
from miles.policies.media import MediaArtifact
from miles.policies.moss_tts_local.data_schema import (
    ACTION_FIELDS,
    PAIRED_SCORE_FIELDS,
    prepare_shard,
    tensorize_shard,
)
from miles.policies.moss_tts_local.objectives import LocalObjective
from miles.policies.moss_tts_local.types import MediaArtifact as LegacyMediaArtifact
from miles.policies.registry import policy_for_args, resolve_policy


def test_text_resolution_does_not_import_local_or_gpu_runtime():
    code = """
import sys
from miles.policies.registry import resolve_policy
policy = resolve_policy()
assert policy.rollout_adapter is None
assert policy.weight_adapter is None
assert policy.create_workflow() is None
assert not any(name.startswith(('megatron', 'miles.policies.moss_tts_local', 'torch')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_local_data_adapter_conforms_without_importing_training_runtime():
    code = """
import sys
from miles.policies.base import RolloutDataAdapter
from miles.policies.registry import resolve_policy
adapter = resolve_policy('moss_tts_local').rollout_adapter
assert isinstance(adapter, RolloutDataAdapter)
assert not any(name.startswith(('megatron', 'transformer_engine')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert isinstance(resolve_policy("moss_tts_local").rollout_adapter, RolloutDataAdapter)
    assert policy_for_args(SimpleNamespace()).family == "text"
    with pytest.raises(ValueError, match="Unknown policy"):
        resolve_policy("misspelled")


def test_media_compatibility_import_keeps_the_same_class():
    assert MediaArtifact is LegacyMediaArtifact
    artifact = MediaArtifact.from_inline_audio("d2F2", mime_type="audio/wav", sample_rate=48000)
    assert MediaArtifact.from_dict(artifact.to_dict()) == artifact


def test_one_schema_prepares_actions_and_optional_paired_scores():
    data = {field.name: [[0]] for field in ACTION_FIELDS}
    data.update({field.name: [[0] * 12] if field.code_scores else [[0]] for field in PAIRED_SCORE_FIELDS})
    data.update(weight_versions=["7"], rollout_event_mask_sums=[1])
    tensorize_shard(data)
    result = prepare_shard(data, device="cpu")
    assert result is data
    for field in (*ACTION_FIELDS, *PAIRED_SCORE_FIELDS):
        assert data[field.name][0].dtype == field.dtype
        if field.code_scores:
            assert data[field.name][0].shape == (1, 12)
    assert data["rollout_event_mask_sums"].dtype == torch.float32
    del data["codes"]
    with pytest.raises(ValueError, match="codes"):
        prepare_shard(data, device="cpu")


def _runtime(*, retain=True):
    events = []
    updater = SimpleNamespace(weight_version=1, pop_metrics=lambda: {"refit": 3})
    runtime = MegatronPolicyRuntime(
        weight_updater=updater,
        weights_backuper=SimpleNamespace(backup=lambda tag: events.append(("backup", tag))),
        switch_model=lambda tag: events.append(("switch", tag)),
        profiler=SimpleNamespace(step=lambda **kwargs: events.append(("profiler", kwargs["rollout_id"]))),
        backup_required=True,
        rollout_data_postprocess=None,
        retain_behavior=retain,
    )
    return runtime, updater, events


def test_snapshot_lease_rotates_by_publication_and_restores_on_failure():
    runtime, updater, events = _runtime()
    assert runtime.on_weights_published()
    updater.weight_version = 2
    runtime.on_weights_published()
    assert runtime.published_version == "2"
    with pytest.raises(RuntimeError, match="forward failed"), runtime.behavior_snapshot("1"):
        raise RuntimeError("forward failed")
    assert events == [("switch", "old_actor"), ("switch", "actor")]
    updater.weight_version = 3
    runtime.on_weights_published()
    with pytest.raises(ValueError, match="No retained"):
        with runtime.behavior_snapshot("1"):
            pytest.fail("Expired snapshot was leased")
    updater.weight_version = 5
    with pytest.raises(ValueError, match="consecutive"):
        runtime.on_weights_published()


def test_runtime_finalization_and_no_snapshot_mode():
    runtime, updater, events = _runtime(retain=False)
    assert not runtime.on_weights_published()
    runtime.finish_rollout(4)
    assert events == [("profiler", 4), ("backup", "actor")]
    assert runtime.pop_weight_metrics() == {"refit": 3}


@pytest.mark.parametrize(
    "kind,native,dense",
    [
        ("sampled", False, False),
        ("sampled_native", True, False),
        ("dense_reverse", True, True),
        ("dense_forward", True, True),
    ],
)
def test_objective_requirements_are_explicit(kind, native, dense):
    objective = LocalObjective.from_args(SimpleNamespace(moss_local_objective="mopd", moss_local_mopd_estimator=kind))
    assert objective.is_distillation
    assert objective.uses_native_teacher == native
    assert objective.is_dense == dense
    assert not LocalObjective.from_args(SimpleNamespace()).is_distillation


def test_workflows_are_per_trainer_and_reuse_their_native_pool(monkeypatch):
    from miles.policies.moss_tts_local import workflow

    made = []

    def create(args):
        pool = object()
        made.append(pool)
        return pool

    monkeypatch.setattr(workflow, "NativeTeacherPool", create)
    policy = resolve_policy("moss_tts_local")
    first, second = policy.create_workflow(), policy.create_workflow()
    args = SimpleNamespace(moss_local_objective="mopd", moss_local_mopd_estimator="dense_reverse")
    assert first._get_native_teachers(args) is first._get_native_teachers(args)
    assert first._get_native_teachers(args) is not second._get_native_teachers(args)
    assert len(made) == 2
