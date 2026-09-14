import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from miles.policies.moss_tts_local.rollout_data import MossTTSLocalRolloutDataAdapter
from miles.policies.moss_tts_local.teacher_replay import FrozenTeacherReplay
from miles.policies.moss_tts_local.types import MossTTSLocalTrajectoryV2
from miles.utils.types import Sample
from test_trajectory import _trajectory


def replay_config(tmp_path):
    trace = _trajectory()
    trace.weight_version = "teacher-6000"
    trace.sampling.pop("temperature")
    trace.sampling.update(text_temperature=0.7, audio_temperature=0.7)
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace.to_dict(), default=lambda x: x.tolist()))
    row = dict(
        metadata={"domain": "dialect", "bucket": "shanghai", "text_hash": "teacher-text"},
        text="teacher prompt",
        teacher_weight_sha256="a" * 64,
        trajectory_file=str(path),
        trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        audio="/audio.wav",
        sample_rate=24000,
        audio_sha256="b" * 64,
        audio_bytes=42,
        replay_id="d-shanghai-1",
    )
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps(row) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                schema_version=1,
                temperature=0.7,
                pool_file=str(pool),
                pool_sha256=hashlib.sha256(pool.read_bytes()).hexdigest(),
                teachers={"dialect": {"weight_sha256": "a" * 64}},
            )
        )
    )
    return SimpleNamespace(
        moss_local_replay_manifest=str(manifest),
        rollout_temperature=0.7,
        moss_local_replay_every_n_groups=5,
        rollout_batch_size=16,
        moss_local_mopd_estimator="dense_reverse",
        load=None,
        save=str(tmp_path / "save"),
    )


def test_replay_json_roundtrip_preserves_teacher_behavior_and_slot_domain(tmp_path):
    cfg = replay_config(tmp_path)
    replay = FrozenTeacherReplay(cfg)
    assert sum(replay.select_group(i, j) for i in range(5) for j in range(16)) == 16
    sample = Sample(
        index=3, prompt="original", metadata={"domain": "dialect", "bucket": "shanghai", "text_hash": "slot"}
    )
    replay.materialize(sample, rollout_id=128, seed=42)
    assert sample.prompt == "teacher prompt" and sample.status == Sample.Status.COMPLETED
    assert sample.structured_trajectory.weight_version == "teacher-6000"
    assert sample.metadata["moss_teacher_replay"]["original_slot_text_hash"] == "slot"
    torch.testing.assert_close(sample.structured_trajectory.code_logprobs, _trajectory().code_logprobs)
    assert sample.artifacts[0].uri == "/audio.wav"
    with pytest.raises(ValueError, match="pending"):
        replay.materialize(sample, rollout_id=128, seed=42)
    missing = Sample(index=2, metadata={"domain": "instruction", "bucket": "normal"})
    with pytest.raises(ValueError, match="lacks"):
        replay.materialize(missing, rollout_id=128, seed=42)


@pytest.mark.parametrize("mutation", ["temperature", "pool", "trace", "resume"])
def test_replay_rejects_changed_inputs(tmp_path, mutation):
    cfg = replay_config(tmp_path)
    if mutation == "temperature":
        cfg.rollout_temperature = 1
    elif mutation == "pool":
        with (tmp_path / "pool.jsonl").open("a") as f:
            f.write(" ")
    elif mutation == "trace":
        with (tmp_path / "trace.json").open("a") as f:
            f.write(" ")
    else:
        cfg.load = str(tmp_path)
        (tmp_path / "teacher_replay_pool.json").write_text("{}")
    with pytest.raises(ValueError):
        FrozenTeacherReplay(cfg)


def test_mixed_data_preserves_true_behavior_and_requires_current_student_version(tmp_path):
    cfg = replay_config(tmp_path)
    sample = Sample(index=0, metadata={"domain": "dialect", "bucket": "shanghai"})
    FrozenTeacherReplay(cfg).materialize(sample, rollout_id=128, seed=42)
    sample.metadata["moss_training_policy_version"] = "128"
    student = Sample(
        index=1,
        status=Sample.Status.COMPLETED,
        metadata={"moss_training_policy_version": "128"},
        structured_trajectory=_trajectory(),
    )
    student.structured_trajectory.weight_version = "128"
    adapter = MossTTSLocalRolloutDataAdapter()
    postprocess = lambda samples: ([0.0] * len(samples), [0.0] * len(samples))
    data = adapter.samples_to_train_data(cfg, [sample, student], reward_postprocess=postprocess)
    assert data["weight_versions"] == ["128", "128"]
    assert data["behavior_weight_versions"] == ["teacher-6000", "128"]
    assert data["trajectory_origins"] == ["teacher_replay", "student"]
    student.metadata["moss_training_policy_version"] = "127"
    with pytest.raises(ValueError, match="versions"):
        adapter.samples_to_train_data(cfg, [sample, student], reward_postprocess=postprocess)
    cfg.moss_local_mopd_estimator = "sampled"
    with pytest.raises(ValueError, match="native mixed"):
        adapter.samples_to_train_data(cfg, [sample], reward_postprocess=postprocess)


@pytest.mark.parametrize(
    "case", ["valid", "unpublished_student", "stale_metadata", "disabled", "teacher_changed", "behavior_changed"]
)
def test_shared_version_guard_preserves_frozen_teacher_and_requires_published_student(tmp_path, case):
    from miles.utils.weight_version import assert_samples_weight_version_sane

    cfg = replay_config(tmp_path)
    cfg.debug_rollout_only = cfg.debug_skip_weight_update = cfg.moss_local_async = False
    cfg.policy_family = "moss_tts_local"
    cfg.moss_local_objective = "mopd"
    cfg.moss_local_old_policy_source = "trainer_preupdate"
    sample = Sample(index=0, metadata={"domain": "dialect", "bucket": "shanghai"})
    FrozenTeacherReplay(cfg).materialize(sample, rollout_id=128, seed=42)
    sample.metadata["moss_training_policy_version"] = "128"
    sample.metadata["mopd_scores"] = dict(teacher_weight_sha256="a" * 64, student_score={"weight_version": "128"})
    if case == "unpublished_student":
        sample.metadata["moss_training_policy_version"] = "default"
        sample.metadata["mopd_scores"]["student_score"]["weight_version"] = "default"
    elif case == "stale_metadata":
        sample.metadata["moss_training_policy_version"] = "127"
    elif case == "disabled":
        cfg.moss_local_replay_manifest = None
    elif case == "teacher_changed":
        sample.metadata["mopd_scores"]["teacher_weight_sha256"] = "b" * 64
    elif case == "behavior_changed":
        sample.structured_trajectory.weight_version = "other-teacher"
    if case == "valid":
        assert_samples_weight_version_sane(cfg, [sample])
        assert sample.structured_trajectory.weight_version == "teacher-6000"
    else:
        with pytest.raises(AssertionError):
            assert_samples_weight_version_sane(cfg, [sample])


def exact_replay_config(tmp_path, *, truncated=False):
    cfg = replay_config(tmp_path)
    pool = tmp_path / "pool.jsonl"
    entry = json.loads(pool.read_text())
    entry.update(target_rollout_id=128, target_sample_index=8192)
    entry["metadata"]["tts_params"] = {"instructions": "shanghai"}
    if truncated:
        trace = json.loads((tmp_path / "trace.json").read_text())
        for key in ["decisions", "decision_logprobs", "decision_mask"]:
            trace[key] = trace[key][:-1]
        trace["finish_reason"] = "length"
        (tmp_path / "trace.json").write_text(json.dumps(trace))
        entry["trajectory_sha256"] = hashlib.sha256((tmp_path / "trace.json").read_bytes()).hexdigest()
    pool.write_text(json.dumps(entry) + "\n")
    manifest = tmp_path / "manifest.json"
    meta = json.loads(manifest.read_text())
    meta.update(
        selection="exact_sample", allow_truncated=True, pool_sha256=hashlib.sha256(pool.read_bytes()).hexdigest()
    )
    manifest.write_text(json.dumps(meta))
    return cfg, entry


@pytest.mark.parametrize("truncated", [False, True])
def test_exact_replay_keeps_original_prompt_and_real_finish_status(tmp_path, truncated):
    cfg, entry = exact_replay_config(tmp_path, truncated=truncated)
    pool = FrozenTeacherReplay(cfg)
    sample = Sample(index=8192, prompt=entry["text"], metadata=dict(entry["metadata"]))
    pool.materialize(sample, rollout_id=128, seed=1)
    assert sample.prompt == entry["text"] and sample.structured_trajectory.weight_version == "teacher-6000"
    assert sample.status == (Sample.Status.TRUNCATED if truncated else Sample.Status.COMPLETED)
    assert sample.structured_trajectory.finish_reason == ("length" if truncated else "stop")
    wrong = Sample(index=8192, prompt="wrong prompt", metadata=dict(entry["metadata"]))
    with pytest.raises(ValueError, match="original prompt"):
        pool.materialize(wrong, rollout_id=128, seed=1)
    missing = Sample(index=8193, prompt=entry["text"], metadata=dict(entry["metadata"]))
    with pytest.raises(ValueError, match="lacks"):
        pool.materialize(missing, rollout_id=128, seed=1)


def test_prompt_only_changes_prompt_without_importing_teacher_behavior(tmp_path):
    cfg = replay_config(tmp_path)
    cfg.moss_local_replay_mode = "prompt_only"
    pool = FrozenTeacherReplay(cfg)
    sample = Sample(
        index=1, prompt="original", metadata={"domain": "dialect", "bucket": "shanghai", "text_hash": "slot"}
    )
    pool.materialize_prompt(sample, rollout_id=128, seed=42)
    assert sample.prompt == "teacher prompt" and sample.status == Sample.Status.PENDING
    assert sample.structured_trajectory is None and not sample.artifacts
    assert "moss_teacher_replay" not in sample.metadata
    assert sample.metadata["moss_prompt_selection"]["original_slot_text_hash"] == "slot"
