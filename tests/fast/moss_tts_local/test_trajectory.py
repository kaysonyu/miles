from __future__ import annotations

import base64

import pytest
import torch

from miles.policies.moss_tts_local.spec import MOSS_TTS_LOCAL_SPEC
from miles.policies.moss_tts_local.types import MediaArtifact, MossTTSLocalTrajectoryV2
from miles.utils.types import Sample

NUM_GPUS = 0


def _identity() -> dict:
    return MOSS_TTS_LOCAL_SPEC.identity()


def _trajectory(*, frames: int = 2, finish_reason: str = "stop") -> MossTTSLocalTrajectoryV2:
    decisions = torch.zeros(frames + (finish_reason == "stop"), dtype=torch.long)
    if finish_reason == "stop":
        decisions[-1] = 1
    return MossTTSLocalTrajectoryV2(
        version=2,
        model_family="moss_tts_local",
        prompt_rows=torch.tensor(
            [[151656, *([1024] * 12)], [151644, *([1024] * 12)]],
            dtype=torch.long,
        ),
        decisions=decisions,
        decision_logprobs=torch.arange(len(decisions), dtype=torch.float32).neg(),
        decision_mask=torch.ones(len(decisions), dtype=torch.bool),
        codes=torch.arange(frames * 12, dtype=torch.long).reshape(frames, 12),
        code_logprobs=-torch.ones((frames, 12), dtype=torch.float32),
        code_mask=torch.ones((frames, 12), dtype=torch.bool),
        finish_reason=finish_reason,
        sampling={"logprob_semantics": "temperature_scaled_full_vocab_v1", "temperature": 1.0},
        model_identity=_identity(),
        weight_version="7",
        request_id="req-1",
    )


@pytest.mark.parametrize(("frames", "finish_reason"), [(0, "stop"), (1, "stop"), (3, "length")])
def test_trajectory_geometry_and_round_trip(frames, finish_reason):
    trajectory = _trajectory(frames=frames, finish_reason=finish_reason)
    trajectory.validate()

    restored = MossTTSLocalTrajectoryV2.from_dict(trajectory.to_dict())

    assert restored.finish_reason == finish_reason
    assert torch.equal(restored.prompt_rows, trajectory.prompt_rows)
    assert torch.equal(restored.decisions, trajectory.decisions)
    assert torch.equal(restored.codes, trajectory.codes)
    assert torch.equal(restored.server_joint_logprobs(), trajectory.server_joint_logprobs())


def test_joint_logprob_adds_codes_only_to_continue_events():
    trajectory = _trajectory(frames=2, finish_reason="stop")

    joint = trajectory.server_joint_logprobs()

    assert joint.tolist() == pytest.approx([-12.0, -13.0, -2.0])


def test_stop_rejects_missing_terminal_decision():
    trajectory = _trajectory(frames=2, finish_reason="stop")
    trajectory.decisions = trajectory.decisions[:-1]
    trajectory.decision_logprobs = trajectory.decision_logprobs[:-1]
    trajectory.decision_mask = trajectory.decision_mask[:-1]

    with pytest.raises(ValueError, match=r"R=T\+1"):
        trajectory.validate()


def test_length_rejects_stop_decision():
    trajectory = _trajectory(frames=2, finish_reason="length")
    trajectory.decisions[-1] = 1

    with pytest.raises(ValueError, match="cannot contain a stop"):
        trajectory.validate()


def test_trajectory_rejects_nonfinite_action_logprob():
    trajectory = _trajectory()
    trajectory.code_logprobs[0, 3] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        trajectory.validate()


def test_trajectory_rejects_wrong_model_identity():
    trajectory = _trajectory()
    trajectory.model_identity["n_vq"] = 8

    with pytest.raises(ValueError, match="identity mismatch"):
        trajectory.validate()


def test_trajectory_rejects_wrong_local_ffn_identity():
    trajectory = _trajectory()
    trajectory.model_identity["local_ffn_hidden_size"] = 10240

    with pytest.raises(ValueError, match="local_ffn_hidden_size"):
        trajectory.validate()


def test_sample_explicitly_round_trips_structured_objects():
    payload = base64.b64encode(b"fake-wave").decode()
    artifact = MediaArtifact.from_inline_audio(payload, mime_type="audio/wav", sample_rate=48000)
    sample = Sample(
        index=3,
        structured_trajectory=_trajectory(frames=1),
        artifacts=[artifact],
        status=Sample.Status.COMPLETED,
    )

    restored = Sample.from_dict(sample.to_dict())

    assert isinstance(restored.structured_trajectory, MossTTSLocalTrajectoryV2)
    assert isinstance(restored.artifacts[0], MediaArtifact)
    assert restored.artifacts[0].sha256 == artifact.sha256
    assert torch.equal(restored.structured_trajectory.codes, sample.structured_trajectory.codes)


def test_media_artifact_detects_checksum_mismatch():
    payload = base64.b64encode(b"fake-wave").decode()
    artifact = MediaArtifact.from_inline_audio(payload, mime_type="audio/wav", sample_rate=48000)
    artifact.sha256 = "0" * 64

    with pytest.raises(ValueError, match="checksum mismatch"):
        artifact.validate()
