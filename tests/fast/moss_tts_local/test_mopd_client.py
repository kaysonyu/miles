import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
import torch

from miles.policies.moss_tts_local.mopd_client import LocalTeacherClient, payload_hash
from miles.policies.moss_tts_local.spec import MOSS_TTS_LOCAL_SPEC


def args(tmp_path):
    return SimpleNamespace(
        moss_local_mopd_teachers=["en=http://en/score_actions", "zh=http://zh/score_actions"],
        moss_local_student_score_endpoint="http://student/score_actions",
        moss_local_mopd_default_domain=None,
        moss_local_mopd_score_batch_size=4,
        moss_local_mopd_batch_wait_ms=1,
        moss_local_mopd_retries=0,
        moss_local_mopd_timeout=5,
        moss_local_mopd_concurrency=2,
        save=str(tmp_path / "checkpoint"),
        load=None,
    )


def sample(index, domain):
    trace = SimpleNamespace(
        prompt_rows=torch.tensor([[151656] + [1024] * 12]),
        decisions=torch.tensor([0, 1]),
        codes=torch.zeros((1, 12), dtype=torch.long),
        weight_version="7",
    )
    return SimpleNamespace(index=index, metadata={"domain": domain}, structured_trajectory=trace, reward=None)


def responder(calls, *, wrong_hash=False):
    identity = MOSS_TTS_LOCAL_SPEC.identity() | {"config_sha256": MOSS_TTS_LOCAL_SPEC.identity_sha256()}

    async def respond(request):
        host = request.url.host
        body = json.loads(request.content)
        if request.url.path == "/model_info":
            return httpx.Response(
                200,
                json={
                    "stages": [
                        {
                            "stage": "tts_engine",
                            "data": {
                                "supports_action_scoring": True,
                                "supports_weight_update": False,
                                "weight_version": "base",
                                "teacher_weight_sha256": "a" * 64,
                                "model_identity": identity,
                            },
                        }
                    ]
                },
            )
        calls.append((host, [x["sample_id"] for x in body["samples"]]))
        batch_hash = hashlib.sha256("".join(payload_hash(x) for x in body["samples"]).encode()).hexdigest()
        values = [
            dict(
                version=1,
                sample_id=x["sample_id"],
                input_sha256="bad" if wrong_hash else payload_hash(x),
                decision_logprobs=[-0.1, -0.2],
                code_logprobs=[[-0.3] * 12],
                temperature=1.0,
                logprob_semantics="temperature_scaled_full_vocab_v1",
                model_identity=identity,
                weight_version="7" if host == "student" else "base",
                teacher_weight_sha256="a" * 64,
                scoring_batch_sha256=batch_hash,
                score_chunk_size=128,
            )
            for x in body["samples"]
        ]
        return httpx.Response(200, json={"results": values})

    return respond


@pytest.mark.asyncio
async def test_routes_batch_scores_and_persists_frozen_identity(tmp_path):
    c = LocalTeacherClient(args(tmp_path))
    await c.client.aclose()
    calls = []
    c.client = httpx.AsyncClient(transport=httpx.MockTransport(responder(calls)))
    samples = [sample(i, "en" if i % 2 == 0 else "zh") for i in range(8)]
    await asyncio.gather(*(c.score(x, temperature=1.0) for x in samples))
    await c.close()
    assert all(x.reward == 0 and x.metadata["mopd_teacher"] == ("en" if x.index % 2 == 0 else "zh") for x in samples)
    assert sorted(i for host, ids in calls if host == "en" for i in ids) == ["0", "2", "4", "6"]
    assert sorted(i for host, ids in calls if host == "zh" for i in ids) == ["1", "3", "5", "7"]
    manifest = json.loads((tmp_path / "checkpoint/mopd_teachers.json").read_text())
    assert set(manifest) == {"en", "zh"}


@pytest.mark.asyncio
async def test_wrong_score_cannot_become_zero_filler_reward(tmp_path):
    c = LocalTeacherClient(args(tmp_path))
    await c.client.aclose()
    c.client = httpx.AsyncClient(transport=httpx.MockTransport(responder([], wrong_hash=True)))
    x = sample(0, "en")
    with pytest.raises(ValueError, match="exact student trajectory"):
        await c.score(x, temperature=1.0)
    assert x.reward is None and "mopd_scores" not in x.metadata
    await c.close()


@pytest.mark.asyncio
async def test_resume_rejects_changed_teacher(tmp_path):
    cfg = args(tmp_path)
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "mopd_teachers.json").write_text(json.dumps({"en": {"weight_sha256": "b" * 64}}))
    cfg.load = str(prior)
    c = LocalTeacherClient(cfg)
    await c.client.aclose()
    c.client = httpx.AsyncClient(transport=httpx.MockTransport(responder([])))
    with pytest.raises(ValueError, match="resume changed teacher"):
        await c.score(sample(0, "en"), temperature=1.0)
    await c.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["normal_stale", "replay_ok", "replay_wrong_teacher", "replay_wrong_version", "disabled"]
)
async def test_replay_keeps_behavior_version_and_checks_real_teacher(tmp_path, case):
    cfg = args(tmp_path)
    cfg.moss_local_replay_manifest = None if case == "disabled" else "pool.json"
    c = LocalTeacherClient(cfg)
    await c.client.aclose()
    c.client = httpx.AsyncClient(transport=httpx.MockTransport(responder([])))
    x = sample(0, "en")
    x.structured_trajectory.weight_version = "teacher-6000"
    if case != "normal_stale":
        x.metadata["moss_teacher_replay"] = dict(
            teacher_weight_sha256=("b" if case == "replay_wrong_teacher" else "a") * 64,
            behavior_weight_version="wrong" if case == "replay_wrong_version" else "teacher-6000",
        )
    if case == "replay_ok":
        await c.score(x, temperature=1.0)
        assert x.metadata["moss_training_policy_version"] == "7"
        assert x.structured_trajectory.weight_version == "teacher-6000"
    else:
        with pytest.raises(ValueError):
            await c.score(x, temperature=1.0)
    await c.close()
