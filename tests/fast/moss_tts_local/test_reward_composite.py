import asyncio
from types import SimpleNamespace

import pytest

from miles.policies.moss_tts_local import reward_composite
from miles.policies.moss_tts_local.reward_components import RewardScore
from miles.policies.moss_tts_local.types import MediaArtifact
from miles.utils.types import Sample


def test_parse_components_supports_sim_and_rm_aliases():
    assert reward_composite.parse_components("wer=0.4 sim=0.4 rm=0.2") == (
        reward_composite.RewardComponent("wer", 0.4),
        reward_composite.RewardComponent("sim", 0.4),
        reward_composite.RewardComponent("rm", 0.2),
    )
    assert reward_composite.parse_components("reference_similarity=1") == (
        reward_composite.RewardComponent("sim", 1.0),
    )


@pytest.mark.parametrize("value", ["", "wer=0.5 sim=0.4", "wer=1 wer=0", "wer=-1", "unknown=1"])
def test_parse_components_rejects_invalid_configuration(value):
    with pytest.raises(ValueError):
        reward_composite.parse_components(value)


@pytest.mark.asyncio
async def test_default_composite_preserves_wer_reward(monkeypatch):
    sample = Sample(
        label="hello", artifacts=[MediaArtifact.from_inline_audio("d2F2", mime_type="audio/wav", sample_rate=48_000)]
    )
    calls = []

    async def fake_wer(_args, item):
        calls.append(item)
        return RewardScore(0.75, {"wer": 0.25, "wer_raw": 0.25})

    monkeypatch.setattr(reward_composite.wer_reward, "score_sample", fake_wer)
    values = await reward_composite.reward_batch(
        SimpleNamespace(moss_local_reward_components="wer=1.0"),
        [sample],
    )
    assert values == [0.75]
    assert calls == [sample]
    assert sample.metadata["reward_components"]["wer"]["weight"] == 1.0


@pytest.mark.asyncio
async def test_composite_records_all_active_component_values(monkeypatch):
    sample = Sample(label="hello", artifacts=[])
    sample.metadata = {}

    async def fake_wer(_args, _sample):
        return RewardScore(0.8, {})

    async def fake_rm(_args, _sample):
        return RewardScore(0.6, {})

    monkeypatch.setattr(reward_composite.wer_reward, "score_sample", fake_wer)
    monkeypatch.setattr(reward_composite.rm_reward, "score_sample", fake_rm)
    result = await reward_composite.reward_batch(
        SimpleNamespace(moss_local_reward_components="wer=0.5 rm=0.5"),
        [sample],
    )
    assert result == [pytest.approx(0.7)]
    assert set(sample.metadata["reward_components"]) == {"wer", "rm"}


@pytest.mark.asyncio
async def test_composite_entrypoint_accepts_group_rm(monkeypatch):
    samples = [Sample(label="one", artifacts=[]), Sample(label="two", artifacts=[])]

    async def fake_wer(_args, sample):
        return RewardScore(0.9, {"wer": 0.1})

    monkeypatch.setattr(reward_composite.wer_reward, "score_sample", fake_wer)
    values = await reward_composite.reward_func(
        SimpleNamespace(moss_local_reward_components="wer=1.0"),
        samples,
    )

    assert values == [0.9, 0.9]
    assert all(sample.metadata["reward"] == 0.9 for sample in samples)


@pytest.mark.asyncio
async def test_sim_composite_can_be_aggregated_by_rollout_metrics(monkeypatch):
    from miles.policies.moss_tts_local.rollout import _wer_rollout_metrics

    async def similarity(samples):
        return [
            SimpleNamespace(
                reward=0.8,
                raw_cosine=0.6,
                reference_cache_hit=True,
                reference_bucket_seconds=1,
                candidate_bucket_seconds=1,
                service_elapsed_ms=2,
                candidate_sha256="a" * 64,
                item_error_code=None,
            )
            for _ in samples
        ]

    monkeypatch.setattr(reward_composite.sim_wer_reward, "score_similarity_batch", similarity)
    sample = Sample(metadata={"source": "kept"})
    rewards = await reward_composite.reward_batch(SimpleNamespace(moss_local_reward_components="sim=1"), [sample])
    assert rewards == [0.8]
    assert sample.metadata["source"] == "kept"
    metrics = _wer_rollout_metrics([[sample]])
    assert metrics["moss_tts_local/sim_mean"] == 0.8
    assert metrics["moss_tts_local/mixed_reward_mean"] == 0.8


@pytest.mark.asyncio
async def test_scorers_start_concurrently_and_publish_diagnostics_once(monkeypatch):
    started = set()
    ready = asyncio.Event()
    sample = Sample(metadata={"original": True})

    def scorer(name):
        async def score(args, item):
            started.add(name)
            if len(started) == 2:
                ready.set()
            await ready.wait()
            assert item.metadata == {"original": True}
            return RewardScore(0.5, {name + "_audit": True})

        return score

    monkeypatch.setattr(reward_composite.wer_reward, "score_sample", scorer("wer"))
    monkeypatch.setattr(reward_composite.rm_reward, "score_sample", scorer("rm"))
    values = await asyncio.wait_for(
        reward_composite.reward_batch(SimpleNamespace(moss_local_reward_components="wer=.5 rm=.5"), [sample]), 2
    )
    assert values == [0.5]
    assert sample.metadata["wer_audit"] and sample.metadata["rm_audit"]


@pytest.mark.asyncio
async def test_invalid_component_leaves_the_entire_batch_unmodified(monkeypatch):
    samples = [Sample(index=0, metadata={"original": 0}), Sample(index=1, metadata={"original": 1})]

    async def score(args, sample):
        return RewardScore(0.5 if sample.index == 0 else float("nan"), {"scored": True})

    monkeypatch.setattr(reward_composite.wer_reward, "score_sample", score)
    with pytest.raises(FloatingPointError):
        await reward_composite.reward_batch(SimpleNamespace(moss_local_reward_components="wer=1"), samples)
    assert [sample.metadata for sample in samples] == [{"original": 0}, {"original": 1}]


@pytest.mark.asyncio
async def test_composite_preserves_its_explicit_bounded_wer_semantics(monkeypatch):
    async def score(args, sample):
        return RewardScore(-0.5, {"wer": 1.5})

    monkeypatch.setattr(reward_composite.wer_reward, "score_sample", score)
    sample = Sample()
    assert await reward_composite.reward_func(SimpleNamespace(moss_local_reward_components="wer=1"), sample) == 0
    assert sample.metadata["wer"] == 1.5
