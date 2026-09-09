from types import SimpleNamespace

import pytest

from miles.policies.moss_tts_local import reward_composite
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
    sample = Sample(label="hello", artifacts=[MediaArtifact.from_inline_audio(
        "d2F2", mime_type="audio/wav", sample_rate=48_000
    )])
    calls = []

    async def fake_wer(_args, item):
        calls.append(item)
        item.metadata["wer"] = 0.25
        item.metadata["wer_raw"] = 0.25
        return 0.75

    monkeypatch.setattr(reward_composite.wer_reward, "reward_func", fake_wer)
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
        return 0.8

    async def fake_rm(_args, _sample):
        return 0.6

    monkeypatch.setattr(reward_composite.wer_reward, "reward_func", fake_wer)
    monkeypatch.setattr(reward_composite.rm_reward, "reward_func", fake_rm)
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
        sample.metadata = dict(sample.metadata or {})
        sample.metadata["wer"] = 0.1
        return 0.9

    monkeypatch.setattr(reward_composite.wer_reward, "reward_func", fake_wer)
    values = await reward_composite.reward_func(
        SimpleNamespace(moss_local_reward_components="wer=1.0"),
        samples,
    )

    assert values == [0.9, 0.9]
    assert all(sample.metadata["reward"] == 0.9 for sample in samples)
