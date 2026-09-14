"""Concurrent WER/SIM/RM scoring with one owner for sample diagnostics.

Scorers return results without changing samples. Composition bounds each active
score to [0, 1] and publishes metadata only after the whole batch succeeds.
The historical scalar and group-RM entrypoints remain available.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from miles.policies.moss_tts_local import rm_reward, sim_wer_reward, wer_reward
from miles.policies.moss_tts_local.reward_components import RewardComponent as RewardComponent
from miles.policies.moss_tts_local.reward_components import RewardScore, bound_reward
from miles.policies.moss_tts_local.reward_components import parse_components as parse_components
from miles.utils.types import Sample


def active_components(args: Any) -> tuple[RewardComponent, ...]:
    return parse_components(getattr(args, "moss_local_reward_components", None))


def _similarity_result(score) -> RewardScore:
    value = bound_reward(float(score.reward), "SIM")
    return RewardScore(
        value=value,
        metadata={
            "sim_model": sim_wer_reward.EXPECTED_SIM_MODEL,
            "sim_raw_cosine": score.raw_cosine,
            "sim_reward": value,
            "sim_reference_cache_hit": score.reference_cache_hit,
            "sim_reference_bucket_seconds": score.reference_bucket_seconds,
            "sim_candidate_bucket_seconds": score.candidate_bucket_seconds,
            "sim_service_elapsed_ms": score.service_elapsed_ms,
            "sim_candidate_sha256": score.candidate_sha256,
            "sim_item_error_code": score.item_error_code,
            "sim_dimensions": {
                "timbre": {"reward": value, "implemented": True},
                "accent": {"reward": 0.0, "implemented": False},
                "prosody": {"reward": 0.0, "implemented": False},
                "emotion": {"reward": 0.0, "implemented": False},
            },
        },
    )


async def _score_component(name: str, args, samples: list[Sample]) -> list[RewardScore]:
    if name == "wer":
        return await asyncio.gather(*(wer_reward.score_sample(args, sample) for sample in samples))
    if name == "rm":
        return await asyncio.gather(*(rm_reward.score_sample(args, sample) for sample in samples))
    if name == "sim":
        return [_similarity_result(score) for score in await sim_wer_reward.score_similarity_batch(samples)]
    raise ValueError(f"Unknown reward component {name!r}")


class RewardPipeline:
    """A resolved composition shared across all samples in a scoring batch."""

    def __init__(self, components: tuple[RewardComponent, ...]):
        self.components = tuple(item for item in components if item.weight > 0)
        self.formula = " + ".join(f"{item.weight:g}*{item.name}" for item in self.components)

    def _compose(self, results: dict[str, RewardScore]) -> RewardScore:
        metadata = {}
        diagnostics = {}
        total = 0.0
        for component in self.components:
            score = results[component.name]
            value = bound_reward(float(score.value), component.name.upper())
            metadata.update(score.metadata)
            if component.name == "wer":
                raw = score.metadata.get("wer_raw", score.metadata.get("wer"))
            elif component.name == "sim":
                raw = score.metadata["sim_raw_cosine"]
            else:
                raw = value
                metadata["rm_reward"] = value
            diagnostics[component.name] = {"weight": component.weight, "reward": value, "raw": raw}
            total += component.weight * value
        if not math.isfinite(total) or not 0.0 <= total <= 1.0:
            raise FloatingPointError("Composite MOSS-TTS reward must be finite and within [0, 1].")
        metadata.update(
            reward_components=diagnostics,
            reward=total,
            reward_formula=self.formula,
            reward_components_version="moss_tts_local_composite_v1",
        )
        return RewardScore(total, metadata)

    async def score(self, args, samples: list[Sample]) -> list[RewardScore]:
        if not samples:
            return []
        names = tuple(sorted(item.name for item in self.components))
        values = await asyncio.gather(*(_score_component(name, args, samples) for name in names))
        if any(len(scores) != len(samples) for scores in values):
            raise ValueError("Reward components must return exactly one result per sample")
        results = dict(zip(names, values, strict=True))
        return [
            self._compose({name: scores[index] for name, scores in results.items()}) for index in range(len(samples))
        ]


async def reward_batch(args: Any, samples: list[Sample], **kwargs: Any) -> list[float]:
    del kwargs
    if not samples:
        return []
    results = await RewardPipeline(active_components(args)).score(args, samples)
    for sample, score in zip(samples, results, strict=True):
        sample.metadata = dict(sample.metadata or {}) | score.metadata
    return [score.value for score in results]


async def reward_func(args: Any, sample: Sample | list[Sample], **kwargs: Any) -> float | list[float]:
    """Handle both Miles' per-sample and group-RM reward dispatch."""
    if isinstance(sample, list):
        return await reward_batch(args, sample, **kwargs)
    return (await reward_batch(args, [sample], **kwargs))[0]
