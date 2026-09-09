"""Configurable WER/SIM/RM reward composition for MOSS-TTS Local.

The default Local recipe remains the single WER component.  Composite scoring
is opt-in through ``--moss-local-reward-components`` and uses the same
per-sample reward function contract as the existing WER scorer.

Component names are ``wer``, ``sim`` (or ``reference_similarity``), and
``rm`` (or ``judge``).  SIM exposes four reference uses (timbre, accent,
prosody, emotion); only timbre has a service implementation today, while the
other declared uses are retained as explicit zero-score placeholders.
"""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass
from typing import Any

from miles.policies.moss_tts_local import sim_wer_reward, wer_reward
from miles.policies.moss_tts_local import rm_reward
from miles.utils.types import Sample

_ALIASES = {
    "reference_similarity": "sim",
    "timbre_sim": "sim",
    "judge": "rm",
}
_KNOWN_COMPONENTS = frozenset({"wer", "sim", "rm"})


@dataclass(frozen=True)
class RewardComponent:
    name: str
    weight: float


def parse_components(raw: str | None) -> tuple[RewardComponent, ...]:
    """Parse ``name=weight`` entries and require a normalized convex sum."""

    value = (
        os.getenv("MOSS_TTS_REWARD_COMPONENTS", "wer=1.0") if raw is None else raw
    ).strip()
    if not value:
        raise ValueError("MOSS-TTS reward components must not be empty.")
    parsed: list[RewardComponent] = []
    for entry in value.replace(",", " ").split():
        name, separator, raw_weight = entry.partition("=")
        if not separator:
            raise ValueError(f"Reward component {entry!r} must use name=weight syntax.")
        canonical = _ALIASES.get(name.strip().casefold(), name.strip().casefold())
        if canonical not in _KNOWN_COMPONENTS:
            raise ValueError(f"Unknown MOSS-TTS reward component {name!r}; expected wer, sim, or rm.")
        try:
            weight = float(raw_weight)
        except ValueError as error:
            raise ValueError(f"Reward component {entry!r} has a non-numeric weight.") from error
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Reward component {entry!r} must have a finite non-negative weight.")
        parsed.append(RewardComponent(canonical, weight))
    if not parsed:
        raise ValueError("MOSS-TTS reward components must contain at least one entry.")
    names = [item.name for item in parsed]
    if len(names) != len(set(names)):
        raise ValueError("MOSS-TTS reward components must not repeat a component.")
    total = sum(item.weight for item in parsed)
    if total <= 0 or not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"MOSS-TTS reward component weights must sum to 1, got {total}.")
    return tuple(parsed)


def active_components(args: Any) -> tuple[RewardComponent, ...]:
    return parse_components(getattr(args, "moss_local_reward_components", None))


def _bounded(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise FloatingPointError(f"{name} reward is NaN or Inf.")
    return max(0.0, min(1.0, value))


async def reward_batch(args: Any, samples: list[Sample], **kwargs: Any) -> list[float]:
    """Score all active components concurrently and return one scalar per sample."""

    del kwargs
    if not samples:
        return []
    components = active_components(args)
    tasks: dict[str, Any] = {}
    names = {item.name for item in components if item.weight > 0}
    if "wer" in names:
        tasks["wer"] = asyncio.gather(*(wer_reward.reward_func(args, sample) for sample in samples))
    if "sim" in names:
        tasks["sim"] = sim_wer_reward.score_similarity_batch(samples)
    if "rm" in names:
        tasks["rm"] = asyncio.gather(*(rm_reward.reward_func(args, sample) for sample in samples))
    results = {name: await task for name, task in tasks.items()}

    rewards: list[float] = []
    for index, sample in enumerate(samples):
        metadata = dict(sample.metadata or {})
        diagnostics: dict[str, dict[str, Any]] = {}
        total = 0.0
        for component in components:
            if component.weight == 0:
                continue
            if component.name == "wer":
                value = _bounded(float(results["wer"][index]), "WER")
                raw = metadata.get("wer_raw", metadata.get("wer"))
            elif component.name == "sim":
                score = results["sim"][index]
                value = _bounded(float(score.reward), "SIM")
                raw = score.raw_cosine
                metadata.update(
                    {
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
                    }
                )
            else:
                value = _bounded(float(results["rm"][index]), "RM")
                raw = value
                metadata["rm_reward"] = value
            diagnostics[component.name] = {
                "weight": component.weight,
                "reward": value,
                "raw": raw,
            }
            total += component.weight * value
        if not math.isfinite(total) or not 0.0 <= total <= 1.0:
            raise FloatingPointError("Composite MOSS-TTS reward must be finite and within [0, 1].")
        metadata["reward_components"] = diagnostics
        metadata["reward"] = total
        metadata["reward_formula"] = " + ".join(
            f"{item.weight:g}*{item.name}" for item in components if item.weight > 0
        )
        metadata["reward_components_version"] = "moss_tts_local_composite_v1"
        sample.metadata = metadata
        rewards.append(total)
    return rewards


async def reward_func(
    args: Any,
    sample: Sample | list[Sample],
    **kwargs: Any,
) -> float | list[float]:
    """Handle both Miles' per-sample and group-RM reward dispatch."""

    if isinstance(sample, list):
        return await reward_batch(args, sample, **kwargs)
    return (await reward_batch(args, [sample], **kwargs))[0]
