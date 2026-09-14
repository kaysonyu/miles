"""Configuration and scalar validation for MOSS-TTS Local rewards."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class RewardScore:
    """A scorer result; the caller owns merging diagnostics into a sample."""

    value: float
    metadata: dict[str, Any]


def is_plain_wer(components: tuple[RewardComponent, ...]) -> bool:
    """Select the historical unbounded WER recipe by values, not spelling."""
    return tuple(item for item in components if item.weight > 0) == (RewardComponent("wer", 1.0),)


def parse_components(raw: str | None) -> tuple[RewardComponent, ...]:
    """Parse ``name=weight`` entries and require a normalized convex sum."""

    value = (os.getenv("MOSS_TTS_REWARD_COMPONENTS", "wer=1.0") if raw is None else raw).strip()
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


def bound_reward(value: float, name: str) -> float:
    """Validate and clamp one component reward to the GRPO scalar range."""

    if not math.isfinite(value):
        raise FloatingPointError(f"{name} reward is NaN or Inf.")
    return max(0.0, min(1.0, value))
