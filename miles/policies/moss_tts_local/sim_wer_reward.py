"""Speaker-SIM plus intelligibility reward for MOSS-TTS Local.

The two bounded components follow the deployed MOSS-TTS reward contract::

    sim_reward = clip(cosine(WavLM-ECAPA(ref), WavLM-ECAPA(gen)), 0, 1)
    wer_reward = clip(1 - effective_english_wer, 0, 1)
    reward = 0.6 * sim_reward + 0.4 * wer_reward

Generated audio is materialized below a shared run directory because the SIM
Serving deliberately accepts allowlisted shared paths instead of audio bytes.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp

from miles.policies.moss_tts_local import wer_reward
from miles.policies.moss_tts_local.types import MediaArtifact
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

SIM_WEIGHT = 0.6
WER_WEIGHT = 0.4
REWARD_FORMULA_VERSION = "moss_tts_local_sim0.6_wer0.4_v1"
EXPECTED_SIM_MODEL = "wavlm_large_ecapa_tdnn"
MAX_SIM_ITEMS_PER_REQUEST = 16
_DEFAULT_API_KEY_ENV = "INSPIRE_API_KEY"
_RETRYABLE_HTTP_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_LONG_BASE64_REPLACEMENT = "<base64-redacted>"


class SIMServiceError(RuntimeError):
    """Raised when the configured speaker-SIM service cannot score audio."""


@dataclass(frozen=True)
class SIMConfig:
    url: str
    api_key: str
    candidate_dir: Path
    timeout_seconds: float = 180.0
    connect_timeout_seconds: float = 15.0
    max_retries: int = 3
    max_items_per_request: int = MAX_SIM_ITEMS_PER_REQUEST
    expected_model: str = EXPECTED_SIM_MODEL

    @classmethod
    def from_env(cls) -> "SIMConfig":
        url = os.getenv("MOSS_TTS_SIM_URL", "").strip()
        if not url:
            raise ValueError("MOSS_TTS_SIM_URL must name the speaker-SIM /v1/similarities endpoint.")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("MOSS_TTS_SIM_URL must be an absolute HTTP(S) URL.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("MOSS_TTS_SIM_URL must not embed credentials.")
        if not parsed.path.rstrip("/").endswith("/v1/similarities"):
            raise ValueError("MOSS_TTS_SIM_URL must end with /v1/similarities.")

        api_key_env = os.getenv(
            "MOSS_TTS_SIM_API_KEY_ENV",
            os.getenv("MOSS_TTS_WER_ASR_API_KEY_ENV", _DEFAULT_API_KEY_ENV),
        ).strip()
        if not api_key_env:
            raise ValueError("MOSS_TTS_SIM_API_KEY_ENV must not be empty.")
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            key_file = os.getenv(
                "MOSS_TTS_SIM_API_KEY_FILE",
                os.getenv("MOSS_TTS_WER_ASR_API_KEY_FILE", ""),
            ).strip()
            if key_file:
                api_key = _read_env_value(Path(key_file), api_key_env)
        if not api_key:
            raise ValueError(f"SIM API key is missing: set {api_key_env} or MOSS_TTS_SIM_API_KEY_FILE.")

        candidate_dir_value = os.getenv("MOSS_TTS_SIM_CANDIDATE_DIR", "").strip()
        if not candidate_dir_value:
            raise ValueError("MOSS_TTS_SIM_CANDIDATE_DIR must name a shared output directory.")
        candidate_dir = Path(candidate_dir_value)
        if not candidate_dir.is_absolute():
            raise ValueError("MOSS_TTS_SIM_CANDIDATE_DIR must be absolute.")

        timeout_seconds = float(os.getenv("MOSS_TTS_SIM_TIMEOUT_SECONDS", "180"))
        connect_timeout_seconds = float(os.getenv("MOSS_TTS_SIM_CONNECT_TIMEOUT_SECONDS", "15"))
        max_retries = int(os.getenv("MOSS_TTS_SIM_MAX_RETRIES", "3"))
        max_items_per_request = int(os.getenv("MOSS_TTS_SIM_MAX_ITEMS_PER_REQUEST", str(MAX_SIM_ITEMS_PER_REQUEST)))
        if timeout_seconds <= 0 or connect_timeout_seconds <= 0 or max_retries <= 0:
            raise ValueError("SIM timeout and retry settings must be positive.")
        if not 1 <= max_items_per_request <= MAX_SIM_ITEMS_PER_REQUEST:
            raise ValueError(f"MOSS_TTS_SIM_MAX_ITEMS_PER_REQUEST must be between 1 and {MAX_SIM_ITEMS_PER_REQUEST}.")
        expected_model = os.getenv("MOSS_TTS_SIM_EXPECTED_MODEL", EXPECTED_SIM_MODEL).strip()
        if not expected_model:
            raise ValueError("MOSS_TTS_SIM_EXPECTED_MODEL must not be empty.")
        return cls(
            url=url,
            api_key=api_key,
            candidate_dir=candidate_dir,
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            max_retries=max_retries,
            max_items_per_request=max_items_per_request,
            expected_model=expected_model,
        )


@dataclass(frozen=True)
class SimilarityScore:
    raw_cosine: float
    reward: float
    reference_cache_hit: bool
    reference_bucket_seconds: int | None
    candidate_bucket_seconds: int | None
    service_elapsed_ms: float
    candidate_sha256: str
    item_error_code: str | None = None


@dataclass(frozen=True)
class _SIMItem:
    result_index: int
    item_id: str
    reference_path: Path
    candidate_path: Path
    candidate_sha256: str


def _read_env_value(path: Path, name: str) -> str:
    if not path.is_file():
        raise ValueError(f"SIM API key file does not exist: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator or key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        return value.strip()
    raise ValueError(f"SIM API key file {path} does not define {name}.")


def _reference_audio_path(sample: Sample) -> Path:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    value = metadata.get("reference_audio_path")
    if value is None:
        references = metadata.get("reference_audios")
        if isinstance(references, list):
            matches = [
                item.get("path")
                for item in references
                if isinstance(item, dict)
                and isinstance(item.get("path"), str)
                and "timbre" in item.get("uses", [])
            ]
            if len(matches) == 1:
                value = matches[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "MOSS-TTS SIM reward requires metadata.reference_audio_path or one "
            "reference_audios item with the timbre use."
        )
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("MOSS-TTS SIM reference audio path must be absolute.")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("MOSS-TTS SIM reference audio must be a readable file.") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError("MOSS-TTS SIM reference audio must be a non-empty file.")
    return resolved


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_candidate(artifact: MediaArtifact, candidate_dir: Path) -> Path:
    """Atomically expose one validated inline WAV to the shared SIM service."""

    artifact.validate()
    if artifact.inline_base64 is None:
        raise ValueError("MOSS-TTS SIM reward requires an inline generated audio artifact.")
    if artifact.mime_type.casefold() not in ("audio/wav", "audio/x-wav"):
        raise ValueError("MOSS-TTS SIM reward requires a WAV artifact.")
    payload = base64.b64decode(artifact.inline_base64, validate=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    root = candidate_dir.resolve(strict=True)
    target = root / f"{artifact.sha256}.wav"
    if target.exists():
        if not target.is_file() or target.stat().st_size != artifact.num_bytes:
            raise ValueError("Existing SIM candidate artifact has inconsistent size.")
        if _sha256_path(target) != artifact.sha256:
            raise ValueError("Existing SIM candidate artifact has inconsistent checksum.")
        return target.resolve(strict=True)

    temporary = root / f".{artifact.sha256}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        file_descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    if _sha256_path(target) != artifact.sha256:
        raise ValueError("Materialized SIM candidate checksum mismatch.")
    return target.resolve(strict=True)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SIMServiceError(f"SIM response field {field} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise SIMServiceError(f"SIM response field {field} must be a finite number.")
    return number


def _optional_bucket(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SIMServiceError(f"SIM response field {field} must be a positive integer or null.")
    return value


def _parse_response(
    payload: Any,
    items: list[_SIMItem],
    *,
    expected_model: str = EXPECTED_SIM_MODEL,
) -> list[tuple[int, SimilarityScore]]:
    if not isinstance(payload, dict):
        raise SIMServiceError("SIM response must be a JSON object.")
    if payload.get("model") != expected_model:
        raise SIMServiceError(f"SIM response model must be {expected_model!r}.")
    elapsed_ms = _finite_number(payload.get("elapsed_ms"), "elapsed_ms")
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise SIMServiceError("SIM response requires a results list.")
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SIMServiceError("SIM response result rows must be objects.")
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id or item_id in rows_by_id:
            raise SIMServiceError("SIM response result ids must be unique non-empty strings.")
        rows_by_id[item_id] = row
    if set(rows_by_id) != {item.item_id for item in items}:
        raise SIMServiceError("SIM response result ids do not match the request.")

    parsed = []
    for item in items:
        row = rows_by_id[item.item_id]
        error = row.get("error")
        if error is not None:
            code = error.get("code") if isinstance(error, dict) else None
            if code != "audio_too_short":
                safe_code = code if isinstance(code, str) and code else "unknown"
                raise SIMServiceError(f"SIM item failed with code {safe_code}.")
            parsed.append(
                (
                    item.result_index,
                    SimilarityScore(
                        raw_cosine=0.0,
                        reward=0.0,
                        reference_cache_hit=bool(row.get("reference_cache_hit", False)),
                        reference_bucket_seconds=_optional_bucket(
                            row.get("reference_bucket_seconds"), "reference_bucket_seconds"
                        ),
                        candidate_bucket_seconds=_optional_bucket(
                            row.get("candidate_bucket_seconds"), "candidate_bucket_seconds"
                        ),
                        service_elapsed_ms=elapsed_ms,
                        candidate_sha256=item.candidate_sha256,
                        item_error_code=code,
                    ),
                )
            )
            continue
        raw_cosine = _finite_number(row.get("similarity"), "similarity")
        reward = _finite_number(row.get("reward_similarity"), "reward_similarity")
        if not 0.0 <= reward <= 1.0:
            raise SIMServiceError("SIM reward_similarity must be within [0, 1].")
        cache_hit = row.get("reference_cache_hit")
        if not isinstance(cache_hit, bool):
            raise SIMServiceError("SIM reference_cache_hit must be boolean.")
        parsed.append(
            (
                item.result_index,
                SimilarityScore(
                    raw_cosine=raw_cosine,
                    reward=reward,
                    reference_cache_hit=cache_hit,
                    reference_bucket_seconds=_optional_bucket(
                        row.get("reference_bucket_seconds"), "reference_bucket_seconds"
                    ),
                    candidate_bucket_seconds=_optional_bucket(
                        row.get("candidate_bucket_seconds"), "candidate_bucket_seconds"
                    ),
                    service_elapsed_ms=elapsed_ms,
                    candidate_sha256=item.candidate_sha256,
                ),
            )
        )
    return parsed


def _safe_error_body(body: str) -> str:
    value = str(body or "").replace("\n", " ")
    if len(value) > 80 and all(character.isalnum() or character in "+/=" for character in value):
        value = _LONG_BASE64_REPLACEMENT
    return value[:800]


def _reference_affinity_key(reference_path: Path) -> str:
    return hashlib.sha256(b"moss-tts-local-sim-ref-v1\0" + os.fsencode(reference_path)).hexdigest()


async def _score_chunk(
    session: aiohttp.ClientSession,
    config: SIMConfig,
    items: list[_SIMItem],
) -> list[tuple[int, SimilarityScore]]:
    payload = {
        "items": [
            {
                "id": item.item_id,
                "reference_path": os.fspath(item.reference_path),
                "candidate_path": os.fspath(item.candidate_path),
            }
            for item in items
        ]
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "x-inspire-inference-key": _reference_affinity_key(items[0].reference_path),
    }
    for attempt in range(config.max_retries):
        try:
            async with session.post(config.url, json=payload, headers=headers) as response:
                body = await response.text()
                if response.status >= 400:
                    error = SIMServiceError(f"SIM HTTP {response.status}: {_safe_error_body(body)}")
                    if response.status not in _RETRYABLE_HTTP_STATUS:
                        raise error
                    if attempt + 1 >= config.max_retries:
                        raise error
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message="retryable SIM response",
                        headers=response.headers,
                    )
                try:
                    response_payload = json.loads(body)
                except json.JSONDecodeError as exc:
                    raise SIMServiceError(f"SIM returned non-JSON content: {_safe_error_body(body)}") from exc
                return _parse_response(
                    response_payload,
                    items,
                    expected_model=config.expected_model,
                )
        except SIMServiceError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt + 1 >= config.max_retries:
                raise SIMServiceError(
                    f"SIM scoring failed after {attempt + 1} attempt(s): {type(exc).__name__}"
                ) from exc
            delay = min(2**attempt, 8) + random.random() * 0.25
            logger.warning(
                "SIM scoring attempt %s/%s failed with %s; retrying in %.2fs.",
                attempt + 1,
                config.max_retries,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)
    raise AssertionError("SIM retry loop exited unexpectedly.")


async def score_similarity_batch(
    samples: list[Sample],
    *,
    config: SIMConfig | None = None,
) -> list[SimilarityScore]:
    if not samples:
        return []
    config = config or SIMConfig.from_env()
    grouped: dict[Path, list[_SIMItem]] = defaultdict(list)
    for result_index, sample in enumerate(samples):
        if sample.index is None:
            raise ValueError("MOSS-TTS SIM reward requires sample.index.")
        if len(sample.artifacts) != 1:
            raise ValueError(f"MOSS-TTS SIM reward requires exactly one audio artifact, got {len(sample.artifacts)}.")
        reference_path = _reference_audio_path(sample)
        candidate_path = materialize_candidate(sample.artifacts[0], config.candidate_dir)
        item = _SIMItem(
            result_index=result_index,
            item_id=f"sample-{sample.index}-{sample.artifacts[0].sha256[:16]}",
            reference_path=reference_path,
            candidate_path=candidate_path,
            candidate_sha256=sample.artifacts[0].sha256,
        )
        grouped[reference_path].append(item)

    chunks = [
        items[start : start + config.max_items_per_request]
        for items in grouped.values()
        for start in range(0, len(items), config.max_items_per_request)
    ]
    timeout = aiohttp.ClientTimeout(
        total=config.timeout_seconds,
        connect=min(config.connect_timeout_seconds, config.timeout_seconds),
        sock_connect=min(config.connect_timeout_seconds, config.timeout_seconds),
    )
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        chunk_results = await asyncio.gather(*(_score_chunk(session, config, chunk) for chunk in chunks))
    results: list[SimilarityScore | None] = [None] * len(samples)
    for chunk_result in chunk_results:
        for result_index, score in chunk_result:
            results[result_index] = score
    if any(result is None for result in results):
        raise RuntimeError("SIM batch result is incomplete.")
    return [result for result in results if result is not None]


async def reward_batch(args: Any, samples: list[Sample], **kwargs: Any) -> list[float]:
    """Miles group-RM entry point for the fixed 0.6 SIM + 0.4 WER objective."""

    if kwargs:
        logger.debug("Ignoring MOSS-TTS mixed-reward auxiliary keys: %s", sorted(kwargs))
    if not samples:
        return []
    wer_values, similarity_scores = await asyncio.gather(
        asyncio.gather(*(wer_reward.reward_func(args, sample) for sample in samples)),
        score_similarity_batch(samples),
    )
    rewards = []
    for sample, wer_value, similarity in zip(samples, wer_values, similarity_scores, strict=True):
        unbounded_wer_reward = float(wer_value)
        if not math.isfinite(unbounded_wer_reward):
            raise FloatingPointError("WER component reward is NaN or Inf.")
        bounded_wer_reward = max(0.0, min(1.0, unbounded_wer_reward))
        reward = SIM_WEIGHT * similarity.reward + WER_WEIGHT * bounded_wer_reward
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise FloatingPointError("MOSS-TTS mixed reward must be finite and within [0, 1].")
        sample.metadata = dict(sample.metadata or {})
        sample.metadata.update(
            {
                "sim_model": EXPECTED_SIM_MODEL,
                "sim_raw_cosine": similarity.raw_cosine,
                "sim_reward": similarity.reward,
                "sim_reference_cache_hit": similarity.reference_cache_hit,
                "sim_reference_bucket_seconds": similarity.reference_bucket_seconds,
                "sim_candidate_bucket_seconds": similarity.candidate_bucket_seconds,
                "sim_service_elapsed_ms": similarity.service_elapsed_ms,
                "sim_candidate_sha256": similarity.candidate_sha256,
                "sim_item_error_code": similarity.item_error_code,
                "wer_reward_unbounded": unbounded_wer_reward,
                "wer_reward_bounded": bounded_wer_reward,
                "mixed_reward": reward,
                "mixed_reward_formula": REWARD_FORMULA_VERSION,
                "reward_components": {
                    "reference_similarity": {
                        "weight": SIM_WEIGHT,
                        "reward": similarity.reward,
                        "raw": similarity.raw_cosine,
                    },
                    "wer": {
                        "weight": WER_WEIGHT,
                        "reward": bounded_wer_reward,
                        "raw": float(sample.metadata["wer"]),
                    },
                },
            }
        )
        rewards.append(reward)
    return rewards


async def bounded_wer_reward_func(args: Any, sample: Sample, **kwargs: Any) -> float:
    """Return exactly the bounded WER component used by the mixed objective."""

    value = float(await wer_reward.reward_func(args, sample, **kwargs))
    if not math.isfinite(value):
        raise FloatingPointError("WER component reward is NaN or Inf.")
    bounded = max(0.0, min(1.0, value))
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["wer_reward_unbounded"] = value
    sample.metadata["wer_reward_bounded"] = bounded
    return bounded


async def weighted_wer_component_reward_func(
    args: Any,
    sample: Sample,
    **kwargs: Any,
) -> float:
    """Return the mixed objective's WER component, including its 0.4 weight."""

    bounded = await bounded_wer_reward_func(args, sample, **kwargs)
    weighted = WER_WEIGHT * bounded
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["wer_component_weight"] = WER_WEIGHT
    sample.metadata["weighted_wer_component_reward"] = weighted
    return weighted


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> float:
    """Per-sample entry point used by evaluation and small probes."""

    return (await reward_batch(args, [sample], **kwargs))[0]
