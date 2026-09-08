"""Qwen3-ASR-backed word-error-rate reward for MOSS-TTS Local.

The reward is intentionally single-objective::

    reward = 1 - effective_english_WER(reference_text, ASR(generated_audio))

Healthy English transcripts use unclipped raw WER. Explicit language mismatch
or ASR decoder failure maps to one stable 100%-error result; raw transcript WER
is retained separately for audit. GRPO's per-prompt affine normalization still
preserves reward ordering within each prompt group.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import random
import re
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp

from miles.policies.moss_tts_local.types import MediaArtifact
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_DEFAULT_ASR_MODEL = "qwen3-asr-1.7b"
_DEFAULT_API_KEY_ENV = "INSPIRE_API_KEY"
_ASR_TEXT_MARKER = "<asr_text>"
_ENGLISH_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
_LONG_BASE64_RE = re.compile(r"[A-Za-z0-9+/=]{80,}")
_APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u02bc": "'",
        "\uff07": "'",
    }
)


class ASRServiceError(RuntimeError):
    """Raised when the configured ASR service cannot score an artifact."""


@dataclass(frozen=True)
class ASRTranscription:
    """Parsed ASR text plus decoder termination metadata."""

    language: str
    transcript: str
    finish_reason: str = ""
    completion_tokens: int | None = None


@dataclass(frozen=True)
class ASRConfig:
    url: str
    model: str
    api_key: str
    timeout_seconds: float = 180.0
    connect_timeout_seconds: float = 15.0
    max_retries: int = 3
    max_tokens: int = 512
    max_words_per_second: float = 12.0
    repeats: int = 1

    @classmethod
    def from_env(cls) -> "ASRConfig":
        url = os.getenv("MOSS_TTS_WER_ASR_URL", "").strip()
        if not url:
            raise ValueError("MOSS_TTS_WER_ASR_URL must name the Qwen3-ASR OpenAI-compatible endpoint.")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("MOSS_TTS_WER_ASR_URL must be an absolute HTTP(S) URL.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("MOSS_TTS_WER_ASR_URL must not embed credentials.")

        api_key_env = os.getenv("MOSS_TTS_WER_ASR_API_KEY_ENV", _DEFAULT_API_KEY_ENV).strip()
        if not api_key_env:
            raise ValueError("MOSS_TTS_WER_ASR_API_KEY_ENV must not be empty.")
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            key_file = os.getenv("MOSS_TTS_WER_ASR_API_KEY_FILE", "").strip()
            if key_file:
                api_key = _read_env_value(Path(key_file), api_key_env)
        if not api_key:
            raise ValueError(f"ASR API key is missing: set {api_key_env} or MOSS_TTS_WER_ASR_API_KEY_FILE.")

        model = os.getenv("MOSS_TTS_WER_ASR_MODEL", _DEFAULT_ASR_MODEL).strip()
        if not model:
            raise ValueError("MOSS_TTS_WER_ASR_MODEL must not be empty.")
        timeout_seconds = float(os.getenv("MOSS_TTS_WER_ASR_TIMEOUT_SECONDS", "180"))
        connect_timeout_seconds = float(os.getenv("MOSS_TTS_WER_ASR_CONNECT_TIMEOUT_SECONDS", "15"))
        max_retries = int(os.getenv("MOSS_TTS_WER_ASR_MAX_RETRIES", "3"))
        max_tokens = int(os.getenv("MOSS_TTS_WER_ASR_MAX_TOKENS", "512"))
        max_words_per_second = float(os.getenv("MOSS_TTS_WER_ASR_MAX_WORDS_PER_SECOND", "12"))
        repeats = int(os.getenv("MOSS_TTS_WER_ASR_REPEATS", "1"))
        if (
            timeout_seconds <= 0
            or connect_timeout_seconds <= 0
            or max_retries <= 0
            or max_tokens <= 0
            or max_words_per_second <= 0
        ):
            raise ValueError("ASR timeout, retry count, max token count, and max word rate must all be positive.")
        if repeats <= 0 or repeats % 2 == 0:
            raise ValueError("MOSS_TTS_WER_ASR_REPEATS must be a positive odd integer.")
        return cls(
            url=url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            max_retries=max_retries,
            max_tokens=max_tokens,
            max_words_per_second=max_words_per_second,
            repeats=repeats,
        )


@dataclass(frozen=True)
class WordErrorRateScore:
    reference_normalized: str
    hypothesis_normalized: str
    reference_words: tuple[str, ...]
    hypothesis_words: tuple[str, ...]
    substitutions: int
    deletions: int
    insertions: int
    hits: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if not self.reference_words:
            raise ValueError("WER requires a non-empty normalized reference.")
        return self.errors / len(self.reference_words)

    @property
    def reward(self) -> float:
        return 1.0 - self.wer

    @property
    def exact_match(self) -> bool:
        return self.errors == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_normalized": self.reference_normalized,
            "hypothesis_normalized": self.hypothesis_normalized,
            "reference_word_count": len(self.reference_words),
            "hypothesis_word_count": len(self.hypothesis_words),
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "hits": self.hits,
            "errors": self.errors,
            "wer": self.wer,
            "reward": self.reward,
            "exact_match": self.exact_match,
        }


def _read_env_value(path: Path, name: str) -> str:
    """Read one shell-style KEY=VALUE entry without evaluating the file."""

    if not path.is_file():
        raise ValueError(f"ASR API key file does not exist: {path}")
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
    raise ValueError(f"ASR API key file {path} does not define {name}.")


def normalize_english_for_wer(text: str) -> str:
    """Apply the experiment's explicit, dependency-free English normalization."""

    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.translate(_APOSTROPHE_TRANSLATION).casefold()
    return " ".join(_ENGLISH_WORD_RE.findall(normalized))


def _edit_counts(reference: tuple[str, ...], hypothesis: tuple[str, ...]) -> tuple[int, int, int, int]:
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    operations = [[""] * columns for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row
        operations[row][0] = "delete"
    for column in range(1, columns):
        costs[0][column] = column
        operations[0][column] = "insert"

    operation_priority = {"match": 0, "substitute": 1, "delete": 2, "insert": 3}
    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                costs[row][column] = costs[row - 1][column - 1]
                operations[row][column] = "match"
                continue
            candidates = (
                (costs[row - 1][column - 1] + 1, "substitute"),
                (costs[row - 1][column] + 1, "delete"),
                (costs[row][column - 1] + 1, "insert"),
            )
            costs[row][column], operations[row][column] = min(
                candidates,
                key=lambda candidate: (candidate[0], operation_priority[candidate[1]]),
            )

    substitutions = deletions = insertions = hits = 0
    row = len(reference)
    column = len(hypothesis)
    while row or column:
        operation = operations[row][column]
        if operation == "match":
            hits += 1
            row -= 1
            column -= 1
        elif operation == "substitute":
            substitutions += 1
            row -= 1
            column -= 1
        elif operation == "delete":
            deletions += 1
            row -= 1
        elif operation == "insert":
            insertions += 1
            column -= 1
        else:
            raise AssertionError(f"Missing WER backtrace operation at row={row}, column={column}.")
    return substitutions, deletions, insertions, hits


def compute_word_error_rate(reference: str, hypothesis: str) -> WordErrorRateScore:
    reference_normalized = normalize_english_for_wer(reference)
    hypothesis_normalized = normalize_english_for_wer(hypothesis)
    reference_words = tuple(reference_normalized.split())
    hypothesis_words = tuple(hypothesis_normalized.split())
    if not reference_words:
        raise ValueError("WER reference is empty after English normalization.")
    substitutions, deletions, insertions, hits = _edit_counts(reference_words, hypothesis_words)
    return WordErrorRateScore(
        reference_normalized=reference_normalized,
        hypothesis_normalized=hypothesis_normalized,
        reference_words=reference_words,
        hypothesis_words=hypothesis_words,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        hits=hits,
    )


def parse_qwen3_asr_output(raw_output: Any) -> tuple[str, str]:
    """Return ``(language, transcript)`` from Qwen3-ASR chat or transcription output."""

    if raw_output is None:
        return "", ""
    if isinstance(raw_output, list):
        raw_output = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in raw_output)
    value = str(raw_output).strip()
    if not value:
        return "", ""

    marker_index = value.casefold().find(_ASR_TEXT_MARKER)
    if marker_index < 0:
        return "", value
    prefix = value[:marker_index]
    transcript = value[marker_index + len(_ASR_TEXT_MARKER) :].strip()
    language_match = re.search(r"language\s+([^\n<]+)", prefix, flags=re.IGNORECASE)
    language = language_match.group(1).strip() if language_match else ""
    if language.casefold() == "none":
        return "", ""
    return language, transcript


def _safe_error_body(body: str) -> str:
    value = _LONG_BASE64_RE.sub("<base64-redacted>", str(body or "").replace("\n", " "))
    return value[:800]


def _extract_response_content(payload: Any) -> ASRTranscription:
    if not isinstance(payload, dict):
        raise ASRServiceError(f"ASR response must be a JSON object, got {type(payload).__name__}.")
    if "text" in payload:
        language, transcript = parse_qwen3_asr_output(payload.get("text"))
        return ASRTranscription(language=language, transcript=transcript)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ASRServiceError("ASR response contains neither text nor choices[0].message.content.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ASRServiceError("ASR response choices[0].message is missing.")
    language, transcript = parse_qwen3_asr_output(message.get("content"))
    finish_reason = str(choices[0].get("finish_reason") or "")
    usage = payload.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    return ASRTranscription(
        language=language,
        transcript=transcript,
        finish_reason=finish_reason,
        completion_tokens=int(completion_tokens) if completion_tokens is not None else None,
    )


async def _post_asr_request(
    session: aiohttp.ClientSession,
    config: ASRConfig,
    artifact: MediaArtifact,
) -> ASRTranscription:
    headers = {"Authorization": f"Bearer {config.api_key}"}
    if urlparse(config.url).path.rstrip("/").endswith("/audio/transcriptions"):
        if artifact.inline_base64 is None:
            raise ValueError("The OpenAI transcription endpoint requires an inline audio artifact.")
        form = aiohttp.FormData()
        form.add_field("model", config.model)
        form.add_field(
            "file",
            base64.b64decode(artifact.inline_base64, validate=True),
            filename="generated.wav",
            content_type=artifact.mime_type,
        )
        response_context = session.post(config.url, data=form, headers=headers)
    else:
        if artifact.inline_base64 is None:
            raise ValueError("Qwen3-ASR chat scoring requires an inline audio artifact.")
        payload = {
            "model": config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": f"data:{artifact.mime_type};base64,{artifact.inline_base64}",
                            },
                        }
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": config.max_tokens,
            "stream": False,
        }
        response_context = session.post(
            config.url, json=payload, headers={**headers, "Content-Type": "application/json"}
        )

    async with response_context as response:
        body = await response.text()
        if response.status >= 400:
            raise ASRServiceError(f"ASR HTTP {response.status}: {_safe_error_body(body)}")
        try:
            payload = await response.json(content_type=None)
        except Exception as exc:
            raise ASRServiceError(f"ASR returned non-JSON content: {_safe_error_body(body)}") from exc
        return _extract_response_content(payload)


async def transcribe_media_artifact_detailed(
    artifact: MediaArtifact,
    *,
    config: ASRConfig | None = None,
) -> ASRTranscription:
    """Transcribe one generated WAV with decoder metadata and bounded retries."""

    artifact.validate()
    config = config or ASRConfig.from_env()
    timeout = aiohttp.ClientTimeout(
        total=config.timeout_seconds,
        connect=min(config.connect_timeout_seconds, config.timeout_seconds),
        sock_connect=min(config.connect_timeout_seconds, config.timeout_seconds),
    )
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        return await _transcribe_with_session(session, artifact, config=config)


async def _transcribe_with_session(
    session: aiohttp.ClientSession,
    artifact: MediaArtifact,
    *,
    config: ASRConfig,
) -> ASRTranscription:
    for attempt in range(config.max_retries):
        try:
            return await _post_asr_request(session, config, artifact)
        except (aiohttp.ClientError, asyncio.TimeoutError, ASRServiceError) as exc:
            retryable = not isinstance(exc, ASRServiceError) or any(
                f"HTTP {status}" in str(exc) for status in (408, 409, 425, 429, 500, 502, 503, 504)
            )
            if not retryable or attempt + 1 >= config.max_retries:
                raise ASRServiceError(
                    f"ASR scoring failed after {attempt + 1} attempt(s): {_safe_error_body(str(exc))}"
                ) from exc
            delay = min(2**attempt, 8) + random.random() * 0.25
            logger.warning(
                "ASR scoring attempt %s/%s failed with %s; retrying in %.2fs.",
                attempt + 1,
                config.max_retries,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)
    raise AssertionError("ASR retry loop exited unexpectedly.")


async def transcribe_media_artifact_repeated(
    artifact: MediaArtifact,
    *,
    config: ASRConfig | None = None,
) -> list[ASRTranscription]:
    """Run repeated ASR decodes over one keep-alive HTTP session."""

    artifact.validate()
    config = config or ASRConfig.from_env()
    timeout = aiohttp.ClientTimeout(
        total=config.timeout_seconds,
        connect=min(config.connect_timeout_seconds, config.timeout_seconds),
        sock_connect=min(config.connect_timeout_seconds, config.timeout_seconds),
    )
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        return [await _transcribe_with_session(session, artifact, config=config) for _ in range(config.repeats)]


async def transcribe_media_artifact(
    artifact: MediaArtifact,
    *,
    config: ASRConfig | None = None,
) -> tuple[str, str]:
    """Backward-compatible transcription API returning ``(language, text)``."""

    result = await transcribe_media_artifact_detailed(artifact, config=config)
    return result.language, result.transcript


def _wav_duration_seconds(artifact: MediaArtifact) -> float | None:
    if artifact.inline_base64 is None or artifact.mime_type.casefold() not in ("audio/wav", "audio/x-wav"):
        return None
    try:
        payload = base64.b64decode(artifact.inline_base64, validate=True)
        with wave.open(io.BytesIO(payload), "rb") as reader:
            return reader.getnframes() / reader.getframerate()
    except (ValueError, wave.Error, EOFError, ZeroDivisionError):
        return None


def _is_english_language(language: str) -> bool:
    normalized = str(language or "").strip().casefold().replace("_", "-")
    return not normalized or normalized == "english" or normalized == "en" or normalized.startswith("en-")


@dataclass(frozen=True)
class _ScoredTranscription:
    transcription: ASRTranscription
    raw_score: WordErrorRateScore
    score: WordErrorRateScore
    raw_word_count: int
    word_rate: float | None
    language_mismatch: bool
    quality_flags: tuple[str, ...]
    effective_reason: str

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "language": self.transcription.language,
            "transcript": self.transcription.transcript,
            "finish_reason": self.transcription.finish_reason,
            "completion_tokens": self.transcription.completion_tokens,
            "wer": self.score.wer,
            "wer_raw": self.raw_score.wer,
            "effective_reason": self.effective_reason,
            "quality_flags": list(self.quality_flags),
            "language_mismatch": self.language_mismatch,
        }


def _score_transcription(
    reference: str,
    transcription: ASRTranscription,
    *,
    duration_seconds: float | None,
    max_words_per_second: float,
) -> _ScoredTranscription:
    raw_score = compute_word_error_rate(reference, transcription.transcript)
    raw_word_count = len(str(transcription.transcript or "").split())
    word_rate = raw_word_count / duration_seconds if duration_seconds and duration_seconds > 0 else None
    language_mismatch = not _is_english_language(transcription.language)
    quality_flags = []
    if transcription.finish_reason.casefold() == "length":
        quality_flags.append("decoder_length_truncated")
    if word_rate is not None and word_rate > max_words_per_second:
        quality_flags.append("implausible_word_rate")

    effective_reason = "raw_english_wer"
    if language_mismatch:
        effective_reason = "language_mismatch"
    elif quality_flags:
        effective_reason = "asr_quality_failure"
    score = compute_word_error_rate(reference, "") if effective_reason != "raw_english_wer" else raw_score
    return _ScoredTranscription(
        transcription=transcription,
        raw_score=raw_score,
        score=score,
        raw_word_count=raw_word_count,
        word_rate=word_rate,
        language_mismatch=language_mismatch,
        quality_flags=tuple(quality_flags),
        effective_reason=effective_reason,
    )


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """Miles entry point for cap-independent effective English WER reward.

    Raw transcript WER remains in ``wer_raw``.  Explicit non-English language,
    decoder length truncation, and physically implausible transcript word rate
    are mapped to one stable 100%-error English failure instead of allowing ASR
    token limits or writing system to choose an arbitrary reward magnitude.
    """

    del args, kwargs
    if not isinstance(sample.label, str) or not sample.label.strip():
        raise ValueError("MOSS-TTS WER reward requires a non-empty string Sample.label.")
    if len(sample.artifacts) != 1:
        raise ValueError(f"MOSS-TTS WER reward requires exactly one audio artifact, got {len(sample.artifacts)}.")
    config = ASRConfig.from_env()
    duration_seconds = _wav_duration_seconds(sample.artifacts[0])
    scored_transcriptions = []
    transcriptions = await transcribe_media_artifact_repeated(sample.artifacts[0], config=config)
    for transcription in transcriptions:
        scored_transcriptions.append(
            _score_transcription(
                sample.label,
                transcription,
                duration_seconds=duration_seconds,
                max_words_per_second=config.max_words_per_second,
            )
        )
    selected = sorted(scored_transcriptions, key=lambda item: item.score.wer)[config.repeats // 2]
    transcription = selected.transcription
    raw_score = selected.raw_score
    score = selected.score
    quality_flags = list(selected.quality_flags)
    repeat_audit = [item.to_audit_dict() for item in scored_transcriptions]
    repeat_wers = [item.score.wer for item in scored_transcriptions]
    repeat_disagreement = (
        len(
            {
                (item.transcription.language, item.transcription.transcript, item.transcription.finish_reason)
                for item in scored_transcriptions
            }
        )
        > 1
    )
    sample.metadata = dict(sample.metadata or {})
    sample.metadata.update(
        {
            "asr_language": transcription.language,
            "asr_transcript": transcription.transcript,
            "asr_finish_reason": transcription.finish_reason,
            "asr_completion_tokens": transcription.completion_tokens,
            "asr_raw_word_count": selected.raw_word_count,
            "asr_audio_duration_seconds": duration_seconds,
            "asr_words_per_second": selected.word_rate,
            "asr_quality_flags": quality_flags,
            "asr_language_mismatch": selected.language_mismatch,
            "asr_repeat_count": config.repeats,
            "asr_repeat_disagreement": repeat_disagreement,
            "asr_repeat_wer_spread": max(repeat_wers) - min(repeat_wers),
            "asr_repeats": repeat_audit,
            "wer": score.wer,
            "wer_raw": raw_score.wer,
            "wer_reward": score.reward,
            "wer_effective_reason": selected.effective_reason,
            "wer_details": {**score.to_dict(), "effective_reason": selected.effective_reason},
            "wer_raw_details": raw_score.to_dict(),
        }
    )
    return score.reward
