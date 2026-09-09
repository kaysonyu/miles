"""AnyAudio-style audio reward-model client for MOSS-TTS Local.

The RM contract mirrors the slime Judge implementation: each rubric item is
sent as an independent deterministic yes/no request and its targeted token
logprobs are converted to ``p(yes)``.  The sample reward is the mean over its
rubric items.  This module is opt-in; missing rubric metadata is an explicit
configuration error rather than a silent zero.
"""

from __future__ import annotations

import asyncio
import base64
import math
import os
import re
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import aiohttp

from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

_YES_NO_VARIANTS = {
    "yes": ("yes", " yes", "Yes", " Yes", "YES", " YES"),
    "no": ("no", " no", "No", " No", "NO", " NO"),
}
_DEFAULT_MODEL = "qwen3-omni-30b-a3b-thinker"
_DEFAULT_TOKENIZER = "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/AnyAudio-Judge-30B"
_SYSTEM_PROMPT = "你是一位专业的音频感知评估专家。请只回答 yes 或 no。"
_MAX_RUBRICS = 32
_MAX_DIMENSION_CHARS = 128
_MAX_STATEMENT_CHARS = 512


@dataclass(frozen=True)
class RMConfig:
    url: str
    model: str
    api_key: str
    tokenizer_path: str
    timeout_seconds: float = 300.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "RMConfig":
        url = os.getenv("MOSS_TTS_RM_URL", os.getenv("MOSS_TTS_JUDGE_URL", "")).strip()
        if not url:
            raise ValueError("MOSS_TTS_RM_URL must be set when the RM component is active.")
        key_env = os.getenv("MOSS_TTS_RM_API_KEY_ENV", os.getenv("MOSS_TTS_WER_ASR_API_KEY_ENV", "INSPIRE_API_KEY"))
        api_key = os.getenv(key_env, "").strip()
        if not api_key:
            key_file = os.getenv("MOSS_TTS_RM_API_KEY_FILE", os.getenv("MOSS_TTS_WER_ASR_API_KEY_FILE", "")).strip()
            if key_file:
                api_key = _read_env_value(Path(key_file), key_env)
        if not api_key:
            raise ValueError(f"RM API key is missing: set {key_env} or MOSS_TTS_RM_API_KEY_FILE.")
        timeout_seconds = float(os.getenv("MOSS_TTS_RM_TIMEOUT_SECONDS", "300"))
        max_retries = int(os.getenv("MOSS_TTS_RM_MAX_RETRIES", "2"))
        if timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("RM timeout must be positive and retry count must be non-negative.")
        return cls(
            url=url,
            model=os.getenv("MOSS_TTS_RM_MODEL", _DEFAULT_MODEL).strip(),
            api_key=api_key,
            tokenizer_path=os.getenv("MOSS_TTS_RM_TOKENIZER_PATH", _DEFAULT_TOKENIZER).strip(),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )


def _read_env_value(path: Path, name: str) -> str:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            value = value.strip()
            return value[1:-1] if len(value) >= 2 and value[0] == value[-1] else value
    raise ValueError(f"RM API key file {path} does not define {name}.")


def _label_ids(tokenizer: Any, label: str) -> tuple[int, ...]:
    values: set[int] = set()
    for variant in _YES_NO_VARIANTS[label]:
        for token_id in tokenizer.encode(variant, add_special_tokens=False):
            if tokenizer.decode([token_id]).strip().casefold() == label:
                values.add(int(token_id))
    return tuple(sorted(values))


@lru_cache(maxsize=4)
def _token_ids(path: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    tokenizer = load_tokenizer(path, trust_remote_code=True)
    yes, no = _label_ids(tokenizer, "yes"), _label_ids(tokenizer, "no")
    if not yes or not no:
        raise ValueError("RM tokenizer must expose yes and no token IDs.")
    return yes, no


def _probability(response: Any, yes_ids: tuple[int, ...], no_ids: tuple[int, ...]) -> float:
    try:
        entries = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("RM response is missing targeted token logprobs.") from error
    values = {"yes": [], "no": []}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        token = entry.get("token")
        match = re.fullmatch(r"token_id:(\d+)", str(token))
        logprob = entry.get("logprob")
        if not match or not isinstance(logprob, (int, float)):
            continue
        token_id = int(match.group(1))
        if token_id in yes_ids:
            values["yes"].append(float(logprob))
        elif token_id in no_ids:
            values["no"].append(float(logprob))
    if not values["yes"] or not values["no"]:
        raise ValueError("RM response must contain both yes and no targeted logprobs.")
    maximum = max(max(values["yes"]), max(values["no"]))
    yes = math.exp(max(values["yes"]) - maximum)
    no = math.exp(max(values["no"]) - maximum)
    return yes / (yes + no)


def _audio_data_url(sample: Sample) -> str:
    if len(sample.artifacts) != 1 or sample.artifacts[0].inline_base64 is None:
        raise ValueError("RM reward requires exactly one inline audio artifact.")
    artifact = sample.artifacts[0]
    artifact.validate()
    return f"data:{artifact.mime_type};base64,{artifact.inline_base64}"


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> float:
    del args, kwargs
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    raw_rubric = _parse_rubric(metadata)
    config = RMConfig.from_env()
    yes_ids, no_ids = _token_ids(config.tokenizer_path)
    audio_url = _audio_data_url(sample)
    timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)

    async def score_one(item: dict[str, Any]) -> float:
        dimension = str(item.get("dimension", "")).strip()
        statement = str(item.get("statement", "")).strip()
        if not dimension or not statement:
            raise ValueError("Each RM rubric item requires dimension and statement.")
        payload = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_url}},
                        {"type": "text", "text": f"请判断：【{dimension}】{statement}"},
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": True,
            "logprob_token_ids": [*yes_ids, *no_ids],
            "return_tokens_as_token_ids": True,
        }
        for attempt in range(config.max_retries + 1):
            try:
                headers = {"Authorization": f"Bearer {config.api_key}"}
                async with session.post(config.url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    return _probability(await response.json(content_type=None), yes_ids, no_ids)
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt >= config.max_retries:
                    raise
                await asyncio.sleep(min(2**attempt, 8))
        raise AssertionError("RM retry loop exited unexpectedly.")

    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        values = await asyncio.gather(*(score_one(item) for item in raw_rubric))
    reward = sum(values) / len(values)
    metadata = dict(metadata)
    metadata["rm_rubric_scores"] = values
    metadata["rm_model"] = config.model
    metadata["rm_reward"] = reward
    current_metadata = dict(sample.metadata or {})
    current_metadata.update(
        {
            "rm_rubric_scores": values,
            "rm_model": config.model,
            "rm_reward": reward,
        }
    )
    sample.metadata = current_metadata
    return reward


def _parse_rubric(metadata: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    raw_rubric = metadata.get("rubric")
    if not isinstance(raw_rubric, list) or not 1 <= len(raw_rubric) <= _MAX_RUBRICS:
        raise ValueError(f"RM reward requires between 1 and {_MAX_RUBRICS} metadata.rubric items.")
    parsed: list[dict[str, str]] = []
    for index, item in enumerate(raw_rubric):
        if not isinstance(item, Mapping):
            raise TypeError(f"metadata.rubric[{index}] must be an object.")
        dimension = item.get("dimension")
        statement = item.get("statement")
        if not isinstance(dimension, str) or not dimension.strip():
            raise ValueError(f"metadata.rubric[{index}].dimension must be a non-empty string.")
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError(f"metadata.rubric[{index}].statement must be a non-empty string.")
        dimension = dimension.strip()
        statement = statement.strip()
        if len(dimension) > _MAX_DIMENSION_CHARS:
            raise ValueError(f"metadata.rubric[{index}].dimension is too long.")
        if len(statement) > _MAX_STATEMENT_CHARS:
            raise ValueError(f"metadata.rubric[{index}].statement is too long.")
        parsed.append({"dimension": dimension, "statement": statement})
    return tuple(parsed)
