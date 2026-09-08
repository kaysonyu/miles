"""External SGLang-Omni rollout for the mossLite MOSS-TTS Local policy."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import uuid
from argparse import Namespace
from contextlib import asynccontextmanager
from typing import Any

from miles.backends.sglang_omni_utils.http_adapter import SGLangOmniHttpAdapter
from miles.policies.moss_tts_local.trace_codec import decode_moss_tts_local_trace
from miles.policies.moss_tts_local.types import MediaArtifact
from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.utils.async_utils import run
from miles.utils.misc import SingletonMeta
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _audio_mime_type(audio_format: str | None) -> str:
    normalized = str(audio_format or "wav").lower()
    aliases = {"pcm": "audio/L16", "mp3": "audio/mpeg"}
    return aliases.get(normalized, f"audio/{normalized}")


def build_moss_tts_local_request(
    args: Namespace,
    sample: Sample,
    *,
    rollout_id: int,
    seed: int,
) -> dict[str, Any]:
    """Map one Miles sample to the existing Omni ``POST /generate`` request."""

    temperature = float(args.rollout_temperature)
    max_new_tokens = int(args.rollout_max_response_len)
    metadata = dict(sample.metadata or {})
    tts_params = dict(metadata.get("tts_params") or {})
    reference_audio_path = metadata.get("reference_audio_path")
    if reference_audio_path is not None:
        if not isinstance(reference_audio_path, str) or not reference_audio_path.strip():
            raise ValueError("MOSS-TTS Local reference_audio_path metadata must be a non-empty string.")
        configured_reference = tts_params.get("ref_audio")
        if configured_reference is not None and configured_reference != reference_audio_path:
            raise ValueError(
                "MOSS-TTS Local reference audio is inconsistent between metadata.reference_audio_path "
                "and metadata.tts_params.ref_audio."
            )
        # SGLang-Omni's MOSS-TTS Local request builder resolves ``ref_audio``
        # before rendering moss_tts_v2.  Keep the canonical dataset field out
        # of the engine-specific tts_params until this request boundary.
        tts_params["ref_audio"] = reference_audio_path
    tts_params.update(
        {
            "text_temperature": temperature,
            "text_top_p": 1.0,
            "text_top_k": -1,
            "audio_temperature": temperature,
            "audio_top_p": 1.0,
            "audio_top_k": -1,
            "audio_repetition_penalty": 1.0,
            "seed": int(seed),
        }
    )
    request_metadata = {
        **metadata,
        "request_id": str(uuid.uuid4()),
        "rollout_id": int(rollout_id),
        "sample_id": sample.index,
        "tts_params": tts_params,
    }
    payload: dict[str, Any] = {
        "sampling_params": {
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": max_new_tokens,
            "seed": int(seed),
        },
        "stage_params": {
            getattr(args, "sglang_omni_train_stage", "tts_engine"): tts_params,
        },
        "output_modalities": ["audio"],
        "return_logprob": True,
        "return_omni_rollout": True,
        "stream": False,
        "metadata": request_metadata,
    }
    if isinstance(sample.prompt, str):
        payload["prompt"] = sample.prompt
    elif isinstance(sample.prompt, list):
        payload["messages"] = copy.deepcopy(sample.prompt)
    else:
        raise TypeError(f"MOSS-TTS Local prompt must be str or messages list, got {type(sample.prompt).__name__}.")
    return payload


def apply_moss_tts_local_response(sample: Sample, response: dict[str, Any]) -> Sample:
    if not isinstance(response, dict):
        raise TypeError(f"SGLang-Omni /generate response must be a dict, got {type(response).__name__}.")
    meta_info = response.get("meta_info")
    if not isinstance(meta_info, dict):
        raise ValueError("SGLang-Omni /generate response is missing meta_info.")
    trace = meta_info.get("omni_rollout")
    if not isinstance(trace, dict):
        raise ValueError("SGLang-Omni MOSS-TTS Local response is missing meta_info.omni_rollout.")
    trajectory = decode_moss_tts_local_trace(trace, meta_info=meta_info)

    audio = response.get("audio")
    artifacts = []
    if isinstance(audio, dict) and audio.get("data"):
        sample_rate = int(audio.get("sample_rate") or trajectory.model_identity.get("sample_rate") or 48000)
        artifacts.append(
            MediaArtifact.from_inline_audio(
                str(audio["data"]),
                mime_type=_audio_mime_type(audio.get("format")),
                sample_rate=sample_rate,
            )
        )
    elif trajectory.num_frames != 0:
        raise ValueError("SGLang-Omni MOSS-TTS Local response is missing inline audio data.")

    sample.structured_trajectory = trajectory
    sample.artifacts = artifacts
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["moss_weight_version"] = trajectory.weight_version
    sample.status = Sample.Status.COMPLETED if trajectory.finish_reason == "stop" else Sample.Status.TRUNCATED
    sample.response = str(response.get("text") or "")
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["omni_request_id"] = trajectory.request_id
    sample.metadata["audio_sha256"] = artifacts[0].sha256 if artifacts else None
    sample.metadata["frame_count"] = trajectory.num_frames
    sample.metadata["action_count"] = trajectory.num_actions
    return sample


class MossTTSLocalRolloutState(metaclass=SingletonMeta):
    def __init__(self, args: Namespace) -> None:
        key_env = getattr(args, "sglang_omni_admin_api_key_env", None)
        admin_key = os.getenv(key_env) if key_env else None
        endpoints = list(args.sglang_omni_endpoints)
        self.adapters = [SGLangOmniHttpAdapter(endpoint, admin_api_key=admin_key) for endpoint in endpoints]
        concurrency = max(1, int(args.sglang_server_concurrency) * len(self.adapters))
        self.semaphore = asyncio.Semaphore(concurrency)
        self._inflight = [0] * len(self.adapters)
        self._selection_lock = asyncio.Lock()
        self.teacher = None
        if getattr(args, "moss_local_objective", "wer_grpo") == "mopd":
            from miles.policies.moss_tts_local.mopd_client import LocalTeacherClient

            self.teacher = LocalTeacherClient(args)

    @asynccontextmanager
    async def lease(self):
        async with self.semaphore:
            async with self._selection_lock:
                index = min(range(len(self.adapters)), key=lambda candidate: self._inflight[candidate])
                self._inflight[index] += 1
            try:
                yield self.adapters[index]
            finally:
                async with self._selection_lock:
                    self._inflight[index] -= 1

    def close(self) -> None:
        if self.teacher is not None:
            run(self.teacher.close())
        for adapter in self.adapters:
            adapter.close()


def dispose_rollout_state() -> None:
    state = SingletonMeta._instances.pop(MossTTSLocalRolloutState, None)
    if state is not None:
        state.close()


async def generate_one(
    args: Namespace,
    sample: Sample,
    *,
    rollout_id: int,
    seed: int,
    adapter=None,
) -> Sample:
    if sample.status is not Sample.Status.PENDING:
        raise ValueError(f"MOSS-TTS Local rollout expects a pending sample, got {sample.status.value!r}.")
    request = build_moss_tts_local_request(args, sample, rollout_id=rollout_id, seed=seed)
    if adapter is not None:
        return apply_moss_tts_local_response(sample, await adapter.generate(request))
    state = MossTTSLocalRolloutState(args)
    async with state.lease() as leased_adapter:
        response = await leased_adapter.generate(request)
    return apply_moss_tts_local_response(sample, response)


async def _generate_and_reward(
    args: Namespace,
    sample: Sample,
    *,
    rollout_id: int,
    seed: int,
) -> Sample:
    sample = await generate_one(args, sample, rollout_id=rollout_id, seed=seed)
    state = MossTTSLocalRolloutState(args)
    if state.teacher is not None:
        return await state.teacher.score(sample, temperature=args.rollout_temperature)
    if not args.group_rm and sample.reward is None:
        sample.reward = await async_rm(args, sample)
    return sample


async def _generate_group(args: Namespace, group: list[Sample], *, rollout_id: int, group_index: int) -> list[Sample]:
    tasks = []
    for sample_index, sample in enumerate(group):
        seed = int(args.rollout_seed) + rollout_id * 1_000_003 + group_index * 10_007 + sample_index
        tasks.append(_generate_and_reward(args, sample, rollout_id=rollout_id, seed=seed))
    generated = await asyncio.gather(*tasks)
    if args.group_rm:
        rewards = await batched_async_rm(args, generated)
        if len(rewards) != len(generated):
            raise ValueError("Group reward implementation returned the wrong number of rewards.")
        for sample, reward in zip(generated, rewards, strict=True):
            sample.reward = reward
    return generated


def _wer_rollout_metrics(groups: list[list[Sample]]) -> dict[str, float]:
    scored = [
        sample for group in groups for sample in group if isinstance((sample.metadata or {}).get("wer"), (int, float))
    ]
    metrics: dict[str, float] = {}
    if scored:
        wers = [float(sample.metadata["wer"]) for sample in scored]
        metrics.update(
            {
                "moss_tts_local/wer_mean": sum(wers) / len(wers),
                "moss_tts_local/wer_min": min(wers),
                "moss_tts_local/wer_max": max(wers),
                "moss_tts_local/wer_exact_match_rate": sum(wer == 0.0 for wer in wers) / len(wers),
                "moss_tts_local/wer_scored_samples": float(len(wers)),
            }
        )
    similarity_scored = [
        sample
        for group in groups
        for sample in group
        if isinstance((sample.metadata or {}).get("sim_reward"), (int, float))
    ]
    if similarity_scored:
        similarities = [float(sample.metadata["sim_reward"]) for sample in similarity_scored]
        mixed_rewards = [float(sample.metadata["mixed_reward"]) for sample in similarity_scored]
        metrics.update(
            {
                "moss_tts_local/sim_mean": sum(similarities) / len(similarities),
                "moss_tts_local/sim_min": min(similarities),
                "moss_tts_local/sim_max": max(similarities),
                "moss_tts_local/mixed_reward_mean": sum(mixed_rewards) / len(mixed_rewards),
                "moss_tts_local/sim_scored_samples": float(len(similarities)),
            }
        )
    return metrics


async def generate_rollout_async(args: Namespace, rollout_id: int, data_source) -> RolloutFnTrainOutput:
    if getattr(args, "dynamic_sampling_filter_path", None) is not None:
        raise ValueError("MOSS-TTS Local P0 does not support dynamic oversampling/filtering.")
    groups = data_source.get_samples(args.rollout_batch_size)
    if len(groups) != args.rollout_batch_size:
        raise ValueError(f"Data source returned {len(groups)} groups; expected {args.rollout_batch_size}.")
    generated = await asyncio.gather(
        *(_generate_group(args, group, rollout_id=rollout_id, group_index=index) for index, group in enumerate(groups))
    )
    generated = sorted(generated, key=lambda group: group[0].index)
    metrics = {
        "moss_tts_local/frame_count": sum(
            sample.structured_trajectory.num_frames for group in generated for sample in group
        ),
        "moss_tts_local/action_count": sum(
            sample.structured_trajectory.num_actions for group in generated for sample in group
        ),
        "moss_tts_local/artifact_bytes": sum(
            artifact.num_bytes for group in generated for sample in group for artifact in sample.artifacts
        ),
    }
    metrics.update(_wer_rollout_metrics(generated))
    state = MossTTSLocalRolloutState(args)
    if state.teacher is not None:
        metrics.update(
            {
                "mopd/teacher_requests_total": state.teacher.requests,
                "mopd/teacher_samples_total": state.teacher.samples,
                "mopd/teacher_http_seconds_total": state.teacher.http_seconds,
            }
        )
    return RolloutFnTrainOutput(samples=generated, metrics=metrics)


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    if evaluation:
        raise NotImplementedError("MOSS-TTS Local evaluation rollout is not implemented in P0.")
    if not args.rollout_global_dataset:
        raise ValueError("MOSS-TTS Local P0 requires the global rollout dataset.")
    return run(generate_rollout_async(args, rollout_id, data_source))
