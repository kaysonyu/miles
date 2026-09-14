#!/usr/bin/env python3
"""Run a fixed-seed MOSS-TTS Local WER evaluation against an Omni server."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import os
import statistics
import time
import wave
from argparse import Namespace
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from miles.backends.sglang_omni_utils.http_adapter import SGLangOmniHttpAdapter
from miles.policies.moss_tts_local.rollout import generate_one
from miles.policies.moss_tts_local.sim_wer_reward import reward_func as sim_wer_reward_func
from miles.policies.moss_tts_local.wer_reward import reward_func as wer_reward_func
from miles.utils.types import Sample


class _AdapterPool:
    """Lease the least-loaded Omni endpoint without exposing endpoint URLs."""

    def __init__(self, endpoints: list[str], *, generate_timeout_seconds: float) -> None:
        if not endpoints:
            raise ValueError("At least one Omni endpoint is required.")
        self.adapters = [
            SGLangOmniHttpAdapter(endpoint, generate_timeout_s=generate_timeout_seconds) for endpoint in endpoints
        ]
        self.inflight = [0] * len(self.adapters)
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def lease(self):
        async with self._lock:
            index = min(range(len(self.adapters)), key=lambda candidate: self.inflight[candidate])
            self.inflight[index] += 1
        try:
            yield index, self.adapters[index]
        finally:
            async with self._lock:
                self.inflight[index] -= 1

    def close(self) -> None:
        for adapter in self.adapters:
            adapter.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object.")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No evaluation rows found in {path}.")
    return rows


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _wav_duration_seconds(payload: bytes) -> float:
    with wave.open(io.BytesIO(payload), "rb") as reader:
        return reader.getnframes() / reader.getframerate()


def _tensor_sha256(tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _summary(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    configured_endpoints = getattr(args, "omni_endpoint", ["<legacy-single-endpoint>"])
    if isinstance(configured_endpoints, str):
        configured_endpoints = [configured_endpoints]
    wers = [float(result["wer"]) for result in results]
    raw_wers = [float(result.get("wer_raw", result["wer"])) for result in results]
    rewards = [float(result["reward"]) for result in results]
    asr_seconds = [float(result["asr_seconds"]) for result in results if result.get("asr_seconds") is not None]
    total_errors = sum(int(result["errors"]) for result in results)
    raw_total_errors = sum(int(result.get("raw_errors", result["errors"])) for result in results)
    total_reference_words = sum(int(result["reference_word_count"]) for result in results)
    mixed_reward = any("sim_reward" in result for result in results)
    summary = {
        "schema_version": 2,
        "task": "moss_tts_local_english_wer_sim" if mixed_reward else "moss_tts_local_english_wer",
        "dataset": str(args.data),
        "dataset_sha256": _sha256(args.data),
        "num_prompts": len({result["source_id"] for result in results}),
        "num_samples": len(results),
        "seeds": args.seed,
        "temperature": args.temperature,
        "max_response_len": args.max_response_len,
        "warmup_count": args.warmup_count,
        "warmup_seed": args.warmup_seed,
        "reference_audio_used": mixed_reward,
        "wer_definition": (
            "English NFKC/casefold word Levenshtein; explicit language mismatch or ASR quality failure "
            "maps to effective WER=1; reward=1-effective WER; raw WER retained"
        ),
        "mean_wer": statistics.fmean(wers),
        "mean_raw_wer": statistics.fmean(raw_wers),
        "median_wer": statistics.median(wers),
        "median_raw_wer": statistics.median(raw_wers),
        "p90_wer": _percentile(wers, 0.9),
        "p90_raw_wer": _percentile(raw_wers, 0.9),
        "min_wer": min(wers),
        "max_wer": max(wers),
        "min_raw_wer": min(raw_wers),
        "max_raw_wer": max(raw_wers),
        "corpus_wer": total_errors / total_reference_words,
        "raw_corpus_wer": raw_total_errors / total_reference_words,
        "mean_reward": statistics.fmean(rewards),
        "exact_match_rate": statistics.fmean(float(result["exact_match"]) for result in results),
        "total_errors": total_errors,
        "raw_total_errors": raw_total_errors,
        "total_reference_words": total_reference_words,
        "language_mismatch_count": sum(bool(result.get("asr_language_mismatch")) for result in results),
        "asr_quality_failure_count": sum(bool(result.get("asr_quality_flags")) for result in results),
        "asr_repeat_counts": sorted({int(result.get("asr_repeat_count", 1)) for result in results}),
        "asr_repeat_disagreement_count": sum(bool(result.get("asr_repeat_disagreement")) for result in results),
        "asr_length_truncated_count": sum(
            result.get("asr_finish_reason", "").casefold() == "length" for result in results
        ),
        "mean_audio_duration_seconds": statistics.fmean(float(result["audio_duration_seconds"]) for result in results),
        "mean_generation_seconds": statistics.fmean(float(result["generation_seconds"]) for result in results),
        "mean_asr_seconds": statistics.fmean(asr_seconds) if asr_seconds else None,
        "weight_versions": sorted({str(result["weight_version"]) for result in results}),
        "omni_endpoint_count": len(configured_endpoints),
        "omni_endpoint_sample_counts": {
            str(index): sum(result.get("omni_endpoint_index") == index for result in results)
            for index in range(len(configured_endpoints))
        },
    }
    if mixed_reward:
        sim_rewards = [float(result["sim_reward"]) for result in results]
        raw_cosines = [float(result["sim_raw_cosine"]) for result in results]
        bounded_wer_rewards = [float(result["wer_reward_bounded"]) for result in results]
        summary.update(
            {
                "schema_version": 3,
                "mixed_reward_definition": "0.6*clip(SIM cosine,0,1) + 0.4*clip(1-effective WER,0,1)",
                "sim_model": "wavlm_large_ecapa_tdnn",
                "sim_weight": 0.6,
                "wer_weight": 0.4,
                "mean_sim_reward": statistics.fmean(sim_rewards),
                "median_sim_reward": statistics.median(sim_rewards),
                "p10_sim_reward": _percentile(sim_rewards, 0.1),
                "min_sim_reward": min(sim_rewards),
                "max_sim_reward": max(sim_rewards),
                "mean_sim_raw_cosine": statistics.fmean(raw_cosines),
                "mean_wer_reward_bounded": statistics.fmean(bounded_wer_rewards),
                "sim_reference_cache_hit_rate": statistics.fmean(
                    float(result["sim_reference_cache_hit"]) for result in results
                ),
                "sim_item_error_count": sum(result.get("sim_item_error_code") is not None for result in results),
                "mean_reward_seconds": statistics.fmean(float(result["reward_seconds"]) for result in results),
            }
        )
    return summary


async def _evaluate_one(
    *,
    adapter_pool: _AdapterPool,
    semaphore: asyncio.Semaphore,
    rollout_args: Namespace,
    row: dict[str, Any],
    row_index: int,
    seed: int,
    seed_index: int,
    output_dir: Path,
    save_audio: bool,
) -> dict[str, Any]:
    prompt = row.get("text")
    label = row.get("label")
    metadata = row.get("metadata") or {}
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Evaluation row {row_index} has no non-empty text field.")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"Evaluation row {row_index} has no non-empty label field.")
    if not isinstance(metadata, dict):
        raise ValueError(f"Evaluation row {row_index} metadata must be an object.")

    sample_index = row_index * len(rollout_args.eval_seeds) + seed_index
    sample = Sample(
        group_index=row_index,
        index=sample_index,
        prompt=prompt,
        label=label,
        metadata=dict(metadata),
    )
    async with semaphore:
        async with adapter_pool.lease() as (endpoint_index, adapter):
            generate_started = time.monotonic()
            sample = await generate_one(
                rollout_args,
                sample,
                rollout_id=rollout_args.eval_rollout_id,
                seed=seed,
                adapter=adapter,
            )
            generation_seconds = time.monotonic() - generate_started
        reward_started = time.monotonic()
        reward_function = (
            sim_wer_reward_func if getattr(rollout_args, "reward_mode", "wer") == "sim_wer" else wer_reward_func
        )
        reward = await reward_function(rollout_args, sample)
        reward_seconds = time.monotonic() - reward_started

    artifact = sample.artifacts[0]
    assert artifact.inline_base64 is not None
    audio_bytes = base64.b64decode(artifact.inline_base64, validate=True)
    source_id = str(metadata.get("source_id") or f"row-{row_index:04d}")
    reference_audio_path = metadata.get("reference_audio_path")
    reference_audio_identity = (
        hashlib.sha256(os.fsencode(reference_audio_path)).hexdigest()
        if isinstance(reference_audio_path, str)
        else None
    )
    audio_relative_path = None
    if save_audio:
        audio_relative_path = f"audio/{row_index:04d}-seed-{seed}.wav"
        audio_path = output_dir / audio_relative_path
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(audio_bytes)

    details = sample.metadata["wer_details"]
    raw_details = sample.metadata.get("wer_raw_details", details)
    trajectory = sample.structured_trajectory
    assert trajectory is not None
    mixed_reward_mode = getattr(rollout_args, "reward_mode", "wer") == "sim_wer"
    result = {
        "source_id": source_id,
        "row_index": row_index,
        "sample_index": sample_index,
        "seed": seed,
        "reference": label,
        "hypothesis": sample.metadata["asr_transcript"],
        "asr_language": sample.metadata["asr_language"],
        "asr_finish_reason": sample.metadata.get("asr_finish_reason", ""),
        "asr_completion_tokens": sample.metadata.get("asr_completion_tokens"),
        "asr_raw_word_count": sample.metadata.get("asr_raw_word_count"),
        "asr_words_per_second": sample.metadata.get("asr_words_per_second"),
        "asr_quality_flags": sample.metadata.get("asr_quality_flags", []),
        "asr_language_mismatch": sample.metadata.get("asr_language_mismatch", False),
        "asr_repeat_count": sample.metadata.get("asr_repeat_count", 1),
        "asr_repeat_disagreement": sample.metadata.get("asr_repeat_disagreement", False),
        "asr_repeat_wer_spread": sample.metadata.get("asr_repeat_wer_spread", 0.0),
        "asr_repeats": sample.metadata.get("asr_repeats", []),
        "reference_normalized": details["reference_normalized"],
        "hypothesis_normalized": details["hypothesis_normalized"],
        "reference_word_count": details["reference_word_count"],
        "hypothesis_word_count": details["hypothesis_word_count"],
        "substitutions": details["substitutions"],
        "deletions": details["deletions"],
        "insertions": details["insertions"],
        "hits": details["hits"],
        "errors": details["errors"],
        "wer": details["wer"],
        "wer_raw": raw_details["wer"],
        "wer_effective_reason": sample.metadata.get("wer_effective_reason", "raw_english_wer"),
        "raw_substitutions": raw_details["substitutions"],
        "raw_deletions": raw_details["deletions"],
        "raw_insertions": raw_details["insertions"],
        "raw_errors": raw_details["errors"],
        "reward": reward,
        "exact_match": details["exact_match"],
        "finish_reason": trajectory.finish_reason,
        "frame_count": trajectory.num_frames,
        "action_count": trajectory.num_actions,
        "weight_version": trajectory.weight_version,
        "prompt_rows_sha256": _tensor_sha256(trajectory.prompt_rows),
        "decisions_sha256": _tensor_sha256(trajectory.decisions),
        "codes_sha256": _tensor_sha256(trajectory.codes),
        "audio_sha256": artifact.sha256,
        "audio_bytes": artifact.num_bytes,
        "audio_duration_seconds": _wav_duration_seconds(audio_bytes),
        "audio_path": audio_relative_path,
        "generation_seconds": generation_seconds,
        "omni_endpoint_index": endpoint_index,
        "asr_seconds": None if mixed_reward_mode else reward_seconds,
        "reward_seconds": reward_seconds,
        "reference_audio_identity": reference_audio_identity,
    }
    if mixed_reward_mode:
        result.update(
            {
                "sim_model": sample.metadata["sim_model"],
                "sim_raw_cosine": sample.metadata["sim_raw_cosine"],
                "sim_reward": sample.metadata["sim_reward"],
                "sim_reference_cache_hit": sample.metadata["sim_reference_cache_hit"],
                "sim_reference_bucket_seconds": sample.metadata["sim_reference_bucket_seconds"],
                "sim_candidate_bucket_seconds": sample.metadata["sim_candidate_bucket_seconds"],
                "sim_service_elapsed_ms": sample.metadata["sim_service_elapsed_ms"],
                "sim_candidate_sha256": sample.metadata["sim_candidate_sha256"],
                "sim_item_error_code": sample.metadata["sim_item_error_code"],
                "wer_reward_unbounded": sample.metadata["wer_reward_unbounded"],
                "wer_reward_bounded": sample.metadata["wer_reward_bounded"],
                "mixed_reward_formula": sample.metadata["mixed_reward_formula"],
            }
        )
    return result


async def _warm_up(
    *,
    adapter_pool: _AdapterPool,
    semaphore: asyncio.Semaphore,
    rollout_args: Namespace,
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
) -> None:
    async def generate_warmup(index: int) -> None:
        row = rows[index % len(rows)]
        prompt = row.get("text")
        metadata = row.get("metadata") or {}
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(metadata, dict):
            raise ValueError(f"Warmup row {index % len(rows)} has invalid text/metadata.")
        sample = Sample(
            group_index=-(index + 1),
            index=-(index + 1),
            prompt=prompt,
            label=row.get("label"),
            metadata=dict(metadata),
        )
        async with semaphore:
            async with adapter_pool.lease() as (_, adapter):
                await generate_one(
                    rollout_args,
                    sample,
                    rollout_id=-1,
                    seed=seed + index,
                    adapter=adapter,
                )

    if count:
        await asyncio.gather(*(generate_warmup(index) for index in range(count)))


async def run_evaluation(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl(args.data, limit=args.limit)
    endpoints = [args.omni_endpoint] if isinstance(args.omni_endpoint, str) else list(args.omni_endpoint)
    adapter_pool = _AdapterPool(
        endpoints,
        generate_timeout_seconds=args.generate_timeout_seconds,
    )
    args.omni_endpoint = endpoints
    semaphore = asyncio.Semaphore(args.concurrency)
    rollout_args = Namespace(
        rollout_temperature=args.temperature,
        rollout_max_response_len=args.max_response_len,
        sglang_omni_train_stage="tts_engine",
        eval_seeds=args.seed,
        eval_rollout_id=args.rollout_id,
        reward_mode=getattr(args, "reward_mode", "wer"),
    )
    try:
        await _warm_up(
            adapter_pool=adapter_pool,
            semaphore=semaphore,
            rollout_args=rollout_args,
            rows=rows,
            count=args.warmup_count,
            seed=args.warmup_seed,
        )
        tasks = [
            _evaluate_one(
                adapter_pool=adapter_pool,
                semaphore=semaphore,
                rollout_args=rollout_args,
                row=row,
                row_index=row_index,
                seed=seed,
                seed_index=seed_index,
                output_dir=args.output_dir,
                save_audio=args.save_audio,
            )
            for row_index, row in enumerate(rows)
            for seed_index, seed in enumerate(args.seed)
        ]
        total = len(tasks)
        print(
            json.dumps(
                {
                    "event": "wer_eval_started",
                    "num_samples": total,
                    "concurrency": args.concurrency,
                    "rollout_id": args.rollout_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        results = []
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            results.append(await task)
            if completed % args.progress_interval == 0 or completed == total:
                print(
                    json.dumps(
                        {
                            "event": "wer_eval_progress",
                            "completed": completed,
                            "total": total,
                            "rollout_id": args.rollout_id,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        adapter_pool.close()
    results.sort(key=lambda result: (result["row_index"], result["seed"]))
    return results, _summary(results, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--omni-endpoint",
        action="append",
        required=True,
        help="Top-level Omni endpoint. Repeat to load-balance evaluation across replicas.",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True, help="Repeat for paired fixed seeds.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-response-len", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--generate-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--rollout-id", type=int, default=0)
    parser.add_argument("--warmup-count", type=int, default=0)
    parser.add_argument("--warmup-seed", type=int, default=2_026_083_199)
    parser.add_argument("--progress-interval", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-audio", action="store_true")
    parser.add_argument(
        "--reward-mode",
        choices=("wer", "sim_wer"),
        default="wer",
        help="Score WER only or the fixed 0.6 SIM + 0.4 bounded-WER reward.",
    )
    args = parser.parse_args()
    if args.temperature <= 0:
        parser.error("--temperature must be positive.")
    if args.max_response_len <= 0 or args.concurrency <= 0 or args.progress_interval <= 0:
        parser.error("--max-response-len, --concurrency, and --progress-interval must be positive.")
    if args.warmup_count < 0 or args.warmup_seed < 0:
        parser.error("--warmup-count and --warmup-seed must be non-negative.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive when provided.")
    if len(set(args.seed)) != len(args.seed):
        parser.error("--seed values must be unique.")
    if len(set(args.omni_endpoint)) != len(args.omni_endpoint):
        parser.error("--omni-endpoint values must be unique.")
    return args


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, summary = asyncio.run(run_evaluation(args))
    samples_path = args.output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "SUCCESS").touch()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
