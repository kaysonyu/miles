#!/usr/bin/env python3
"""Compare paired pre/post WER evaluations at the prompt level."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_samples(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    samples: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            key = (str(sample["source_id"]), int(sample["seed"]))
            if key in samples:
                raise ValueError(f"Duplicate evaluation key {key} in {path}:{line_number}.")
            samples[key] = sample
    if not samples:
        raise ValueError(f"No samples found in {path}.")
    return samples


def _prompt_mean_wers(samples: dict[tuple[str, int], dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for (source_id, _seed), sample in samples.items():
        values[source_id].append(float(sample["wer"]))
    return {source_id: statistics.fmean(wers) for source_id, wers in values.items()}


def _bootstrap_mean_ci(values: list[float], *, iterations: int, seed: int) -> tuple[float, float]:
    generator = random.Random(seed)
    sample_count = len(values)
    means = [
        statistics.fmean(values[generator.randrange(sample_count)] for _ in range(sample_count))
        for _ in range(iterations)
    ]
    means.sort()
    lower_index = int(0.025 * (iterations - 1))
    upper_index = int(0.975 * (iterations - 1))
    return means[lower_index], means[upper_index]


def _error_summary(samples: dict[tuple[str, int], dict[str, Any]]) -> dict[str, int | float]:
    reference_words = sum(int(sample["reference_word_count"]) for sample in samples.values())
    if reference_words <= 0:
        raise ValueError("Evaluation samples must contain at least one reference word.")
    substitutions = sum(int(sample["substitutions"]) for sample in samples.values())
    deletions = sum(int(sample["deletions"]) for sample in samples.values())
    insertions = sum(int(sample["insertions"]) for sample in samples.values())
    errors = substitutions + deletions + insertions
    return {
        "reference_words": reference_words,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "corpus_wer": errors / reference_words,
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    baseline = _read_samples(args.baseline)
    post = _read_samples(args.post)
    if set(baseline) != set(post):
        missing_post = sorted(set(baseline) - set(post))
        missing_baseline = sorted(set(post) - set(baseline))
        raise ValueError(
            f"Pre/post evaluation keys differ: missing_post={missing_post}, missing_baseline={missing_baseline}."
        )
    for key in baseline:
        if baseline[key]["reference_normalized"] != post[key]["reference_normalized"]:
            raise ValueError(f"Reference changed for paired sample {key}.")

    baseline_wers = [float(baseline[key]["wer"]) for key in sorted(baseline)]
    post_wers = [float(post[key]["wer"]) for key in sorted(post)]
    baseline_raw_wers = [float(baseline[key].get("wer_raw", baseline[key]["wer"])) for key in sorted(baseline)]
    post_raw_wers = [float(post[key].get("wer_raw", post[key]["wer"])) for key in sorted(post)]
    same_audio_sha_count = sum(baseline[key].get("audio_sha256") == post[key].get("audio_sha256") for key in baseline)
    same_hypothesis_count = sum(
        baseline[key].get("hypothesis_normalized") == post[key].get("hypothesis_normalized") for key in baseline
    )
    same_raw_hypothesis_count = sum(
        baseline[key].get("hypothesis") is not None and baseline[key].get("hypothesis") == post[key].get("hypothesis")
        for key in baseline
    )
    same_wer_count = sum(float(baseline[key]["wer"]) == float(post[key]["wer"]) for key in baseline)
    baseline_prompt = _prompt_mean_wers(baseline)
    post_prompt = _prompt_mean_wers(post)
    prompt_ids = sorted(baseline_prompt)
    prompt_deltas = [post_prompt[prompt_id] - baseline_prompt[prompt_id] for prompt_id in prompt_ids]
    ci_low, ci_high = _bootstrap_mean_ci(
        prompt_deltas,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )

    baseline_mean = statistics.fmean(baseline_wers)
    post_mean = statistics.fmean(post_wers)
    mean_delta = post_mean - baseline_mean
    baseline_errors = _error_summary(baseline)
    post_errors = _error_summary(post)
    if baseline_errors["reference_words"] != post_errors["reference_words"]:
        raise ValueError("Pre/post total reference word counts differ.")
    if ci_high < 0:
        verdict = "improved"
    elif ci_low > 0:
        verdict = "regressed"
    else:
        verdict = "inconclusive"

    per_prompt = [
        {
            "source_id": prompt_id,
            "baseline_mean_wer": baseline_prompt[prompt_id],
            "post_mean_wer": post_prompt[prompt_id],
            "wer_delta": post_prompt[prompt_id] - baseline_prompt[prompt_id],
        }
        for prompt_id in prompt_ids
    ]
    return {
        "schema_version": 2,
        "paired_unit": "prompt (WER averaged across fixed seeds before bootstrap)",
        "num_prompts": len(prompt_ids),
        "num_samples": len(baseline),
        "same_audio_sha_count": same_audio_sha_count,
        "same_audio_sha_rate": same_audio_sha_count / len(baseline),
        "same_hypothesis_count": same_hypothesis_count,
        "same_hypothesis_rate": same_hypothesis_count / len(baseline),
        "same_raw_hypothesis_count": same_raw_hypothesis_count,
        "same_raw_hypothesis_rate": same_raw_hypothesis_count / len(baseline),
        "same_wer_count": same_wer_count,
        "same_wer_rate": same_wer_count / len(baseline),
        "baseline_mean_wer": baseline_mean,
        "post_mean_wer": post_mean,
        "mean_wer_delta": mean_delta,
        "relative_wer_change": mean_delta / baseline_mean if baseline_mean else None,
        "baseline_mean_raw_wer": statistics.fmean(baseline_raw_wers),
        "post_mean_raw_wer": statistics.fmean(post_raw_wers),
        "mean_raw_wer_delta": statistics.fmean(post_raw_wers) - statistics.fmean(baseline_raw_wers),
        "baseline_corpus_wer": baseline_errors["corpus_wer"],
        "post_corpus_wer": post_errors["corpus_wer"],
        "corpus_wer_delta": post_errors["corpus_wer"] - baseline_errors["corpus_wer"],
        "baseline_error_counts": baseline_errors,
        "post_error_counts": post_errors,
        "total_error_delta": post_errors["errors"] - baseline_errors["errors"],
        "baseline_language_mismatch_count": sum(
            bool(sample.get("asr_language_mismatch")) for sample in baseline.values()
        ),
        "post_language_mismatch_count": sum(bool(sample.get("asr_language_mismatch")) for sample in post.values()),
        "baseline_asr_quality_failure_count": sum(
            bool(sample.get("asr_quality_flags")) for sample in baseline.values()
        ),
        "post_asr_quality_failure_count": sum(bool(sample.get("asr_quality_flags")) for sample in post.values()),
        "baseline_exact_match_rate": statistics.fmean(float(sample["exact_match"]) for sample in baseline.values()),
        "post_exact_match_rate": statistics.fmean(float(sample["exact_match"]) for sample in post.values()),
        "prompt_level_delta_median": statistics.median(prompt_deltas),
        "prompt_level_delta_bootstrap_95ci": [ci_low, ci_high],
        "prompts_improved": sum(delta < 0 for delta in prompt_deltas),
        "prompts_unchanged": sum(delta == 0 for delta in prompt_deltas),
        "prompts_regressed": sum(delta > 0 for delta in prompt_deltas),
        "verdict": verdict,
        "verdict_rule": "improved/regressed only when the prompt-bootstrap 95% CI excludes zero",
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
        "per_prompt": per_prompt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1_000:
        parser.error("--bootstrap-iterations must be at least 1000.")
    return args


def main() -> None:
    args = parse_args()
    for path in (args.baseline, args.post):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_prompt"}, sort_keys=True))


if __name__ == "__main__":
    main()
