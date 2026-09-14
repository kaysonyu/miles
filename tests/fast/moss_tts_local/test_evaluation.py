from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

NUM_GPUS = 0


def _module():
    path = Path(__file__).parents[3] / "tools" / "moss_tts_local" / "evaluate_wer.py"
    spec = importlib.util.spec_from_file_location("evaluate_wer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluation_reports_progress_and_keeps_canonical_order(tmp_path, monkeypatch, capsys):
    module = _module()
    data_path = tmp_path / "eval.jsonl"
    data_path.write_text(
        "".join(
            json.dumps({"text": f"prompt {index}", "label": f"label {index}", "metadata": {}}) + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )

    class FakeAdapter:
        def __init__(self, *_args, **_kwargs):
            self.closed = False

        def close(self):
            self.closed = True

    async def fake_warm_up(**_kwargs):
        return None

    async def fake_evaluate_one(**kwargs):
        # Complete in a different order from the final canonical row/seed sort.
        delay_rank = kwargs["row_index"] * 2 + kwargs["seed_index"]
        await asyncio.sleep(0.001 * (4 - delay_rank))
        return {"row_index": kwargs["row_index"], "seed": kwargs["seed"]}

    monkeypatch.setattr(module, "SGLangOmniHttpAdapter", FakeAdapter)
    monkeypatch.setattr(module, "_warm_up", fake_warm_up)
    monkeypatch.setattr(module, "_evaluate_one", fake_evaluate_one)
    monkeypatch.setattr(module, "_summary", lambda results, _args: {"num_samples": len(results)})

    args = argparse.Namespace(
        omni_endpoint="http://omni.invalid",
        generate_timeout_seconds=1.0,
        concurrency=2,
        temperature=1.0,
        max_response_len=128,
        seed=[11, 22],
        rollout_id=7,
        output_dir=tmp_path / "output",
        save_audio=False,
        warmup_count=0,
        warmup_seed=1,
        progress_interval=2,
        data=data_path,
        limit=None,
    )
    results, summary = asyncio.run(module.run_evaluation(args))

    assert [(result["row_index"], result["seed"]) for result in results] == [
        (0, 11),
        (0, 22),
        (1, 11),
        (1, 22),
    ]
    assert summary == {"num_samples": 4}
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events == [
        {"concurrency": 2, "event": "wer_eval_started", "num_samples": 4, "rollout_id": 7},
        {"completed": 2, "event": "wer_eval_progress", "rollout_id": 7, "total": 4},
        {"completed": 4, "event": "wer_eval_progress", "rollout_id": 7, "total": 4},
    ]


def test_summary_reports_effective_and_raw_wer_separately(tmp_path):
    module = _module()
    data_path = tmp_path / "eval.jsonl"
    data_path.write_text("{}\n", encoding="utf-8")
    common = {
        "reward": 0.0,
        "reference_word_count": 2,
        "exact_match": False,
        "audio_duration_seconds": 1.0,
        "generation_seconds": 0.2,
        "asr_seconds": 0.1,
        "weight_version": "1",
    }
    results = [
        {
            **common,
            "source_id": "a",
            "wer": 1.0,
            "wer_raw": 10.0,
            "errors": 2,
            "raw_errors": 20,
            "asr_language_mismatch": True,
            "asr_quality_flags": ["decoder_length_truncated"],
            "asr_finish_reason": "length",
            "asr_repeat_count": 3,
            "asr_repeat_disagreement": True,
        },
        {
            **common,
            "source_id": "b",
            "wer": 0.5,
            "wer_raw": 0.5,
            "errors": 1,
            "raw_errors": 1,
            "asr_language_mismatch": False,
            "asr_quality_flags": [],
            "asr_finish_reason": "stop",
            "asr_repeat_count": 3,
            "asr_repeat_disagreement": False,
        },
    ]
    args = argparse.Namespace(
        data=data_path,
        seed=[1],
        temperature=1.0,
        max_response_len=128,
        warmup_count=0,
        warmup_seed=2,
    )

    summary = module._summary(results, args)

    assert summary["schema_version"] == 2
    assert summary["mean_wer"] == pytest.approx(0.75)
    assert summary["mean_raw_wer"] == pytest.approx(5.25)
    assert summary["corpus_wer"] == pytest.approx(0.75)
    assert summary["raw_corpus_wer"] == pytest.approx(5.25)
    assert summary["language_mismatch_count"] == 1
    assert summary["asr_quality_failure_count"] == 1
    assert summary["asr_length_truncated_count"] == 1
    assert summary["asr_repeat_counts"] == [3]
    assert summary["asr_repeat_disagreement_count"] == 1


def test_adapter_pool_spreads_nested_leases_and_closes_every_endpoint(monkeypatch):
    module = _module()
    adapters = []

    class FakeAdapter:
        def __init__(self, endpoint, **_kwargs):
            self.endpoint = endpoint
            self.closed = False
            adapters.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "SGLangOmniHttpAdapter", FakeAdapter)
    pool = module._AdapterPool(
        ["http://engine-0:1", "http://engine-1:1", "http://engine-2:1"],
        generate_timeout_seconds=1.0,
    )

    async def lease_indices():
        async with pool.lease() as (first, _):
            async with pool.lease() as (second, _):
                async with pool.lease() as (third, _):
                    return first, second, third

    assert asyncio.run(lease_indices()) == (0, 1, 2)
    pool.close()
    assert all(adapter.closed for adapter in adapters)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
