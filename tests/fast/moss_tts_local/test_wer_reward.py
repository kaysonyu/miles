from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest

from miles.policies.moss_tts_local import wer_reward
from miles.policies.moss_tts_local.types import MediaArtifact
from miles.policies.moss_tts_local.wer_reward import (
    ASRConfig,
    ASRTranscription,
    compute_word_error_rate,
    normalize_english_for_wer,
    parse_qwen3_asr_output,
)
from miles.utils.types import Sample

NUM_GPUS = 0


def test_english_wer_normalization_is_explicit_and_apostrophe_stable():
    assert normalize_english_for_wer("You aren’t POSITIVE—you’re negative.") == "you aren't positive you're negative"
    assert normalize_english_for_wer("Version １, test_case") == "version 1 test case"


def test_word_error_rate_reports_edit_breakdown_and_reward():
    score = compute_word_error_rate("one two three", "one four three now")

    assert score.substitutions == 1
    assert score.deletions == 0
    assert score.insertions == 1
    assert score.hits == 2
    assert score.wer == pytest.approx(2 / 3)
    assert score.reward == pytest.approx(1 / 3)
    assert score.exact_match is False


def test_word_error_rate_is_not_clipped():
    score = compute_word_error_rate("hello", "one two three")

    assert score.wer == pytest.approx(3.0)
    assert score.reward == pytest.approx(-2.0)


def test_word_error_rate_rejects_empty_normalized_reference():
    with pytest.raises(ValueError, match="reference is empty"):
        compute_word_error_rate("...", "hello")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("language English<asr_text>Hello world.", ("English", "Hello world.")),
        ("language None<asr_text>", ("", "")),
        ("plain transcript", ("", "plain transcript")),
        ([{"type": "text", "text": "language English<asr_text>Hi"}], ("English", "Hi")),
    ],
)
def test_qwen3_asr_output_parser(raw, expected):
    assert parse_qwen3_asr_output(raw) == expected


def test_chat_response_parser_keeps_finish_reason_and_completion_tokens():
    result = wer_reward._extract_response_content(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "language Finnish<asr_text>repeated text"},
                }
            ],
            "usage": {"completion_tokens": 512},
        }
    )

    assert result == ASRTranscription(
        language="Finnish",
        transcript="repeated text",
        finish_reason="length",
        completion_tokens=512,
    )


def test_asr_config_reads_named_key_without_evaluating_env_file(tmp_path, monkeypatch):
    secret = tmp_path / "reward-secret.env"
    secret.write_text("# comment\nexport INSPIRE_API_KEY='file-secret'\nIGNORED=$(false)\n", encoding="utf-8")
    monkeypatch.setenv("MOSS_TTS_WER_ASR_URL", "https://asr.example/v1/chat/completions")
    monkeypatch.setenv("MOSS_TTS_WER_ASR_API_KEY_FILE", str(secret))
    monkeypatch.delenv("INSPIRE_API_KEY", raising=False)

    config = ASRConfig.from_env()

    assert config.api_key == "file-secret"
    assert config.model == "qwen3-asr-1.7b"
    assert config.max_words_per_second == 12.0
    assert config.repeats == 1
    assert config.connect_timeout_seconds == 15.0


def test_asr_config_prefers_process_environment_over_key_file(tmp_path, monkeypatch):
    secret = tmp_path / "reward-secret.env"
    secret.write_text("INSPIRE_API_KEY=file-secret\n", encoding="utf-8")
    monkeypatch.setenv("MOSS_TTS_WER_ASR_URL", "https://asr.example/v1/audio/transcriptions")
    monkeypatch.setenv("MOSS_TTS_WER_ASR_API_KEY_FILE", str(secret))
    monkeypatch.setenv("INSPIRE_API_KEY", "environment-secret")

    assert ASRConfig.from_env().api_key == "environment-secret"


def test_repeated_asr_decodes_reuse_one_http_session(monkeypatch):
    artifact = MediaArtifact.from_inline_audio(
        base64.b64encode(b"wav-bytes").decode(),
        mime_type="audio/wav",
        sample_rate=48_000,
    )
    created_sessions = []
    used_sessions = []

    class FakeSession:
        def __init__(self, *, timeout, trust_env):
            assert timeout.connect == 7.0
            assert timeout.sock_connect == 7.0
            assert trust_env is False
            created_sessions.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    async def fake_transcribe_with_session(session, _artifact, *, config):
        assert _artifact is artifact
        used_sessions.append(session)
        return ASRTranscription(language="English", transcript="hello")

    monkeypatch.setattr(wer_reward.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(wer_reward, "_transcribe_with_session", fake_transcribe_with_session)
    config = ASRConfig(
        url="https://asr.example/v1/chat/completions",
        model="asr",
        api_key="secret",
        connect_timeout_seconds=7.0,
        repeats=3,
    )

    results = asyncio.run(wer_reward.transcribe_media_artifact_repeated(artifact, config=config))

    assert len(results) == 3
    assert len(created_sessions) == 1
    assert used_sessions == [created_sessions[0]] * 3


def test_reward_func_scores_audio_and_records_auditable_metadata(monkeypatch):
    artifact = MediaArtifact.from_inline_audio(
        base64.b64encode(b"wav-bytes").decode(),
        mime_type="audio/wav",
        sample_rate=48_000,
    )
    sample = Sample(label="The quick brown fox", artifacts=[artifact], metadata={})

    async def fake_transcribe(_artifact, *, config):
        assert _artifact is artifact
        assert config.max_words_per_second == 12.0
        return [ASRTranscription(language="English", transcript="the quick fox", finish_reason="stop")]

    monkeypatch.setattr(wer_reward, "transcribe_media_artifact_repeated", fake_transcribe)
    monkeypatch.setattr(
        wer_reward.ASRConfig,
        "from_env",
        classmethod(
            lambda cls: ASRConfig(url="https://asr.example/v1/chat/completions", model="asr", api_key="secret")
        ),
    )

    reward = asyncio.run(wer_reward.reward_func(SimpleNamespace(), sample))

    assert reward == pytest.approx(0.75)
    assert sample.metadata["wer"] == pytest.approx(0.25)
    assert sample.metadata["wer_reward"] == pytest.approx(0.75)
    assert sample.metadata["wer_details"]["deletions"] == 1
    assert sample.metadata["asr_transcript"] == "the quick fox"
    assert sample.metadata["wer_raw"] == pytest.approx(0.25)
    assert sample.metadata["wer_effective_reason"] == "raw_english_wer"
    assert sample.metadata["asr_repeat_count"] == 1
    assert sample.metadata["asr_repeat_disagreement"] is False


def test_reward_uses_median_effective_wer_across_repeated_asr_decodes(monkeypatch):
    artifact = MediaArtifact.from_inline_audio(
        base64.b64encode(b"wav-bytes").decode(),
        mime_type="audio/wav",
        sample_rate=48_000,
    )
    sample = Sample(label="one two three four", artifacts=[artifact], metadata={})
    transcriptions = [
        ASRTranscription(language="English", transcript="one two"),
        ASRTranscription(language="English", transcript="one two three"),
        ASRTranscription(language="English", transcript="wrong words here now"),
    ]

    async def fake_transcribe(_artifact, *, config):
        assert config.repeats == 3
        return transcriptions

    monkeypatch.setattr(wer_reward, "transcribe_media_artifact_repeated", fake_transcribe)
    monkeypatch.setattr(
        wer_reward.ASRConfig,
        "from_env",
        classmethod(
            lambda cls: ASRConfig(
                url="https://asr.example/v1/chat/completions",
                model="asr",
                api_key="secret",
                repeats=3,
            )
        ),
    )

    reward = asyncio.run(wer_reward.reward_func(SimpleNamespace(), sample))

    assert reward == pytest.approx(0.5)
    assert sample.metadata["asr_transcript"] == "one two"
    assert sample.metadata["asr_repeat_count"] == 3
    assert sample.metadata["asr_repeat_disagreement"] is True
    assert sample.metadata["asr_repeat_wer_spread"] == pytest.approx(0.75)
    assert [item["wer"] for item in sample.metadata["asr_repeats"]] == pytest.approx([0.5, 0.25, 1.0])


@pytest.mark.parametrize("repeats", [0, 2, -1])
def test_asr_config_rejects_non_positive_or_even_repeat_count(tmp_path, monkeypatch, repeats):
    secret = tmp_path / "reward-secret.env"
    secret.write_text("INSPIRE_API_KEY=file-secret\n", encoding="utf-8")
    monkeypatch.setenv("MOSS_TTS_WER_ASR_URL", "https://asr.example/v1/chat/completions")
    monkeypatch.setenv("MOSS_TTS_WER_ASR_API_KEY_FILE", str(secret))
    monkeypatch.setenv("MOSS_TTS_WER_ASR_REPEATS", str(repeats))
    monkeypatch.delenv("INSPIRE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="positive odd integer"):
        ASRConfig.from_env()


def test_reward_maps_language_mismatch_to_stable_full_error_and_keeps_raw_wer(monkeypatch):
    artifact = MediaArtifact.from_inline_audio(
        base64.b64encode(b"wav-bytes").decode(),
        mime_type="audio/wav",
        sample_rate=48_000,
    )
    sample = Sample(label="one two", artifacts=[artifact], metadata={})

    async def fake_transcribe(_artifact, *, config):
        return [ASRTranscription(language="Finnish", transcript="one two three four five", finish_reason="length")]

    monkeypatch.setattr(wer_reward, "transcribe_media_artifact_repeated", fake_transcribe)
    monkeypatch.setattr(
        wer_reward.ASRConfig,
        "from_env",
        classmethod(
            lambda cls: ASRConfig(url="https://asr.example/v1/chat/completions", model="asr", api_key="secret")
        ),
    )

    reward = asyncio.run(wer_reward.reward_func(SimpleNamespace(), sample))

    assert reward == 0.0
    assert sample.metadata["wer"] == 1.0
    assert sample.metadata["wer_raw"] == 1.5
    assert sample.metadata["wer_effective_reason"] == "language_mismatch"
    assert sample.metadata["asr_language_mismatch"] is True
    assert sample.metadata["asr_quality_flags"] == ["decoder_length_truncated"]
    assert sample.metadata["wer_details"]["deletions"] == 2
    assert sample.metadata["wer_raw_details"]["insertions"] == 3


def test_reward_maps_length_truncated_english_asr_to_stable_full_error(monkeypatch):
    artifact = MediaArtifact.from_inline_audio(
        base64.b64encode(b"wav-bytes").decode(),
        mime_type="audio/wav",
        sample_rate=48_000,
    )
    sample = Sample(label="one two", artifacts=[artifact], metadata={})

    async def fake_transcribe(_artifact, *, config):
        return [
            ASRTranscription(
                language="English",
                transcript=" ".join(["word"] * 100),
                finish_reason="length",
                completion_tokens=512,
            )
        ]

    monkeypatch.setattr(wer_reward, "transcribe_media_artifact_repeated", fake_transcribe)
    monkeypatch.setattr(
        wer_reward.ASRConfig,
        "from_env",
        classmethod(
            lambda cls: ASRConfig(url="https://asr.example/v1/chat/completions", model="asr", api_key="secret")
        ),
    )

    reward = asyncio.run(wer_reward.reward_func(SimpleNamespace(), sample))

    assert reward == 0.0
    assert sample.metadata["wer"] == 1.0
    assert sample.metadata["wer_raw"] == 50.0
    assert sample.metadata["wer_effective_reason"] == "asr_quality_failure"
    assert sample.metadata["asr_finish_reason"] == "length"
    assert sample.metadata["asr_completion_tokens"] == 512
