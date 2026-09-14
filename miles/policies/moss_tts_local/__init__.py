"""MOSS-TTS Local v1.5 structured policy implementation."""

from miles.policies.moss_tts_local.spec import MOSS_TTS_LOCAL_SPEC, MossTTSLocalPolicySpec
from miles.policies.moss_tts_local.types import MediaArtifact, MossTTSLocalTrajectoryV2

__all__ = [
    "MOSS_TTS_LOCAL_SPEC",
    "MediaArtifact",
    "MossTTSLocalPolicySpec",
    "MossTTSLocalTrajectoryV2",
]
