"""Validated audio artifacts shared by structured policies and reward clients."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class MediaArtifact:
    modality: Literal["audio"]
    mime_type: str
    sample_rate: int
    sha256: str
    num_bytes: int
    inline_base64: str | None = None
    uri: str | None = None

    def validate(self) -> None:
        if self.modality != "audio":
            raise ValueError(f"Unsupported media artifact modality {self.modality!r}.")
        if not self.mime_type.startswith("audio/"):
            raise ValueError(f"Audio artifact mime_type must start with 'audio/', got {self.mime_type!r}.")
        if self.sample_rate <= 0:
            raise ValueError(f"Audio artifact sample_rate must be positive, got {self.sample_rate}.")
        if (self.inline_base64 is None) == (self.uri is None):
            raise ValueError("MediaArtifact requires exactly one of inline_base64 or uri.")
        if self.num_bytes < 0:
            raise ValueError(f"Audio artifact num_bytes must be non-negative, got {self.num_bytes}.")
        if len(self.sha256) != 64:
            raise ValueError("Audio artifact sha256 must be a 64-character hex digest.")
        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise ValueError("Audio artifact sha256 must be hexadecimal.") from exc
        if self.inline_base64 is not None:
            try:
                payload = base64.b64decode(self.inline_base64, validate=True)
            except Exception as exc:
                raise ValueError("Audio artifact inline_base64 is invalid.") from exc
            if len(payload) != self.num_bytes:
                raise ValueError(
                    f"Audio artifact byte count mismatch: metadata={self.num_bytes}, decoded={len(payload)}."
                )
            digest = hashlib.sha256(payload).hexdigest()
            if digest != self.sha256:
                raise ValueError(f"Audio artifact checksum mismatch: expected {self.sha256}, got {digest}.")

    @classmethod
    def from_inline_audio(
        cls,
        inline_base64: str,
        *,
        mime_type: str,
        sample_rate: int,
    ) -> MediaArtifact:
        try:
            payload = base64.b64decode(inline_base64, validate=True)
        except Exception as exc:
            raise ValueError("Audio response is not valid base64.") from exc
        artifact = cls(
            modality="audio",
            mime_type=mime_type,
            sample_rate=int(sample_rate),
            inline_base64=inline_base64,
            uri=None,
            sha256=hashlib.sha256(payload).hexdigest(),
            num_bytes=len(payload),
        )
        artifact.validate()
        return artifact

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "modality": self.modality,
            "mime_type": self.mime_type,
            "sample_rate": self.sample_rate,
            "sha256": self.sha256,
            "num_bytes": self.num_bytes,
            "inline_base64": self.inline_base64,
            "uri": self.uri,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MediaArtifact:
        artifact = cls(
            modality=data["modality"],
            mime_type=str(data["mime_type"]),
            sample_rate=int(data["sample_rate"]),
            sha256=str(data["sha256"]),
            num_bytes=int(data["num_bytes"]),
            inline_base64=data.get("inline_base64"),
            uri=data.get("uri"),
        )
        artifact.validate()
        return artifact
