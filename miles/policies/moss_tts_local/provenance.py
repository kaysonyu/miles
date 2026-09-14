"""Separate trajectory behavior from the student version used to score replay."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SampleProvenance:
    origin: Literal["student", "teacher_replay"]
    behavior_version: str
    student_scoring_version: str | None
    teacher_digest: str | None
    replay_id: str

    @classmethod
    def from_sample(cls, sample) -> SampleProvenance:
        metadata = sample.metadata or {}
        replay = metadata.get("moss_teacher_replay")
        scores = metadata.get("mopd_scores") or {}
        scoring_version = metadata.get("moss_training_policy_version")
        return cls(
            origin="teacher_replay" if replay else "student",
            behavior_version=str(sample.structured_trajectory.weight_version),
            student_scoring_version=str(scoring_version) if scoring_version is not None else None,
            teacher_digest=scores.get("teacher_weight_sha256"),
            replay_id=replay["entry_id"] if replay and "entry_id" in replay else "",
        )

    @property
    def training_version(self) -> str:
        if self.origin == "student":
            return self.behavior_version
        if self.student_scoring_version is None:
            raise ValueError("Teacher replay requires the current student's scoring version")
        return self.student_scoring_version


def validate_sample_version(args, sample) -> None:
    provenance = SampleProvenance.from_sample(sample)
    if provenance.origin == "teacher_replay":
        assert (
            getattr(args, "policy_family", None) == "moss_tts_local"
            and getattr(args, "moss_local_replay_manifest", None)
            and getattr(args, "moss_local_replay_mode", "teacher") == "teacher"
            and getattr(args, "moss_local_mopd_estimator", None) in {"dense_reverse", "dense_forward"}
            and getattr(args, "moss_local_objective", None) == "mopd"
            and getattr(args, "moss_local_old_policy_source", None) == "trainer_preupdate"
            and not getattr(args, "moss_local_async", False)
        ), "Teacher-origin trajectories require explicitly enabled synchronous native mixed distillation"
        replay = sample.metadata["moss_teacher_replay"]
        scores = sample.metadata["mopd_scores"]
        assert (
            replay["behavior_weight_version"] == provenance.behavior_version
        ), "Teacher replay behavior version changed"
        assert replay["teacher_weight_sha256"] == provenance.teacher_digest, "Replay teacher changed"
        assert provenance.student_scoring_version == str(
            scores["student_score"]["weight_version"]
        ), "Student scoring version changed"
    version = provenance.training_version
    assert re.fullmatch(r"[0-9]+", version), f"Unpublished MOSS weight version: {version}"
