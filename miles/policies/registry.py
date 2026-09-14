"""Resolve policy extensions without importing model or distributed runtimes.

Text keeps the existing Miles implementations. Structured policies provide
only the extensions needed by the callers; backend selection is independent.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from miles.utils.function_registry import load_function

if TYPE_CHECKING:
    from miles.policies.base import RolloutDataAdapter, ServingWeightAdapter, TrainingWorkflow


@dataclass(frozen=True)
class PolicyDefinition:
    family: str
    rollout_adapter_path: str | None = None
    workflow_path: str | None = None
    model_path: str | None = None
    weight_adapter_path: str | None = None
    trajectory_path: str | None = None
    serving_spec_path: str | None = None
    resume_validator_path: str | None = None
    uses_processor: bool = True
    uses_hf_config_validation: bool = True

    @cached_property
    def rollout_adapter(self) -> RolloutDataAdapter | None:
        return load_function(self.rollout_adapter_path)() if self.rollout_adapter_path else None

    @cached_property
    def weight_adapter(self) -> ServingWeightAdapter | None:
        return load_function(self.weight_adapter_path)() if self.weight_adapter_path else None

    def create_workflow(self) -> TrainingWorkflow | None:
        return load_function(self.workflow_path)() if self.workflow_path else None

    @cached_property
    def serving_spec(self):
        if self.serving_spec_path is None:
            raise ValueError(f"Policy {self.family!r} has no structured serving contract")
        return load_function(self.serving_spec_path)

    def decode_trajectory(self, data):
        if self.trajectory_path is None:
            raise ValueError(f"Policy {self.family!r} does not define a structured trajectory codec")
        return load_function(self.trajectory_path).from_dict(data)

    def validate_resume(self, args: Namespace) -> None:
        if self.resume_validator_path:
            load_function(self.resume_validator_path)(args)


_LOCAL = "miles.policies.moss_tts_local"
_POLICIES = {
    "text": PolicyDefinition(family="text"),
    "moss_tts_local": PolicyDefinition(
        family="moss_tts_local",
        rollout_adapter_path=f"{_LOCAL}.rollout_data.MossTTSLocalRolloutDataAdapter",
        workflow_path=f"{_LOCAL}.workflow.MossTTSLocalTrainingWorkflow",
        model_path=f"{_LOCAL}.model.MossTTSLocalMegatronModel",
        weight_adapter_path=f"{_LOCAL}.serving_weight_adapter.MossTTSLocalServingWeightAdapter",
        trajectory_path=f"{_LOCAL}.types.MossTTSLocalTrajectoryV2",
        serving_spec_path=f"{_LOCAL}.spec.MOSS_TTS_LOCAL_SPEC",
        resume_validator_path=f"{_LOCAL}.checkpoint.validate_resume_implementation",
        uses_processor=False,
        uses_hf_config_validation=False,
    ),
}


def resolve_policy(family: str = "text") -> PolicyDefinition:
    try:
        return _POLICIES[family]
    except KeyError:
        raise ValueError(f"Unknown policy family {family!r}; expected one of {tuple(_POLICIES)}") from None


def policy_for_args(args: Namespace) -> PolicyDefinition:
    return resolve_policy(getattr(args, "policy_family", "text"))
