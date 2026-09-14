"""Small interfaces for policy-specific Miles behavior.

Only dependency-light types live here.  Concrete training implementations are
loaded lazily by :mod:`miles.policies.registry` so the rollout manager remains
usable in CPU-only processes.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from miles.utils.types import RolloutBatch, Sample


class StructuredTrajectory(Protocol):
    """The shared envelope operations; each policy retains its concrete payload."""

    model_family: str
    weight_version: str

    def validate(self) -> None: ...

    def to_dict(self) -> dict[str, Any]: ...


@runtime_checkable
class RolloutDataAdapter(Protocol):
    """Convert completed samples and partition them for training ranks."""

    def samples_to_train_data(
        self,
        args: Namespace,
        samples: list[Sample],
        *,
        reward_postprocess: Callable[[list[Sample]], tuple[list[float], list[float]]],
    ) -> RolloutBatch:
        """Return one per-sample rollout batch before DP partitioning."""

    def split_by_dp(
        self,
        args: Namespace,
        data: RolloutBatch,
        train_parallel_config: dict[str, int],
    ) -> list[Any]:
        """Return one raw shard per data-parallel rank; Miles owns transport."""

    def prepare_rollout_data(self, data: RolloutBatch, *, device: torch.device | int) -> RolloutBatch:
        """Validate and move policy tensors to the training device."""

    def validate_sample_version(self, args: Namespace, sample: Sample) -> None:
        """Validate the structured sample's behavior and scoring provenance."""

    def dispose(self) -> None:
        """Release policy-owned rollout resources."""

    preserve_partition: bool


@dataclass(frozen=True)
class TrainingContext:
    """Model state and explicit trainer capabilities for one rollout."""

    args: Namespace
    model: Sequence[torch.nn.Module]
    optimizer: Any
    opt_param_scheduler: Any
    runtime: TrainingRuntime
    rollout_id: int


class TrainingRuntime(Protocol):
    """Trainer-owned lifecycle capabilities available to a policy workflow."""

    backup_required: bool
    rollout_data_postprocess: Callable | None

    @property
    def published_version(self) -> str: ...

    def behavior_snapshot(self, version: str) -> AbstractContextManager: ...

    def finish_rollout(self, rollout_id: int) -> None: ...

    def pop_weight_metrics(self) -> dict: ...


@runtime_checkable
class TrainingWorkflow(Protocol):
    """Own the policy-specific path from a DP-local batch to an optimizer step."""

    needs_tokenizer: bool

    def uses_behavior_snapshots(self, args: Namespace) -> bool:
        """Whether published versions must retain a trainer replay snapshot."""

    def prepare_rollout_data(self, data: RolloutBatch, *, device: torch.device | int) -> RolloutBatch:
        """Validate and move policy tensors to the training device."""

    def compute_log_probs(
        self,
        context: TrainingContext,
        data_iterator: Any,
        num_microbatches: list[int],
        *,
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:
        """Replay selected actions and return per-sample values."""

    def train_actor(
        self,
        context: TrainingContext,
        rollout_id: int,
        rollout_data: RolloutBatch,
        *,
        external_data: Any = None,
    ) -> None:
        """Compute old-policy state, advantages, and train the actor."""


@runtime_checkable
class ServingWeightAdapter(Protocol):
    """Convert one gathered trainer parameter to canonical serving tensors."""

    def convert_parameter(
        self,
        args: Namespace,
        model_name: str,
        name: str,
        parameter: torch.Tensor,
        quantization_config: dict[str, Any] | None = None,
        *,
        transform_ue8m0: bool = True,
    ) -> list[tuple[str, torch.Tensor]]:
        """Return zero or more canonical serving tensors for one parameter."""

    def validate_manifest(self, names: set[str]) -> None:
        """Fail if a completed export is missing or duplicates required tensors."""

    def expected_names(self, args: Namespace) -> set[str] | None:
        """Return an exact checkpoint manifest when the artifact provides one."""
