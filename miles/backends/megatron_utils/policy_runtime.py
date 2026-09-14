"""Trainer-owned parameter snapshots and lifecycle for policy workflows.

Only the trainer binds private actor operations. Policies receive a versioned
snapshot lease, publication version and rollout-finalization operations.
"""

from contextlib import contextmanager


class MegatronPolicyRuntime:
    def __init__(
        self,
        *,
        weight_updater,
        weights_backuper,
        switch_model,
        profiler,
        backup_required: bool,
        rollout_data_postprocess,
        retain_behavior: bool,
    ):
        self._weight_updater = weight_updater
        self._weights_backuper = weights_backuper
        self._switch_model = switch_model
        self._profiler = profiler
        self.backup_required = backup_required
        self.rollout_data_postprocess = rollout_data_postprocess
        self._retain_behavior = retain_behavior
        self._snapshot_versions: dict[str, int] = {}

    @property
    def published_version(self) -> str:
        return str(self._weight_updater.weight_version)

    def on_weights_published(self) -> bool:
        """Record labels after the trainer rotates rollout_actor into old_actor."""
        if not self._retain_behavior:
            return False
        current = int(self.published_version)
        previous = self._snapshot_versions.get("rollout_actor", current)
        if current < 1 or (self._snapshot_versions and current != previous + 1):
            raise ValueError("Behavior snapshots require consecutive published versions")
        self._snapshot_versions = {"old_actor": previous, "rollout_actor": current}
        return True

    @contextmanager
    def behavior_snapshot(self, version: str):
        requested = int(version)
        tag = next((tag for tag, value in self._snapshot_versions.items() if value == requested), None)
        if tag is None:
            raise ValueError(f"No retained trainer snapshot matches behavior version {version}")
        try:
            self._switch_model(tag)
            yield
        finally:
            self._switch_model("actor")

    def finish_rollout(self, rollout_id: int) -> None:
        self._profiler.step(rollout_id=rollout_id)
        if self.backup_required:
            self._weights_backuper.backup("actor")

    def pop_weight_metrics(self) -> dict:
        return self._weight_updater.pop_metrics()
