"""Full NCCL refit with Omni's stage pause/version/cache transaction semantics."""

import json
import logging
from pathlib import Path

from miles.backends.training_utils.weight_update.protocols.broadcast import (
    UpdateWeightFromDistributed,
    update_weights_from_distributed,
)
from miles.policies.moss_tts_local.serving_weight_adapter import MossTTSLocalServingWeightAdapter
from miles.utils import async_utils

logger = logging.getLogger(__name__)


class OmniWeightTransfer(UpdateWeightFromDistributed):
    use_weight_update_session = False
    supports_lora = False

    def __init__(self, args):
        super().__init__(args)
        index = Path(args.hf_checkpoint) / "model.safetensors.index.json"
        self._expected_names = None
        if index.exists():
            self._expected_names = set(json.loads(index.read_text())["weight_map"]) - {"text_lm_head.weight"}
        self._sent_names = set()

    def begin_sync(self, weight_version, iter_buckets):
        self._pending_version = weight_version
        self._sent_names.clear()
        if self.is_sender:
            async_utils.wait_futures(
                [async_utils.submit(client.pause_generation(mode="abort")) for client in self.rollout_engines]
            )
        return True

    def send_bucket(self, bucket):
        names = [name for name, _ in bucket]
        if len(set(names)) != len(names) or self._sent_names.intersection(names):
            raise ValueError("MOSS refit contains duplicate tensor names")
        self._sent_names.update(names)
        futures = update_weights_from_distributed(
            self.group_name,
            self._model_update_groups,
            self.rollout_engines,
            bucket,
            selector="all",
            weight_version=str(self._pending_version),
        )
        async_utils.wait_futures(futures)
        bucket.clear()

    def finalize(self, weight_version):
        if not self.is_sender:
            return
        MossTTSLocalServingWeightAdapter().validate_manifest(self._sent_names)
        if self._expected_names is not None and self._sent_names != self._expected_names:
            raise ValueError(
                f"Incomplete MOSS refit: missing={sorted(self._expected_names - self._sent_names)}, "
                f"unexpected={sorted(self._sent_names - self._expected_names)}"
            )
        logger.info("MOSS complete refit: version=%s tensors=%d", weight_version, len(self._sent_names))
        versions = async_utils.wait_futures(
            [async_utils.submit(client.get_weight_version()) for client in self.rollout_engines]
        )
        if any(version != str(weight_version) for version in versions):
            raise RuntimeError(f"Omni refit version mismatch: expected {weight_version}, got {versions}")
        async_utils.wait_futures([async_utils.submit(client.continue_generation()) for client in self.rollout_engines])
