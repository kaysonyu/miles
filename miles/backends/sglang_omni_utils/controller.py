"""Manage externally launched Omni pipelines without allocating SGLang workers."""

import asyncio
import os
from urllib.parse import urlsplit

from miles.backends.sglang_omni_utils.api_client import SGLangOmniApiClient
from miles.backends.sglang_omni_utils.external import discover_external_omni_engines
from miles.ray.rollout.inference_controller import InferenceController, UpdatableEngines
from miles.utils.context_lock import acquires_lock, enforce_lock_discipline, lock_exempt, releases_lock, with_lock


@enforce_lock_discipline
class OmniInferenceController(InferenceController):
    @lock_exempt
    def __init__(self, args):
        super().__init__(args)
        self._clients = []
        self._infos = []

    @lock_exempt
    async def init(self):
        if self.args.debug_train_only:
            return
        key_env = self.args.sglang_omni_admin_api_key_env
        self._infos = await asyncio.to_thread(
            discover_external_omni_engines,
            self.args.sglang_omni_endpoints,
            train_stage=self.args.sglang_omni_train_stage,
            admin_api_key=os.getenv(key_env) if key_env else None,
        )
        self._clients = [SGLangOmniApiClient(info.base_url, info.train_stage, key_env) for info in self._infos]
        if self.args.check_weight_update_equal:
            await asyncio.gather(*(client.check_weights("snapshot") for client in self._clients))
        endpoint = urlsplit(self._infos[0].base_url)
        self.args.sglang_router_ip = endpoint.hostname
        self.args.sglang_router_port = endpoint.port

    @acquires_lock
    async def start_update_weights(self):
        return UpdatableEngines(
            rollout_engines=self._clients,
            engine_gpu_counts=[info.tp_size for info in self._infos],
            engine_gpu_offsets=list(range(len(self._infos))),
            snapshot_cell_id_to_hashes={f"omni-{i}": info.base_url for i, info in enumerate(self._infos)},
        )

    @releases_lock
    async def end_update_weights(self, snapshot_cell_id_to_hashes):
        versions = await asyncio.gather(*(client.get_weight_version() for client in self._clients))
        if len(set(versions)) > 1:
            raise RuntimeError(f"Omni replicas disagree on weight versions: {versions}")

    @with_lock
    async def check_weights(self, action, allow_quant_error=False, selector="all", skip_list=None):
        return await asyncio.gather(*(client.check_weights(action) for client in self._clients))
