"""Serializable, stage-aware Omni client for Miles' weight transfer layer."""

import dataclasses
import logging
import os

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils.http_utils import GeneralHttpClientProvider

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SGLangOmniApiClient(SGLangApiClient):
    train_stage: str = "tts_engine"
    api_key_env: str | None = None

    async def _admin(self, endpoint, payload=None):
        headers = {}
        if self.api_key_env:
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        response = await GeneralHttpClientProvider.client().post(
            f"{self.server_url}/{endpoint}",
            json={**(payload or {}), "stages": [self.train_stage]},
            headers=headers,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("success") is False:
            raise RuntimeError(f"Omni {endpoint} failed: {result}")
        return result

    async def _make_request(self, endpoint, payload=None):
        return await self._admin(endpoint, payload)

    async def get_model_info(self):
        result = await self._admin("model_info")
        for item in result.get("stages", result.get("results", [])):
            if item.get("stage") == self.train_stage:
                data = item.get("data", {})
                if data.get("skipped") or data.get("unsupported"):
                    raise RuntimeError(f"Omni stage is unavailable: {self.train_stage}")
                return data
        raise RuntimeError(f"Omni omitted stage {self.train_stage} in model_info")

    async def get_weight_version(self):
        return str((await self.get_model_info())["weight_version"])

    async def pause_generation(self, mode="abort"):
        if mode not in {"abort", "retract"}:
            raise ValueError("MOSS supports aborting outstanding requests before refit")
        return await self._admin("pause_generation", {"mode": "abort"})

    async def continue_generation(self):
        return await self._admin("continue_generation")

    async def flush_cache(self):
        # Each stage refit flushes its own cache while keeping generation paused.
        return {"success": True}

    async def update_weights_from_distributed(
        self, names, dtypes, shapes, group_name, flush_cache=False, weight_version=None, selector="all"
    ):
        if selector != "all":
            raise ValueError("MOSS only supports full-model refits")
        return await self._admin(
            "update_weights_from_distributed",
            {
                "names": list(names),
                "dtypes": [str(x).replace("torch.", "") for x in dtypes],
                "shapes": [list(x) for x in shapes],
                "group_name": group_name,
                "flush_cache": True,
                "keep_pause": True,
                "weight_version": str(weight_version),
            },
        )

    async def check_weights(self, action, **kwargs):
        result = await self._admin("weights_checker", {"action": action, **kwargs})
        for item in result.get("stages", result.get("results", [])):
            if item.get("stage") != self.train_stage:
                continue
            data = item.get("data", {})
            if action == "compare":
                if data.get("matched") is not True:
                    raise RuntimeError(f"MOSS strict weight comparison failed: {data}")
                logger.info("MOSS strict weight comparison passed: tensors=%s", data.get("tensor_count"))
            return data
        raise RuntimeError(f"Omni weights_checker omitted stage {self.train_stage}")
