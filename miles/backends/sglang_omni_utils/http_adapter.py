"""HTTP and in-memory adapters for the SGLang-Omni interface."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import httpx
import requests


class SGLangOmniResponseError(RuntimeError):
    """Raised when Omni returns a transport or admin failure."""


def normalize_omni_base_url(base_url: str) -> str:
    value = str(base_url).strip().rstrip("/")
    if "://" not in value:
        value = f"http://{value}"
    parsed = httpx.URL(value)
    if parsed.scheme not in ("http", "https") or parsed.host is None or parsed.port is None:
        raise ValueError(f"Invalid SGLang-Omni endpoint {base_url!r}; use host:port or http(s)://host:port.")
    return str(parsed).rstrip("/")


class SGLangOmniHttpAdapter:
    """Production adapter for one top-level SGLang-Omni pipeline endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        admin_api_key: str | None = None,
        admin_timeout_s: float = 300.0,
        generate_timeout_s: float | None = None,
    ) -> None:
        self.base_url = normalize_omni_base_url(base_url)
        self.admin_timeout_s = float(admin_timeout_s)
        self.generate_timeout_s = generate_timeout_s
        self._session = requests.Session()
        self._session.trust_env = False
        self._headers = {"Content-Type": "application/json"}
        if admin_api_key:
            self._headers["Authorization"] = f"Bearer {admin_api_key}"

    @staticmethod
    def _decode_response(response: requests.Response | httpx.Response, *, require_success: bool) -> dict[str, Any]:
        try:
            response.raise_for_status()
        except Exception as exc:
            body = getattr(response, "text", "")
            raise SGLangOmniResponseError(
                f"SGLang-Omni request failed with HTTP {getattr(response, 'status_code', '?')}: {body}"
            ) from exc
        try:
            payload = response.json()
        except Exception as exc:
            raise SGLangOmniResponseError("SGLang-Omni returned a non-JSON response.") from exc
        if not isinstance(payload, dict):
            raise SGLangOmniResponseError(f"SGLang-Omni response must be a JSON object, got {type(payload).__name__}.")
        if require_success and payload.get("success") is not True:
            raise SGLangOmniResponseError(f"SGLang-Omni admin operation failed: {payload!r}")
        return payload

    def _get(self, endpoint: str, *, require_success: bool = False) -> dict[str, Any]:
        response = self._session.get(
            f"{self.base_url}{endpoint}",
            headers=self._headers,
            timeout=self.admin_timeout_s,
        )
        return self._decode_response(response, require_success=require_success)

    def _post(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        require_success: bool = True,
    ) -> dict[str, Any]:
        response = self._session.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            headers=self._headers,
            timeout=self.admin_timeout_s,
        )
        return self._decode_response(response, require_success=require_success)

    async def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        timeout = (
            httpx.Timeout(self.generate_timeout_s) if self.generate_timeout_s is not None else httpx.Timeout(None)
        )
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(
                f"{self.base_url}/generate",
                json=request,
                headers=self._headers,
            )
            return self._decode_response(response, require_success=False)

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def model_info(self, *, stages: list[str] | None = None) -> dict[str, Any]:
        return self._post("/model_info", {"stages": stages} if stages else {}, require_success=True)

    def pause_generation(self, *, stages: list[str], mode: str = "abort") -> dict[str, Any]:
        return self._post("/pause_generation", {"stages": stages, "mode": mode})

    def continue_generation(self, *, stages: list[str]) -> dict[str, Any]:
        return self._post("/continue_generation", {"stages": stages})

    def init_weights_update_group(self, payload: dict[str, Any], *, stages: list[str]) -> dict[str, Any]:
        return self._post("/init_weights_update_group", {**payload, "stages": stages})

    def destroy_weights_update_group(self, payload: dict[str, Any], *, stages: list[str]) -> dict[str, Any]:
        return self._post("/destroy_weights_update_group", {**payload, "stages": stages})

    def update_weights_from_distributed(self, payload: dict[str, Any], *, stages: list[str]) -> dict[str, Any]:
        return self._post("/update_weights_from_distributed", {**payload, "stages": stages})

    def update_weights_from_disk(self, payload: dict[str, Any], *, stages: list[str]) -> dict[str, Any]:
        return self._post("/update_weights_from_disk", {**payload, "stages": stages})

    def weights_checker(self, payload: dict[str, Any], *, stages: list[str]) -> dict[str, Any]:
        return self._post("/weights_checker", {**payload, "stages": stages})

    def close(self) -> None:
        self._session.close()


class InMemorySGLangOmniAdapter:
    """Test adapter with the same observable interface as the HTTP adapter."""

    def __init__(
        self,
        *,
        generate_result: dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        model_info_result: dict[str, Any] | None = None,
        health_result: dict[str, Any] | None = None,
    ) -> None:
        self.generate_result = generate_result or {}
        self.model_info_result = model_info_result or {"success": True, "results": []}
        self.health_result = health_result or {"status": "healthy", "running": True}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def _record(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, copy.deepcopy(payload)))
        return {"success": True, "message": "ok", "results": []}

    async def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("generate", copy.deepcopy(request)))
        result = self.generate_result(request) if callable(self.generate_result) else self.generate_result
        return copy.deepcopy(result)

    def health(self) -> dict[str, Any]:
        self.calls.append(("health", {}))
        return copy.deepcopy(self.health_result)

    def model_info(self, *, stages: list[str] | None = None) -> dict[str, Any]:
        self.calls.append(("model_info", {"stages": copy.deepcopy(stages)}))
        return copy.deepcopy(self.model_info_result)

    def pause_generation(self, *, stages: list[str], mode: str = "abort") -> dict[str, Any]:
        return self._record("pause_generation", {"stages": stages, "mode": mode})

    def continue_generation(self, *, stages: list[str]) -> dict[str, Any]:
        return self._record("continue_generation", {"stages": stages})

    def init_weights_update_group(self, payload: dict[str, Any], *, stages: list[str]) -> dict[str, Any]:
        return self._record("init_weights_update_group", {**payload, "stages": stages})

    def destroy_weights_update_group(self, payload: dict[str, Any], *, stages: list[str]) -> dict[str, Any]:
        return self._record("destroy_weights_update_group", {**payload, "stages": stages})

    def update_weights_from_distributed(self, payload: dict[str, Any], *, stages: list[str]) -> dict[str, Any]:
        return self._record("update_weights_from_distributed", {**payload, "stages": stages})

    def update_weights_from_disk(self, payload: dict[str, Any], *, stages: list[str]) -> dict[str, Any]:
        return self._record("update_weights_from_disk", {**payload, "stages": stages})

    def weights_checker(self, payload: dict[str, Any], *, stages: list[str]) -> dict[str, Any]:
        return self._record("weights_checker", {**payload, "stages": stages})

    def close(self) -> None:
        self.closed = True
