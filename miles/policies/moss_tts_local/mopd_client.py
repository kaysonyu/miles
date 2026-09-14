"""Domain-routed, microbatched Local teacher scoring overlapped with rollout."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
import torch

from miles.policies.moss_tts_local.spec import MOSS_TTS_LOCAL_SPEC


def teacher_routes(entries):
    routes = {}
    for entry in entries or []:
        domain, sep, endpoint = entry.partition("=")
        if not sep or not domain or domain in routes or not endpoint.startswith(("http://", "https://")):
            raise ValueError("MOPD teachers must be unique DOMAIN=http(s)://host/score_actions entries")
        routes[domain] = endpoint.rstrip("/")
    return routes


def score_payload(sample, temperature):
    trace = sample.structured_trajectory
    return dict(
        version=1,
        sample_id=str(sample.index),
        prompt_rows=trace.prompt_rows.tolist(),
        decisions=trace.decisions.tolist(),
        codes=trace.codes.tolist(),
        temperature=float(temperature),
    )


def payload_hash(payload):
    body = {key: value for key, value in payload.items() if key != "sample_id"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_score(payload, score):
    if score.get("sample_id") != payload["sample_id"] or score.get("input_sha256") != payload_hash(payload):
        raise ValueError("Teacher response does not match the exact student trajectory")
    if (
        score.get("temperature") != payload["temperature"]
        or score.get("logprob_semantics") != "temperature_scaled_full_vocab_v1"
    ):
        raise ValueError("Teacher scoring distribution does not match the student sampling contract")
    MOSS_TTS_LOCAL_SPEC.validate_identity(score["model_identity"], require_hash=True)
    d = torch.tensor(score["decision_logprobs"], dtype=torch.float32)
    c = torch.tensor(score["code_logprobs"], dtype=torch.float32).reshape(-1, 12)
    if d.shape != (len(payload["decisions"]),) or c.shape != (len(payload["codes"]), 12):
        raise ValueError("Teacher scores do not cover all student actions")
    if not torch.isfinite(d).all() or not torch.isfinite(c).all():
        raise ValueError("Non-finite teacher scores must not enter MOPD")
    if (d > 1e-6).any() or (c > 1e-6).any():
        raise ValueError("Teacher log probabilities must be non-positive")
    if score.get("weight_version") is None:
        raise ValueError("Teacher must identify its frozen weight version")
    return d, c


@dataclass
class _PendingScore:
    payload: dict
    future: asyncio.Future


class LocalTeacherClient:
    """One rollout-owned client; never shares a student weight-update endpoint."""

    def __init__(self, args):
        self.allow_teacher_replay = (
            bool(getattr(args, "moss_local_replay_manifest", None))
            and getattr(args, "moss_local_replay_mode", "teacher") == "teacher"
        )
        self.routes = teacher_routes(args.moss_local_mopd_teachers)
        self.student_endpoint = args.moss_local_student_score_endpoint
        self.default_domain = args.moss_local_mopd_default_domain
        self.batch_size = args.moss_local_mopd_score_batch_size
        self.wait_s = args.moss_local_mopd_batch_wait_ms / 1000
        self.retries = args.moss_local_mopd_retries
        self.client = httpx.AsyncClient(timeout=args.moss_local_mopd_timeout, trust_env=False)
        self.limit = asyncio.Semaphore(args.moss_local_mopd_concurrency)
        self.pending = {}
        self.flush_tasks = set()
        self.versions = {}
        self.digests = {}
        self.manifest_path = Path(args.save) / "mopd_teachers.json" if args.save else None
        previous = Path(args.load) / "mopd_teachers.json" if args.load else None
        self.expected_manifest = json.loads(previous.read_text()) if previous and previous.exists() else {}
        self.manifest = {}
        self.info_lock = asyncio.Lock()
        self.requests = 0
        self.samples = 0
        self.http_seconds = 0.0

    async def score(self, sample, *, temperature):
        domain = (sample.metadata or {}).get("domain", self.default_domain)
        if domain not in self.routes:
            raise ValueError(f"No MOPD teacher configured for domain {domain!r}")
        endpoint = self.routes[domain]
        payload = score_payload(sample, temperature)
        future = asyncio.get_running_loop().create_future()
        queue = self.pending.setdefault(endpoint, [])
        queue.append(_PendingScore(payload, future))
        if len(queue) == 1:
            task = asyncio.create_task(self._flush_later(endpoint))
            self.flush_tasks.add(task)
            task.add_done_callback(self.flush_tasks.discard)
        result = await future
        validate_score(payload, result)
        replay = (sample.metadata or {}).get("moss_teacher_replay")
        scoring_version = str(result["student_score"]["weight_version"])
        if replay:
            if not self.allow_teacher_replay or replay["teacher_weight_sha256"] != result["teacher_weight_sha256"]:
                raise ValueError("Teacher replay is disabled or its teacher identity changed")
            if replay["behavior_weight_version"] != sample.structured_trajectory.weight_version:
                raise ValueError("Teacher replay behavior provenance changed")
        elif scoring_version != str(sample.structured_trajectory.weight_version):
            raise ValueError("Student scoring version does not match its rollout")
        sample.metadata = dict(sample.metadata or {})
        sample.metadata["moss_training_policy_version"] = scoring_version
        sample.metadata["mopd_teacher"] = domain
        sample.metadata["mopd_scores"] = result
        sample.reward = 0.0
        return sample

    async def _flush_later(self, endpoint):
        await asyncio.sleep(self.wait_s)
        queued = self.pending.pop(endpoint)
        for start in range(0, len(queued), self.batch_size):
            items = queued[start : start + self.batch_size]
            try:
                async with self.limit:
                    result = await self._request(endpoint, items)
                scores = result["teacher"].get("results", [])
                student_scores = result["student"].get("results", [])
                if len(student_scores) != len(items):
                    raise ValueError("Student scoring response size mismatch")
                if len(scores) != len(items):
                    raise ValueError("Teacher batch response size mismatch")
                for item, score, student in zip(items, scores, student_scores, strict=True):
                    validate_score(item.payload, score)
                    validate_score(item.payload, student)
                    if not score.get("scoring_batch_sha256") or score["scoring_batch_sha256"] != student.get(
                        "scoring_batch_sha256"
                    ):
                        raise ValueError("Teacher and student prefill batch contexts differ")
                    if score.get("score_chunk_size") != student.get("score_chunk_size"):
                        raise ValueError("Teacher and student local score chunk sizes differ")
                    score["student_score"] = student
                versions = {str(score["weight_version"]) for score in scores}
                if len(versions) != 1:
                    raise ValueError("Teacher batch crossed a weight update")
                if any(score.get("teacher_weight_sha256") != self.digests[endpoint] for score in scores):
                    raise ValueError("Teacher weight identity changed or is missing")
                version = versions.pop()
                if self.versions.setdefault(endpoint, version) != version:
                    raise ValueError("Frozen teacher changed weights during the run")
                for item, score in zip(items, scores, strict=True):
                    if not item.future.done():
                        item.future.set_result(score)
            except Exception as exc:
                for item in items:
                    if not item.future.done():
                        item.future.set_exception(exc)

    async def _request(self, endpoint, items):
        async with self.info_lock:
            if endpoint not in self.versions:
                url = urlsplit(endpoint)
                response = await self.client.post(
                    urlunsplit((url.scheme, url.netloc, "/model_info", "", "")), json={"stages": ["tts_engine"]}
                )
                response.raise_for_status()
                stages = response.json().get("stages", [])
                info = next((x["data"] for x in stages if x.get("stage") == "tts_engine"), None)
                if not info or not info.get("supports_action_scoring") or info.get("supports_weight_update"):
                    raise ValueError("MOPD endpoint must be a frozen Local scoring pipeline")
                digest = info.get("teacher_weight_sha256")
                if not isinstance(digest, str) or len(digest) != 64:
                    raise ValueError("Teacher must publish a frozen weight SHA256")
                self.digests[endpoint] = digest
                for domain, url_value in self.routes.items():
                    if url_value != endpoint:
                        continue
                    identity = {"weight_sha256": digest, "model_identity": info["model_identity"]}
                    if self.expected_manifest and self.expected_manifest.get(domain) != identity:
                        raise ValueError(f"MOPD resume changed teacher identity for {domain}")
                    self.manifest[domain] = identity
                if self.manifest_path is not None:
                    self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.manifest_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(self.manifest, sort_keys=True, indent=2))
                    temporary.replace(self.manifest_path)
                self.versions[endpoint] = str(info["weight_version"])
        started = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                body = {"samples": [item.payload for item in items]}
                response, student_response = await asyncio.gather(
                    self.client.post(endpoint, json=body),
                    self.client.post(self.student_endpoint, json=body),
                )
                response.raise_for_status()
                student_response.raise_for_status()
                self.requests += 1
                self.samples += len(items)
                self.http_seconds += time.perf_counter() - started
                return {"teacher": response.json(), "student": student_response.json()}
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                    raise
                if attempt == self.retries:
                    raise
                await asyncio.sleep(0.1 * 2**attempt)

    async def close(self):
        if self.flush_tasks:
            await asyncio.gather(*tuple(self.flush_tasks))
        await self.client.aclose()
