"""Audited frozen-teacher trajectories for opt-in mixed native distillation.

The original teacher behavior version and logprobs remain on the trajectory.
Only the current student's scoring version is used for trainer freshness checks;
no importance-ratio or on-policy claim is made for these replay samples.
"""

import hashlib
import json
from pathlib import Path

from miles.policies.moss_tts_local.types import MediaArtifact, MossTTSLocalTrajectoryV2
from miles.utils.types import Sample


class FrozenTeacherReplay:
    """Immutable replay pool owned by the persistent rollout lifecycle."""

    def __init__(self, args):
        manifest_path = Path(args.moss_local_replay_manifest)
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(raw_manifest)
        if manifest.get("schema_version") != 1 or manifest["temperature"] != float(args.rollout_temperature):
            raise ValueError("Teacher replay manifest schema/temperature does not match this run")
        pool_path = Path(manifest["pool_file"])
        raw_pool = pool_path.read_bytes()
        if hashlib.sha256(raw_pool).hexdigest() != manifest["pool_sha256"]:
            raise ValueError("Teacher replay pool contents changed")
        self.identity = {
            "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "pool_sha256": manifest["pool_sha256"],
        }
        self.every_n_groups = int(args.moss_local_replay_every_n_groups)
        self.groups_per_rollout = int(args.rollout_batch_size)
        self.mode = getattr(args, "moss_local_replay_mode", "teacher")
        self.selection = manifest.get("selection", "bucket")
        if self.selection not in {"bucket", "exact_sample"}:
            raise ValueError("Unknown teacher replay selection rule")
        allow_truncated = manifest.get("allow_truncated", False)
        if allow_truncated and self.selection != "exact_sample":
            raise ValueError("Unfiltered truncated replay requires an exact sample plan")
        if self.mode == "prompt_only":
            self.identity["replay_mode"] = self.mode
        self.entries = {}
        self.exact_entries = {}
        for line in raw_pool.decode().splitlines():
            entry = json.loads(line)
            domain, bucket = entry["metadata"]["domain"], entry["metadata"]["bucket"]
            if entry["teacher_weight_sha256"] != manifest["teachers"][domain]["weight_sha256"]:
                raise ValueError("Replay entry has the wrong frozen teacher")
            trace_bytes = Path(entry["trajectory_file"]).read_bytes()
            if hashlib.sha256(trace_bytes).hexdigest() != entry["trajectory_sha256"]:
                raise ValueError("Replay trajectory contents changed")
            trace = json.loads(trace_bytes)
            trajectory = MossTTSLocalTrajectoryV2.from_dict(trace)
            if any(
                trajectory.sampling.get(key, trajectory.sampling.get("temperature")) != manifest["temperature"]
                for key in ("text_temperature", "audio_temperature")
            ):
                raise ValueError("Replay trajectory sampling temperature changed")
            if entry.get("weight_version", trajectory.weight_version) != trajectory.weight_version:
                raise ValueError("Replay trajectory behavior version changed")
            if (trajectory.finish_reason != "stop" and not allow_truncated) or trajectory.num_frames == 0:
                raise ValueError("Replay pool must contain complete nonempty teacher speech")
            self.entries.setdefault((domain, bucket), []).append((entry, trace))
            if self.selection == "exact_sample":
                key = int(entry["target_rollout_id"]), int(entry["target_sample_index"])
                if key in self.exact_entries:
                    raise ValueError("Duplicate replay entry for one planned sample")
                self.exact_entries[key] = entry, trace
        if not self.entries:
            raise ValueError("Teacher replay pool is empty")
        previous = Path(args.load) / "teacher_replay_pool.json" if args.load else None
        if previous and previous.exists() and json.loads(previous.read_text()) != self.identity:
            raise ValueError("Teacher replay pool changed on resume")
        if args.save:
            target = Path(args.save) / "teacher_replay_pool.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(self.identity, indent=2))

    def select_group(self, rollout_id, group_index):
        return (rollout_id * self.groups_per_rollout + group_index) % self.every_n_groups == 0

    def _select_entry(self, sample, *, rollout_id, seed):
        if sample.status != Sample.Status.PENDING:
            raise ValueError("Teacher replay requires a pending sample")
        original = dict(sample.metadata or {})
        key = original["domain"], original["bucket"]
        if self.selection == "exact_sample":
            selected = self.exact_entries.get((rollout_id, sample.index))
            if selected is None:
                raise ValueError("Replay plan lacks this rollout/sample")
            entry, _ = selected
            if (
                entry["text"] != sample.prompt
                or (entry["metadata"]["domain"], entry["metadata"]["bucket"]) != key
                or entry["metadata"]["tts_params"] != original["tts_params"]
                or entry["metadata"].get("text_hash") != original.get("text_hash")
            ):
                raise ValueError("Replay plan does not match the original prompt and instruction")
            return selected
        choices = self.entries.get(key)
        if not choices:
            raise ValueError(f"Replay pool lacks domain/bucket {key}")
        selector = hashlib.sha256(f"{rollout_id}:{sample.index}:{seed}".encode()).digest()
        return choices[int.from_bytes(selector[:8], "big") % len(choices)]

    def materialize_prompt(self, sample, *, rollout_id, seed):
        entry, _ = self._select_entry(sample, rollout_id=rollout_id, seed=seed)
        original = dict(sample.metadata or {})
        sample.prompt = sample.label = entry["text"]
        sample.metadata = dict(entry["metadata"])
        sample.metadata["moss_prompt_selection"] = {
            "entry_id": entry["replay_id"],
            "pool_sha256": self.identity["pool_sha256"],
            "original_slot_text_hash": original.get("text_hash"),
        }
        return sample

    def materialize(self, sample, *, rollout_id, seed):
        entry, trace = self._select_entry(sample, rollout_id=rollout_id, seed=seed)
        original = dict(sample.metadata or {})
        trajectory = MossTTSLocalTrajectoryV2.from_dict(trace)
        sample.prompt = sample.label = entry["text"]
        sample.structured_trajectory = trajectory
        sample.status = Sample.Status.COMPLETED if trajectory.finish_reason == "stop" else Sample.Status.TRUNCATED
        sample.response = ""
        sample.artifacts = [
            MediaArtifact(
                modality="audio",
                mime_type="audio/wav",
                sample_rate=entry["sample_rate"],
                sha256=entry["audio_sha256"],
                num_bytes=entry["audio_bytes"],
                uri=entry["audio"],
                inline_base64=None,
            )
        ]
        sample.metadata = dict(entry["metadata"])
        sample.metadata.update(
            moss_weight_version=trajectory.weight_version,
            frame_count=trajectory.num_frames,
            action_count=trajectory.num_actions,
            audio_sha256=entry["audio_sha256"],
            moss_teacher_replay={
                "entry_id": entry["replay_id"],
                "teacher_weight_sha256": entry["teacher_weight_sha256"],
                "behavior_weight_version": trajectory.weight_version,
                "pool_sha256": self.identity["pool_sha256"],
                "original_slot_text_hash": original.get("text_hash"),
            },
        )
        return sample
