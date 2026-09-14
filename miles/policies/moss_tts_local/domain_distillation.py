"""Stateless per-domain native KL specifications and packed-sample loss views."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from miles.policies.moss_tts_local.dense_mopd import dense_action_kl

_FIELDS = {"estimator", "prefix_frames", "prefix_weight", "codebook_weights", "decision_weight"}
_DEFAULTS = {"prefix_frames": 0, "prefix_weight": 1.0, "codebook_weights": None, "decision_weight": 1.0}


def load_specs(path):
    """Resolve and validate immutable loss settings before actor construction."""
    if path is None:
        return {}
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Domain loss configuration requires an absolute path")
    source = json.loads(path.read_text())
    if not isinstance(source, dict) or not source:
        raise ValueError("Domain loss configuration must map domain names to settings")
    specs = {}
    for domain, settings in source.items():
        if not isinstance(domain, str) or not domain or not isinstance(settings, dict) or set(settings) - _FIELDS:
            raise ValueError("Invalid domain loss settings or unknown fields")
        spec = _DEFAULTS | settings
        if spec.get("estimator") not in {"dense_reverse", "dense_forward"}:
            raise ValueError("Domain estimators require full-distribution forward or reverse KL")
        if type(spec["prefix_frames"]) is not int or spec["prefix_frames"] < 0:
            raise ValueError("Domain prefix frames must be a nonnegative integer")
        weights = [spec["prefix_weight"], spec["decision_weight"]]
        books = spec["codebook_weights"]
        if books is not None:
            if not isinstance(books, list) or len(books) != 12:
                raise ValueError("Domain codebook weights require twelve values")
            weights.extend(books)
        if any(type(value) not in {int, float} or not math.isfinite(value) or value <= 0 for value in weights):
            raise ValueError("Domain loss weights must be positive finite numbers")
        specs[domain] = spec
    return specs


def action_kl(args, output, batch, teacher_decisions, teacher_codes, domains):
    """Apply one declared objective to each sample, retaining existing sample averaging."""
    specs = getattr(args, "moss_local_domain_loss_specs", {})
    if not specs:
        return dense_action_kl(
            output,
            batch,
            teacher_decisions,
            teacher_codes,
            temperature=args.rollout_temperature,
            direction=args.moss_local_mopd_estimator,
            prefix_frames=getattr(args, "moss_local_mopd_prefix_frames", 0),
            prefix_weight=getattr(args, "moss_local_mopd_prefix_weight", 4.0),
            codebook_weights=getattr(args, "moss_local_mopd_codebook_weights", None),
            decision_weight=getattr(args, "moss_local_mopd_decision_weight", 1.0),
        )
    if domains is None or len(domains) != batch.batch_size or set(domains) - set(specs):
        raise ValueError("Every packed sample needs a matching declared domain loss")
    total = output.decision_logits.new_zeros((), dtype=torch.float32)
    for index, domain in enumerate(domains):
        ds, de = int(batch.decision_offsets[index]), int(batch.decision_offsets[index + 1])
        fs, fe = int(batch.frame_offsets[index]), int(batch.frame_offsets[index + 1])
        view = SimpleNamespace(
            batch_size=1,
            decision_offsets=(0, de - ds),
            frame_offsets=(0, fe - fs),
            decision_mask=batch.decision_mask[ds:de],
            code_mask=batch.code_mask[fs:fe],
        )
        logits = SimpleNamespace(decision_logits=output.decision_logits[ds:de], code_logits=output.code_logits[fs:fe])
        spec = specs[domain]
        total = total + dense_action_kl(
            logits,
            view,
            teacher_decisions[ds:de],
            teacher_codes[fs:fe],
            temperature=args.rollout_temperature,
            direction=spec["estimator"],
            prefix_frames=spec["prefix_frames"],
            prefix_weight=spec["prefix_weight"],
            codebook_weights=spec["codebook_weights"],
            decision_weight=spec["decision_weight"],
        )
    return total
