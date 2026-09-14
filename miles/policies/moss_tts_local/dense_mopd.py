"""Full-vocabulary KL on student-generated Local trajectories.

The optional native teachers execute the same model implementation and packed
batch as the student. They are frozen, excluded from the student's optimizer,
checkpoint and refit, and pinned by checkpoint manifest identity on resume.
"""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import torch


def native_teacher_sources(entries):
    sources = {}
    for item in entries or []:
        domain, separator, path = item.partition("=")
        if not separator or not domain or domain in sources or not Path(path).is_absolute():
            raise ValueError("Native MOPD teachers must be unique DOMAIN=/absolute/iteration-directory entries")
        sources[domain] = Path(path)
    if not sources:
        raise ValueError("Native dense MOPD requires frozen checkpoint sources")
    return sources


def categorical_kl(student_logits, teacher_logits, *, temperature, direction):
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
        raise ValueError("Student and teacher distributions must have the same action/vocabulary shape")
    if temperature <= 0:
        raise ValueError("Distillation temperature must be positive")
    student = (student_logits.float() / temperature).log_softmax(-1)
    teacher = (teacher_logits.detach().float() / temperature).log_softmax(-1)
    if direction == "dense_reverse":
        return (student.exp() * (student - teacher)).sum(-1)
    if direction == "dense_forward":
        return (teacher.exp() * (teacher - student)).sum(-1)
    raise ValueError(f"Unsupported dense KL direction: {direction}")


def dense_action_kl(
    output,
    batch,
    teacher_decisions,
    teacher_codes,
    *,
    temperature,
    direction,
    prefix_frames=0,
    prefix_weight=1.0,
    codebook_weights=None,
    decision_weight=1.0,
):
    if output.decision_logits is None or output.code_logits is None:
        raise ValueError("Dense MOPD requires full student logits")
    dm, cm = batch.decision_mask.unsqueeze(-1), batch.code_mask.unsqueeze(-1)
    decisions = categorical_kl(
        torch.where(dm, output.decision_logits, 0),
        torch.where(dm, teacher_decisions, 0),
        temperature=temperature,
        direction=direction,
    )
    codes = categorical_kl(
        torch.where(cm, output.code_logits, 0),
        torch.where(cm, teacher_codes, 0),
        temperature=temperature,
        direction=direction,
    )
    book_weights = None
    if codebook_weights is not None:
        if len(codebook_weights) != codes.shape[-1] or any(not 0 < value < float("inf") for value in codebook_weights):
            raise ValueError("Codebook weights must be positive finite values, one per RVQ codebook")
        if any(value != 1.0 for value in codebook_weights):
            book_weights = codes.new_tensor(codebook_weights).reshape(1, -1)
    if not 0 < decision_weight < float("inf"):
        raise ValueError("Decision weight must be positive and finite")
    totals = decisions.new_zeros(())
    for index in range(batch.batch_size):
        ds, de = int(batch.decision_offsets[index]), int(batch.decision_offsets[index + 1])
        fs, fe = int(batch.frame_offsets[index]), int(batch.frame_offsets[index + 1])
        dm, cm = batch.decision_mask[ds:de], batch.code_mask[fs:fe]
        # Match the existing objective's equal sample weighting and action masks.
        count = decision_weight * dm.sum() + cm.sum()
        count = torch.where(count > 0, count, 1)
        code_values = torch.where(cm, codes[fs:fe], 0)
        code_sum = code_values.sum()
        if (prefix_frames and prefix_weight != 1.0) or book_weights is not None:
            weights = cm.to(code_values.dtype)
            if prefix_frames:
                weights[: min(prefix_frames, fe - fs)] *= prefix_weight
            if book_weights is not None:
                weights = weights * book_weights
            # Reweight visited RVQ frames without changing their aggregate scale
            # relative to continue/stop actions, including terminal-only samples.
            weight_sum = weights.sum()
            weight_sum = torch.where(weight_sum > 0, weight_sum, 1)
            code_sum = (code_values * weights).sum() * cm.sum() / weight_sum
        totals = totals + (decision_weight * torch.where(dm, decisions[ds:de], 0).sum() + code_sum) / count
    return totals


@torch.no_grad()
def native_action_entropies(output, batch, teacher_decisions, teacher_codes, *, temperature):
    """Observe both distributions on visited states, without changing the loss."""
    result = {}
    for label, decisions, codes in [
        ("student", output.decision_logits, output.code_logits),
        ("teacher", teacher_decisions, teacher_codes),
    ]:
        dm, cm = batch.decision_mask, batch.code_mask
        decision_logp = (torch.where(dm.unsqueeze(-1), decisions.float(), 0) / temperature).log_softmax(-1)
        code_logp = (torch.where(cm.unsqueeze(-1), codes.float(), 0) / temperature).log_softmax(-1)
        decision_entropy = -(decision_logp.exp() * decision_logp).sum(-1)
        code_entropy = -(code_logp.exp() * code_logp).sum(-1)
        total, rvq1 = decision_entropy.new_zeros(()), decision_entropy.new_zeros(())
        for index in range(batch.batch_size):
            ds, de = int(batch.decision_offsets[index]), int(batch.decision_offsets[index + 1])
            fs, fe = int(batch.frame_offsets[index]), int(batch.frame_offsets[index + 1])
            dm, cm = batch.decision_mask[ds:de], batch.code_mask[fs:fe]
            total += (
                torch.where(dm, decision_entropy[ds:de], 0).sum() + torch.where(cm, code_entropy[fs:fe], 0).sum()
            ) / (dm.sum() + cm.sum()).clamp_min(1)
            rvq1 += torch.where(cm[:, 0], code_entropy[fs:fe, 0], 0).sum() / cm[:, 0].sum().clamp_min(1)
        result[f"mopd_{label}_action_entropy"] = total
        result[f"mopd_{label}_rvq1_entropy"] = rvq1
    return result


def _model_digest(model):
    checksum = hashlib.sha256()
    for name, tensor in model.named_parameters():
        checksum.update(name.encode())
        checksum.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return checksum.hexdigest()


def get_native_teacher_pool(args, actor):
    """Cache on the long-lived actor; policy workflow objects are recreated."""
    if getattr(args, "moss_local_mopd_estimator", "sampled") == "sampled":
        return None
    pool = getattr(actor, "_moss_native_teachers", None)
    if pool is None:
        pool = NativeTeacherPool(args)
        actor._moss_native_teachers = pool
    return pool


class NativeTeacherPool:
    """Actor-owned frozen native models, initialized on every rank."""

    def __init__(self, args):
        # GPU/distributed imports stay out of CPU-only loss and argument validation.
        import torch.distributed as dist
        from megatron.core.enums import ModelType
        from megatron.core.utils import unwrap_model
        from megatron.training.training import get_model

        from miles.backends.megatron_utils.model_provider import get_model_provider_func
        from miles.policies.moss_tts_local.pretrained_checkpoint import load_moss_tts_local_pretrained_checkpoint

        self.models = {}
        self.writer = not dist.is_initialized() or dist.get_rank() == 0
        self.output = Path(args.save) / "native_mopd_teachers.json" if args.save else None
        previous = Path(args.load) / "native_mopd_teachers.json" if args.load else None
        expected = json.loads(previous.read_text()) if previous and previous.exists() else None
        domain_specs = getattr(args, "moss_local_domain_loss_specs", {})
        previous_specs = Path(args.load) / "native_domain_loss.json" if args.load else None
        if previous_specs and previous_specs.exists() and json.loads(previous_specs.read_text()) != domain_specs:
            raise ValueError("Native domain loss settings changed on resume")
        if self.output is not None and self.writer and domain_specs:
            target = self.output.with_name("native_domain_loss.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(domain_specs, indent=2))
        identities = {}
        for domain, path in sorted(native_teacher_sources(args.moss_local_native_teachers).items()):
            source = path / "checkpoint.json"
            manifest = json.loads(source.read_text())
            identity = {
                "iteration": manifest["iteration"],
                "manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            if expected is not None and expected.get(domain) != identity:
                raise ValueError(f"Native MOPD teacher changed on resume: {domain}")
            teacher_args = Namespace(**vars(args))
            teacher_args.pretrained_checkpoint = str(path.parent)
            teacher_args.ckpt_step = manifest["iteration"]
            teacher_args.finetune = True
            models = get_model(
                get_model_provider_func(teacher_args, "actor"), ModelType.encoder_or_decoder, wrap_with_ddp=False
            )
            load_moss_tts_local_pretrained_checkpoint(teacher_args, unwrap_model(models), {})
            if len(models) != 1:
                raise ValueError("Native Local teachers require one PP model chunk")
            self.models[domain] = models[0].requires_grad_(False).eval()
            identities[domain] = identity
        self.initial_digests = {key: _model_digest(model) for key, model in self.models.items()} if self.writer else {}
        if self.output is not None and self.writer:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.output.write_text(json.dumps(identities, indent=2))

    @torch.no_grad()
    def score(self, batch, domains):
        if len(domains) != batch.batch_size:
            raise ValueError("Native MOPD needs one domain per packed sample")
        selected = set(domains)
        if not selected <= self.models.keys():
            raise ValueError(f"Missing native teacher routes: {selected - self.models.keys()}")
        decision_logits = code_logits = None
        for domain in sorted(selected):
            scored = self.models[domain](policy_batch=batch, with_logits=True)
            if decision_logits is None:
                decision_logits = torch.empty_like(scored.decision_logits)
                code_logits = torch.empty_like(scored.code_logits)
            for index, sample_domain in enumerate(domains):
                if sample_domain != domain:
                    continue
                ds, de = int(batch.decision_offsets[index]), int(batch.decision_offsets[index + 1])
                fs, fe = int(batch.frame_offsets[index]), int(batch.frame_offsets[index + 1])
                decision_logits[ds:de] = scored.decision_logits[ds:de]
                code_logits[fs:fe] = scored.code_logits[fs:fe]
        return decision_logits, code_logits

    def verify_frozen(self):
        if not self.writer:
            return
        current = {key: _model_digest(model) for key, model in self.models.items()}
        result = {"before": self.initial_digests, "after": current, "matched": current == self.initial_digests}
        if self.output is not None:
            self.output.with_name("native_mopd_teacher_audit.json").write_text(json.dumps(result, indent=2))
        if not result["matched"]:
            raise RuntimeError("Native MOPD teacher parameters changed during training")
