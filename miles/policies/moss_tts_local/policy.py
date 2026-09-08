"""Dependency-light local-depth policy replay used by trainer and CPU tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from miles.policies.moss_tts_local.local_transformer import MossTTSLocalTrainingTransformer
from miles.policies.moss_tts_local.selected_logprob import join_moss_action_logprobs, selected_logprobs
from miles.policies.moss_tts_local.spec import MOSS_TTS_LOCAL_SPEC


@dataclass
class MossTTSLocalPolicyOutput:
    decision_logprobs: torch.Tensor
    code_logprobs: torch.Tensor
    joint_logprobs: torch.Tensor
    decision_entropy: torch.Tensor | None = None
    code_entropy: torch.Tensor | None = None


def replay_local_actions(
    *,
    global_hidden: torch.Tensor,
    decisions: torch.Tensor,
    codes: torch.Tensor,
    decision_offsets: torch.Tensor,
    frame_offsets: torch.Tensor,
    code_mask: torch.Tensor,
    audio_embeddings: torch.nn.ModuleList,
    audio_lm_heads: torch.nn.ModuleList,
    local_transformer: MossTTSLocalTrainingTransformer,
    local_text_lm_head: torch.nn.Linear,
    temperature: float,
    with_entropy: bool = False,
) -> MossTTSLocalPolicyOutput:
    """Replay global-time decisions and depth-autoregressive codes under teacher forcing."""

    if global_hidden.ndim != 2 or global_hidden.shape[0] != decisions.numel():
        raise ValueError(
            f"global_hidden must be [sum_decisions, hidden], got {tuple(global_hidden.shape)} "
            f"for {decisions.numel()} decisions."
        )
    if len(audio_embeddings) != MOSS_TTS_LOCAL_SPEC.n_vq:
        raise ValueError(f"Expected {MOSS_TTS_LOCAL_SPEC.n_vq} MOSS audio embedding tables.")
    if len(audio_lm_heads) != MOSS_TTS_LOCAL_SPEC.n_vq:
        raise ValueError(f"Expected {MOSS_TTS_LOCAL_SPEC.n_vq} MOSS audio output heads.")
    num_decisions = decisions.numel()
    local_inputs = global_hidden.new_zeros((num_decisions, MOSS_TTS_LOCAL_SPEC.n_vq, global_hidden.shape[-1]))
    local_inputs[:, 0] = global_hidden

    for sample_index in range(decision_offsets.numel() - 1):
        decision_start = int(decision_offsets[sample_index])
        frame_start = int(frame_offsets[sample_index])
        frame_end = int(frame_offsets[sample_index + 1])
        num_frames = frame_end - frame_start
        for depth in range(1, MOSS_TTS_LOCAL_SPEC.n_vq):
            if num_frames:
                previous_codes = codes[frame_start:frame_end, depth - 1]
                local_inputs[decision_start : decision_start + num_frames, depth] = audio_embeddings[depth - 1](
                    previous_codes
                )

    local_hidden = local_transformer(local_inputs)
    decision_logits = local_text_lm_head(local_hidden[:, 0]).float()
    decision_logprobs = selected_logprobs(decision_logits, decisions, temperature=temperature)

    code_logprob_columns = []
    code_entropy_columns = []
    for depth, head in enumerate(audio_lm_heads):
        frame_hidden = []
        for sample_index in range(decision_offsets.numel() - 1):
            decision_start = int(decision_offsets[sample_index])
            frame_start = int(frame_offsets[sample_index])
            frame_end = int(frame_offsets[sample_index + 1])
            num_frames = frame_end - frame_start
            if num_frames:
                frame_hidden.append(local_hidden[decision_start : decision_start + num_frames, depth])
        hidden_at_depth = (
            torch.cat(frame_hidden, dim=0) if frame_hidden else local_hidden.new_empty((0, local_hidden.shape[-1]))
        )
        logits = F.linear(hidden_at_depth, head.weight).float()
        code_logprob_columns.append(selected_logprobs(logits, codes[:, depth], temperature=temperature))
        if with_entropy:
            distribution = torch.softmax(logits / float(temperature), dim=-1)
            log_distribution = torch.log_softmax(logits / float(temperature), dim=-1)
            code_entropy_columns.append(-(distribution * log_distribution).sum(dim=-1))
    code_logprobs = torch.stack(code_logprob_columns, dim=-1)
    joint_logprobs = join_moss_action_logprobs(
        decision_logprobs,
        code_logprobs,
        decision_offsets=decision_offsets,
        frame_offsets=frame_offsets,
        code_mask=code_mask,
    )

    decision_entropy = None
    code_entropy = None
    if with_entropy:
        decision_distribution = torch.softmax(decision_logits / float(temperature), dim=-1)
        decision_log_distribution = torch.log_softmax(decision_logits / float(temperature), dim=-1)
        decision_entropy = -(decision_distribution * decision_log_distribution).sum(dim=-1)
        code_entropy = torch.stack(code_entropy_columns, dim=-1)
    return MossTTSLocalPolicyOutput(
        decision_logprobs=decision_logprobs,
        code_logprobs=code_logprobs,
        joint_logprobs=joint_logprobs,
        decision_entropy=decision_entropy,
        code_entropy=code_entropy,
    )
