from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from miles.policies.moss_tts_local.batch import collate_moss_tts_local_batch
from miles.policies.moss_tts_local.local_transformer import MossTTSLocalTrainingTransformer, _rotate_half_interleaved
from miles.policies.moss_tts_local.loss import expand_sample_advantages, frame_joint_grpo_loss
from miles.policies.moss_tts_local.policy import replay_local_actions
from miles.policies.moss_tts_local.selected_logprob import join_moss_action_logprobs, selected_logprobs

NUM_GPUS = 0


def _raw_sample(*, prompt_length: int, frames: int, terminal: bool, reward: float, rollout_id: int):
    decisions = torch.zeros(frames + int(terminal), dtype=torch.long)
    if terminal:
        decisions[-1] = 1
    return {
        "prompt_rows": torch.zeros((prompt_length, 13), dtype=torch.long),
        "decisions": decisions,
        "decision_masks": torch.ones_like(decisions, dtype=torch.bool),
        "codes": torch.arange(frames * 12, dtype=torch.long).reshape(frames, 12),
        "code_masks": torch.ones((frames, 12), dtype=torch.bool),
        "rollout_decision_logprobs": torch.full((len(decisions),), -0.5),
        "rollout_code_logprobs": torch.full((frames, 12), -1.0),
        "rewards": reward,
        "rollout_ids": rollout_id,
        "weight_versions": "7",
    }


def _raw_batch(*samples):
    return {key: [sample[key] for sample in samples] for key in samples[0]}


def test_collator_preserves_global_prediction_geometry_for_stop_and_length():
    stop = _raw_sample(prompt_length=3, frames=2, terminal=True, reward=1.0, rollout_id=0)
    length = _raw_sample(prompt_length=2, frames=2, terminal=False, reward=-1.0, rollout_id=1)

    batch = collate_moss_tts_local_batch(_raw_batch(stop, length))

    # First sample occupies global positions 0..4 and predicts at 2,3,4.
    # Second occupies positions 5..8 and predicts at 6,7 (no synthetic terminal).
    assert batch.global_rows.shape == (1, 9, 13)
    assert batch.prediction_positions.tolist() == [2, 3, 4, 6, 7]
    assert batch.decision_offsets.tolist() == [0, 3, 5]
    assert batch.frame_offsets.tolist() == [0, 2, 4]
    assert batch.global_position_ids.tolist() == [[0, 1, 2, 3, 4, 0, 1, 2, 3]]
    assert batch.global_rows[0, 3:, 0].tolist()[:2] == [151656, 151656]
    assert batch.global_lengths.tolist() == [5, 4]
    assert batch.packed_seq_params is None


def test_collator_builds_thd_boundaries_for_ragged_samples():
    first = _raw_sample(prompt_length=3, frames=2, terminal=True, reward=1.0, rollout_id=0)
    second = _raw_sample(prompt_length=2, frames=2, terminal=False, reward=-1.0, rollout_id=1)

    batch = collate_moss_tts_local_batch(_raw_batch(first, second), packed_thd=True)

    packed = batch.packed_seq_params
    assert packed is not None
    assert packed.qkv_format == "thd"
    assert packed.cu_seqlens_q.tolist() == [0, 5, 9]
    assert packed.cu_seqlens_kv is packed.cu_seqlens_q
    assert packed.cu_seqlens_q.dtype == torch.int32
    assert packed.max_seqlen_q == 5
    assert packed.max_seqlen_kv == 5


def test_collator_handles_first_step_stop_without_codes():
    sample = _raw_sample(prompt_length=2, frames=0, terminal=True, reward=1.0, rollout_id=0)

    batch = collate_moss_tts_local_batch(_raw_batch(sample))

    assert batch.global_rows.shape == (1, 2, 13)
    assert batch.prediction_positions.tolist() == [1]
    assert batch.codes.shape == (0, 12)
    assert batch.server_joint_logprobs().tolist() == [-0.5]


def test_single_sample_server_joint_logprobs_include_every_code_channel():
    sample = _raw_sample(
        prompt_length=2,
        frames=2,
        terminal=True,
        reward=1.0,
        rollout_id=0,
    )

    batch = collate_moss_tts_local_batch(_raw_batch(sample))

    assert batch.server_joint_logprobs().tolist() == pytest.approx([-12.5, -12.5, -0.5])


def test_selected_logprobs_match_full_vocab_temperature_distribution():
    logits = torch.tensor([[0.0, 1.0, 2.0], [3.0, 0.0, -1.0]])
    actions = torch.tensor([2, 0])

    actual = selected_logprobs(logits, actions, temperature=2.0)
    expected = torch.log_softmax(logits / 2.0, dim=-1).gather(-1, actions[:, None]).squeeze(-1)

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32


def test_joint_logprob_adds_codes_to_continue_but_not_terminal():
    decision = torch.tensor([-0.1, -0.2, -0.3])
    codes = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])

    joint = join_moss_action_logprobs(
        decision,
        codes,
        decision_offsets=torch.tensor([0, 3]),
        frame_offsets=torch.tensor([0, 2]),
    )

    assert joint.tolist() == pytest.approx([-3.1, -7.2, -0.3])


def test_frame_joint_loss_ratio_one_has_zero_clip_and_kl():
    current = torch.tensor([-3.0, -1.0, -2.0], requires_grad=True)
    old = current.detach().clone()
    offsets = torch.tensor([0, 2, 3])
    advantages = expand_sample_advantages(torch.tensor([2.0, -1.0]), offsets)

    output = frame_joint_grpo_loss(
        current,
        old,
        advantages,
        torch.ones(3, dtype=torch.bool),
        offsets,
        eps_clip=0.2,
    )

    # sample means: first=-2, second=1; batch mean=-0.5
    assert output.loss.item() == pytest.approx(-0.5)
    assert output.clip_fraction.item() == 0.0
    assert output.approx_kl.item() == 0.0
    assert output.ratio_mean.item() == 1.0
    output.loss.backward()
    assert torch.isfinite(current.grad).all()


def test_frame_joint_loss_weights_short_and_long_samples_equally():
    current = torch.zeros(5, requires_grad=True)
    old = torch.zeros(5)
    offsets = torch.tensor([0, 1, 5])
    advantages = expand_sample_advantages(torch.tensor([1.0, 3.0]), offsets)

    output = frame_joint_grpo_loss(
        current,
        old,
        advantages,
        torch.ones(5, dtype=torch.bool),
        offsets,
        eps_clip=0.2,
    )

    assert output.loss.item() == pytest.approx(-2.0)


def test_sample_sum_reduction_is_invariant_to_dynamic_microbatch_grouping():
    advantages = [1.0, 3.0, -2.0]

    def loss_for(samples):
        offsets = [0]
        current_parts = []
        advantage_parts = []
        for event_count, advantage in samples:
            current_parts.append(torch.zeros(event_count))
            advantage_parts.append(torch.full((event_count,), advantage))
            offsets.append(offsets[-1] + event_count)
        current = torch.cat(current_parts).requires_grad_(True)
        output = frame_joint_grpo_loss(
            current,
            torch.zeros_like(current),
            torch.cat(advantage_parts),
            torch.ones_like(current, dtype=torch.bool),
            torch.tensor(offsets),
            eps_clip=0.2,
            sample_reduction="sum",
        )
        return output.loss

    samples = [(1, advantages[0]), (4, advantages[1]), (2, advantages[2])]
    one_microbatch = loss_for(samples)
    dynamic_microbatches = loss_for(samples[:2]) + loss_for(samples[2:])

    assert one_microbatch.item() == pytest.approx(-sum(advantages))
    assert dynamic_microbatches.item() == pytest.approx(one_microbatch.item())


def test_single_sample_all_masked_loss_is_zero_without_host_branch():
    current = torch.tensor([-3.0, -1.0], requires_grad=True)
    output = frame_joint_grpo_loss(
        current,
        current.detach().clone(),
        torch.ones(2),
        torch.zeros(2, dtype=torch.bool),
        torch.tensor([0, 2]),
        eps_clip=0.2,
    )

    assert output.loss.item() == 0.0
    output.loss.backward()
    assert torch.equal(current.grad, torch.zeros_like(current))


def _incremental_reference(module: MossTTSLocalTrainingTransformer, inputs: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, _ = inputs.shape
    key_caches = [[] for _ in module.h]
    value_caches = [[] for _ in module.h]
    outputs = []
    for position in range(sequence_length):
        x = inputs[:, position]
        cos = module.rope_cos[position].to(dtype=x.dtype, device=x.device).view(1, 1, module.head_dim)
        sin = module.rope_sin[position].to(dtype=x.dtype, device=x.device).view(1, 1, module.head_dim)
        for layer_index, block in enumerate(module.h):
            normed = block.ln_1(x)
            query, key, value = block.attn.c_attn(normed).split(module.hidden_size, dim=-1)
            query = query.view(batch_size, module.num_heads, module.head_dim)
            key = key.view(batch_size, module.num_heads, module.head_dim)
            value = value.view(batch_size, module.num_heads, module.head_dim)
            query = query * cos + _rotate_half_interleaved(query) * sin
            key = key * cos + _rotate_half_interleaved(key) * sin
            key_caches[layer_index].append(key)
            value_caches[layer_index].append(value)
            keys = torch.stack(key_caches[layer_index], dim=2)
            values = torch.stack(value_caches[layer_index], dim=2)
            attention = F.scaled_dot_product_attention(query.unsqueeze(2), keys, values).squeeze(2)
            x = x + block.attn.c_proj(attention.reshape(batch_size, module.hidden_size))
            x = x + block.mlp(block.ln_2(x))
        outputs.append(module.ln_f(x))
    return torch.stack(outputs, dim=1)


def test_full_local_teacher_forcing_matches_incremental_causal_semantics():
    torch.manual_seed(7)
    module = MossTTSLocalTrainingTransformer(
        hidden_size=16,
        num_heads=4,
        inner_size=32,
        num_layers=2,
        max_positions=13,
        rope_base=1_000_000.0,
    ).double()
    inputs = torch.randn(3, 6, 16, dtype=torch.double, requires_grad=True)

    full = module(inputs)
    incremental = _incremental_reference(module, inputs)

    torch.testing.assert_close(full, incremental, rtol=1e-10, atol=1e-10)
    full.sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_local_action_replay_returns_selected_values_and_terminal_has_no_codes():
    torch.manual_seed(11)
    hidden_size = 16
    audio_embeddings = torch.nn.ModuleList([torch.nn.Embedding(1024, hidden_size) for _ in range(12)])
    audio_lm_heads = torch.nn.ModuleList([torch.nn.Linear(hidden_size, 1024, bias=False) for _ in range(12)])
    local_transformer = MossTTSLocalTrainingTransformer(
        hidden_size=hidden_size,
        num_heads=4,
        inner_size=32,
        num_layers=1,
        max_positions=13,
        rope_base=1_000_000.0,
    )
    decision_head = torch.nn.Linear(hidden_size, 2, bias=False)
    global_hidden = torch.randn(3, hidden_size, requires_grad=True)
    decisions = torch.tensor([0, 0, 1])
    codes = torch.randint(0, 1024, (2, 12))

    output = replay_local_actions(
        global_hidden=global_hidden,
        decisions=decisions,
        codes=codes,
        decision_offsets=torch.tensor([0, 3]),
        frame_offsets=torch.tensor([0, 2]),
        code_mask=torch.ones(2, 12, dtype=torch.bool),
        audio_embeddings=audio_embeddings,
        audio_lm_heads=audio_lm_heads,
        local_transformer=local_transformer,
        local_text_lm_head=decision_head,
        temperature=1.0,
    )

    assert output.decision_logprobs.shape == (3,)
    assert output.code_logprobs.shape == (2, 12)
    assert output.joint_logprobs.shape == (3,)
    assert output.joint_logprobs[-1] == output.decision_logprobs[-1]
    output.joint_logprobs.sum().backward()
    assert torch.isfinite(global_hidden.grad).all()


def test_local_action_replay_projects_logits_with_untied_heads():
    torch.manual_seed(17)
    hidden_size = 16
    audio_embeddings = torch.nn.ModuleList([torch.nn.Embedding(1024, hidden_size) for _ in range(12)])
    audio_lm_heads = torch.nn.ModuleList([torch.nn.Linear(hidden_size, 1024, bias=False) for _ in range(12)])
    global_hidden = torch.randn(1, hidden_size)
    decisions = torch.tensor([0])
    codes = torch.randint(0, 1024, (1, 12))

    output = replay_local_actions(
        global_hidden=global_hidden,
        decisions=decisions,
        codes=codes,
        decision_offsets=torch.tensor([0, 1]),
        frame_offsets=torch.tensor([0, 1]),
        code_mask=torch.ones(1, 12, dtype=torch.bool),
        audio_embeddings=audio_embeddings,
        audio_lm_heads=audio_lm_heads,
        local_transformer=torch.nn.Identity(),
        local_text_lm_head=torch.nn.Linear(hidden_size, 2, bias=False),
        temperature=1.0,
    )

    action = codes[0, 0]
    head_expected = torch.log_softmax(F.linear(global_hidden, audio_lm_heads[0].weight).float(), dim=-1)[0, action]
    tied_embedding_value = torch.log_softmax(
        F.linear(global_hidden, audio_embeddings[0].weight[:1024]).float(), dim=-1
    )[0, action]
    torch.testing.assert_close(output.code_logprobs[0, 0], head_expected)
    assert not torch.isclose(output.code_logprobs[0, 0], tied_embedding_value)
