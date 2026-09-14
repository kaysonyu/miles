from __future__ import annotations

import pytest
import torch
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.dot_product_attention import DotProductAttention

from miles.policies.moss_tts_local.debug import build_moss_tts_local_debug_shard
from miles.policies.moss_tts_local.model import _enable_packed_thd_attention

NUM_GPUS = 0


def test_packed_thd_replaces_only_local_core_attention():
    spec = get_gpt_layer_local_spec(normalization="RMSNorm")
    attention = spec.submodules.self_attention.submodules
    original_linear_qkv = attention.linear_qkv
    original_linear_proj = attention.linear_proj
    assert attention.core_attention is DotProductAttention

    converted = _enable_packed_thd_attention(spec)

    assert converted is spec
    assert attention.core_attention is TEDotProductAttention
    assert attention.linear_qkv is original_linear_qkv
    assert attention.linear_proj is original_linear_proj


def test_packed_thd_spec_conversion_is_idempotent():
    spec = get_gpt_layer_local_spec(normalization="RMSNorm")

    assert _enable_packed_thd_attention(_enable_packed_thd_attention(spec)) is spec


def test_structured_debug_shard_preserves_packing_and_rollout_join_key():
    rollout_data = {
        "partition": [4, 1],
        "micro_batch_indices": [[0, 1]],
        "num_microbatches": [1],
        "num_rollouts": [2],
        "sample_indices": [40, 10],
        "rollout_ids": [4, 1],
        "weight_versions": ["7", "7"],
        "rewards": [0.25, -0.5],
        "trainer_joint_logprobs": [torch.tensor([-1.0]), torch.tensor([-2.0, -3.0])],
    }

    shard = build_moss_tts_local_debug_shard(rollout_data, rank=3, data_parallel_rank=1)

    assert shard["rank"] == 3
    assert shard["layout"]["micro_batch_indices"] == [[0, 1]]
    assert [sample["rollout_position"] for sample in shard["samples"]] == [4, 1]
    assert [sample["sample_indices"] for sample in shard["samples"]] == [40, 10]
    assert shard["samples"][1]["trainer_joint_logprobs"].tolist() == [-2.0, -3.0]
