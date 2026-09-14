"""Differentiable full-sequence MOSS-TTS Local depth transformer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).reshape_as(x)


class MossTTSLocalMLP(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(hidden_size, inner_size)
        self.fc_out = nn.Linear(inner_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.silu(self.fc_in(hidden_states)))


class MossTTSLocalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} is not divisible by num_heads={num_heads}.")
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_size // num_heads)
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)


class MossTTSLocalBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, inner_size: int, layer_norm_eps: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = MossTTSLocalAttention(hidden_size, num_heads)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = MossTTSLocalMLP(hidden_size, inner_size)


class MossTTSLocalTrainingTransformer(nn.Module):
    """Causal local-depth transformer with checkpoint-compatible parameter names."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        inner_size: int,
        num_layers: int,
        max_positions: int,
        rope_base: float,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.max_positions = int(max_positions)
        self.h = nn.ModuleList(
            [MossTTSLocalBlock(hidden_size, num_heads, inner_size, layer_norm_eps) for _ in range(int(num_layers))]
        )
        self.ln_f = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        inv_freq = 1.0 / (float(rope_base) ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        positions = torch.arange(self.max_positions, dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        self.register_buffer("rope_cos", frequencies.cos().repeat_interleave(2, dim=-1), persistent=False)
        self.register_buffer("rope_sin", frequencies.sin().repeat_interleave(2, dim=-1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"MOSS-TTS Local training transformer expects [batch, depth, hidden], got {tuple(hidden_states.shape)}."
            )
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size or sequence_length > self.max_positions:
            raise ValueError(
                f"Invalid local input shape {tuple(hidden_states.shape)} for hidden={self.hidden_size}, "
                f"max_positions={self.max_positions}."
            )
        cos = self.rope_cos[:sequence_length].to(device=hidden_states.device, dtype=hidden_states.dtype)
        sin = self.rope_sin[:sequence_length].to(device=hidden_states.device, dtype=hidden_states.dtype)
        cos = cos.view(1, 1, sequence_length, self.head_dim)
        sin = sin.view(1, 1, sequence_length, self.head_dim)

        x = hidden_states
        for block in self.h:
            normed = block.ln_1(x)
            query, key, value = block.attn.c_attn(normed).split(self.hidden_size, dim=-1)
            query = query.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
            key = key.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
            value = value.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
            query = query * cos + _rotate_half_interleaved(query) * sin
            key = key * cos + _rotate_half_interleaved(key) * sin
            attention = F.scaled_dot_product_attention(query, key, value, is_causal=True)
            attention = attention.transpose(1, 2).reshape(batch_size, sequence_length, self.hidden_size)
            x = x + block.attn.c_proj(attention)
            x = x + block.mlp(block.ln_2(x))
        return self.ln_f(x)
