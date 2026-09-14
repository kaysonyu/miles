"""Megatron-native MOSS-TTS Local policy and selected-action replay."""

from __future__ import annotations

from typing import Any

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.torch_norm import WrappedTorchNorm

from miles.policies.moss_tts_local.batch import MossTTSLocalPolicyBatch
from miles.policies.moss_tts_local.local_transformer import MossTTSLocalTrainingTransformer
from miles.policies.moss_tts_local.policy import MossTTSLocalPolicyOutput, replay_local_actions
from miles.policies.moss_tts_local.spec import MOSS_TTS_LOCAL_SPEC


def _mark_replicated_parameters(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.tensor_model_parallel = False
        parameter.partition_dim = -1
        parameter.partition_stride = 1
        parameter.parallel_mode = "duplicated"


def _enable_packed_thd_attention(transformer_layer_spec):
    """Keep the local-weight layout but replace its non-THD attention core.

    Megatron's local ``DotProductAttention`` rejects ``PackedSeqParams`` even
    when ``--attention-backend=flash`` is selected.  Transformer Engine's core
    attention is weightless and accepts the same Q/K/V tensors, so replacing
    only that leaf preserves every checkpoint key and leaves projections,
    normalization, and MLPs on the already validated local implementation.
    """

    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.transformer.dot_product_attention import DotProductAttention

    try:
        attention_submodules = transformer_layer_spec.submodules.self_attention.submodules
        core_attention = attention_submodules.core_attention
    except AttributeError as exc:
        raise ValueError("MOSS-TTS Local packed THD requires a GPT self-attention layer spec.") from exc
    if core_attention is DotProductAttention:
        attention_submodules.core_attention = TEDotProductAttention
    elif core_attention is not TEDotProductAttention:
        raise ValueError(
            "MOSS-TTS Local packed THD only supports Megatron local or Transformer Engine attention specs, "
            f"got {core_attention!r}."
        )
    return transformer_layer_spec


class MossTTSLocalMegatronModel(MegatronModule):
    """Qwen3 global decoder plus differentiable frame-local action decoder."""

    def __init__(
        self,
        *,
        config,
        transformer_layer_spec,
        args,
        pre_process: bool,
        post_process: bool,
        vp_stage: int | None = None,
    ) -> None:
        super().__init__(config=config)
        if not pre_process or not post_process:
            raise ValueError("MOSS-TTS Local P0 requires PP=1 so every model chunk owns pre/post processing.")
        self.pre_process = pre_process
        self.post_process = post_process
        self.args = args
        self.temperature = float(args.rollout_temperature)
        self.packed_thd = bool(getattr(args, "moss_local_packed_thd", False))
        if self.packed_thd:
            transformer_layer_spec = _enable_packed_thd_attention(transformer_layer_spec)
        if int(config.hidden_size) != MOSS_TTS_LOCAL_SPEC.hidden_size:
            raise ValueError(
                f"MOSS-TTS Local hidden size mismatch: expected {MOSS_TTS_LOCAL_SPEC.hidden_size}, "
                f"got {config.hidden_size}."
            )

        self.language_model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=True,
            post_process=False,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=False,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            **({"vp_stage": vp_stage} if vp_stage is not None else {}),
        )
        # GPTModel(post_process=False) is required to return hidden states
        # instead of allocating/projecting a 151936-way global LM head, but
        # Megatron also omits the decoder's final RMSNorm in that mode.  MOSS
        # Local consumes the checkpoint's normalized global hidden state, so
        # restore only that norm from the block spec.  TransformerBlock.forward
        # applies any non-None final_layernorm even while GPTModel itself keeps
        # post_process=False.
        if self.language_model.decoder.final_layernorm is None:
            if args.transformer_impl == "local":
                self.language_model.decoder.final_layernorm = WrappedTorchNorm(
                    config=config,
                    hidden_size=config.hidden_size,
                    eps=config.layernorm_epsilon,
                )
            else:
                final_norm_spec = self.language_model.decoder.submodules.layer_norm
                if final_norm_spec is None:
                    raise ValueError("MOSS-TTS Local global decoder spec does not provide a final layer norm.")
                self.language_model.decoder.final_layernorm = build_module(
                    final_norm_spec,
                    config=config,
                    hidden_size=config.hidden_size,
                    eps=config.layernorm_epsilon,
                )
        self.audio_embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(
                    MOSS_TTS_LOCAL_SPEC.audio_vocab_size,
                    config.hidden_size,
                )
                for _ in range(MOSS_TTS_LOCAL_SPEC.n_vq)
            ]
        )
        # mossLite checkpoints use split_v1 storage: teacher-forcing and
        # next-frame feedback consume audio_embeddings, while RVQ logits are
        # projected by independent audio_lm_heads.  These tensors are
        # intentionally not tied; the latest pretraining checkpoint's heads
        # and embeddings are numerically unrelated.
        self.audio_lm_heads = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    config.hidden_size,
                    MOSS_TTS_LOCAL_SPEC.audio_vocab_size,
                    bias=False,
                )
                for _ in range(MOSS_TTS_LOCAL_SPEC.n_vq)
            ]
        )
        self.local_transformer = MossTTSLocalTrainingTransformer(
            hidden_size=config.hidden_size,
            num_heads=int(getattr(args, "moss_local_num_attention_heads", 32)),
            inner_size=int(getattr(args, "moss_local_ffn_hidden_size", MOSS_TTS_LOCAL_SPEC.local_ffn_hidden_size)),
            num_layers=MOSS_TTS_LOCAL_SPEC.local_layers,
            max_positions=MOSS_TTS_LOCAL_SPEC.n_vq + 1,
            rope_base=float(getattr(args, "moss_local_rotary_base", 1_000_000.0)),
            layer_norm_eps=float(getattr(args, "moss_local_layernorm_epsilon", 1e-6)),
        )
        self.local_text_lm_head = torch.nn.Linear(config.hidden_size, 2, bias=False)
        parameter_dtype = getattr(config, "params_dtype", None)
        if parameter_dtype is not None:
            self.audio_embeddings.to(dtype=parameter_dtype)
            self.audio_lm_heads.to(dtype=parameter_dtype)
            self.local_transformer.to(dtype=parameter_dtype)
            self.local_text_lm_head.to(dtype=parameter_dtype)
        _mark_replicated_parameters(self.audio_embeddings)
        _mark_replicated_parameters(self.audio_lm_heads)
        _mark_replicated_parameters(self.local_transformer)
        _mark_replicated_parameters(self.local_text_lm_head)
        trainable_prefixes = {
            "full": None,
            "local_only": ("local_transformer.", "local_text_lm_head."),
            "local_audio": ("local_transformer.", "local_text_lm_head.", "audio_embeddings.", "audio_lm_heads."),
        }[args.moss_local_trainable_scope]
        if trainable_prefixes is not None:
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith(trainable_prefixes))

    @property
    def decoder(self):
        return self.language_model.decoder

    def set_input_tensor(self, input_tensor) -> None:
        self.language_model.set_input_tensor(input_tensor)

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    @staticmethod
    def _masked_audio_embedding(embedding: torch.nn.Embedding, audio_ids: torch.Tensor) -> torch.Tensor:
        """Embed real RVQ ids while mapping the out-of-vocab pad code to zero."""

        valid = audio_ids.ne(MOSS_TTS_LOCAL_SPEC.audio_pad_code)
        safe_ids = audio_ids.masked_fill(~valid, 0)
        return embedding(safe_ids) * valid.unsqueeze(-1)

    def _global_decoder_input(
        self,
        global_rows: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        text_ids = global_rows[..., 0]
        decoder_input = self.language_model.embedding(input_ids=text_ids, position_ids=position_ids)
        for depth, embedding in enumerate(self.audio_embeddings):
            audio_ids = global_rows[..., depth + 1]
            audio_input = self._masked_audio_embedding(embedding, audio_ids).transpose(0, 1).contiguous()
            decoder_input = decoder_input + audio_input.to(dtype=decoder_input.dtype)
        return decoder_input

    def forward(
        self,
        *,
        policy_batch: MossTTSLocalPolicyBatch,
        with_entropy: bool = False,
        with_logits: bool = False,
        **_kwargs: Any,
    ) -> MossTTSLocalPolicyOutput:
        packed_seq_params = policy_batch.packed_seq_params
        if self.packed_thd != (packed_seq_params is not None):
            raise ValueError(
                "MOSS-TTS Local batch/model packed-THD mode mismatch; construct the batch through "
                "collate_moss_tts_local_batch(..., packed_thd=args.moss_local_packed_thd)."
            )
        if policy_batch.batch_size != 1 and packed_seq_params is None:
            raise ValueError("MOSS-TTS Local multi-sample batches require packed THD attention boundaries.")
        decoder_input = self._global_decoder_input(
            policy_batch.global_rows,
            policy_batch.global_position_ids,
        )
        hidden_states = self.language_model(
            input_ids=policy_batch.global_rows[..., 0],
            position_ids=policy_batch.global_position_ids,
            attention_mask=None,
            decoder_input=decoder_input,
            labels=None,
            packed_seq_params=packed_seq_params,
        )
        if hidden_states.ndim != 3:
            raise RuntimeError(f"MOSS-TTS Local global model returned shape {tuple(hidden_states.shape)}.")
        hidden_flat = hidden_states[:, 0] if hidden_states.shape[1] == 1 else hidden_states[0]
        global_hidden = hidden_flat.index_select(0, policy_batch.prediction_positions)
        return replay_local_actions(
            global_hidden=global_hidden,
            decisions=policy_batch.decisions,
            codes=policy_batch.codes,
            decision_offsets=policy_batch.decision_offsets,
            frame_offsets=policy_batch.frame_offsets,
            code_mask=policy_batch.code_mask,
            audio_embeddings=self.audio_embeddings,
            audio_lm_heads=self.audio_lm_heads,
            local_transformer=self.local_transformer,
            local_text_lm_head=self.local_text_lm_head,
            temperature=self.temperature,
            with_entropy=with_entropy,
            with_logits=with_logits,
        )
