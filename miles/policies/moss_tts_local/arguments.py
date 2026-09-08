"""Opt-in arguments and correctness boundaries for MOSS-TTS Local RL."""

import argparse


def add_arguments(parser):
    group = parser.add_argument_group("MOSS-TTS Local")
    group.add_argument("--policy-family", choices=["text", "moss_tts_local"], default="text")
    group.add_argument("--rollout-backend", choices=["sglang", "sglang_omni"], default="sglang")
    group.add_argument("--sglang-omni-endpoints", nargs="+")
    group.add_argument("--sglang-omni-train-stage", default="tts_engine")
    group.add_argument("--sglang-omni-admin-api-key-env")
    group.add_argument("--structured-rollout-version", type=int, default=2)
    group.add_argument("--moss-local-trainable-scope", choices=["full", "local_audio", "local_only"], default="full")
    group.add_argument(
        "--moss-local-old-policy-source", choices=["trainer_preupdate", "server_behavior"], default="trainer_preupdate"
    )
    group.add_argument("--moss-local-logprob-parity-max-abs", type=float, default=1e-3)
    group.add_argument("--moss-local-reuse-train-forward", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--moss-local-packed-thd", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--moss-local-num-attention-heads", type=int, default=32)
    group.add_argument("--moss-local-ffn-hidden-size", type=int, default=9728)
    group.add_argument("--moss-local-rotary-base", type=float, default=1_000_000.0)
    group.add_argument("--moss-local-layernorm-epsilon", type=float, default=1e-6)
    group.add_argument("--max-samples-per-microbatch", type=int)
    group.add_argument("--use-torch-adam", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--custom-pretrained-checkpoint-loader-path")
    return parser


def validate_args(args):
    cap = args.max_samples_per_microbatch
    if cap is not None and (cap <= 0 or not args.use_dynamic_batch_size):
        raise ValueError("--max-samples-per-microbatch requires a positive value and dynamic batching")
    if args.policy_family != "moss_tts_local":
        return
    if args.train_backend != "megatron" or args.rollout_backend != "sglang_omni":
        raise ValueError("MOSS requires Megatron training and --rollout-backend=sglang_omni")
    if not args.debug_train_only and not args.sglang_omni_endpoints:
        raise ValueError("MOSS requires pre-launched --sglang-omni-endpoints")
    for field in [
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
    ]:
        if getattr(args, field, 1) != 1:
            raise ValueError(f"MOSS currently requires {field}=1; scale with data parallelism")
    for field in [
        "colocate",
        "use_critic",
        "use_kl_loss",
        "partial_rollout",
        "enable_mtp_training",
        "use_opd",
        "use_tis",
        "use_fault_tolerance",
        "indep_dp",
        "calculate_per_token_loss",
        "balance_by_flops",
        "sequence_parallel",
    ]:
        if getattr(args, field, False):
            raise ValueError(f"MOSS does not support {field}")
    if args.kl_coef or args.entropy_coef or args.attention_dropout or args.hidden_dropout:
        raise ValueError("MOSS baseline requires zero dropout, reference KL and entropy coefficients")
    if args.rollout_top_p != 1.0 or args.rollout_top_k != -1:
        raise ValueError("MOSS selected-action replay requires top_p=1 and top_k=-1")
    if args.delay_split_train_data_by_dp:
        raise ValueError("MOSS requires rollout-side DP scheduling")
    if args.object_store_backend != "ray":
        raise ValueError("MOSS structured rollout transport currently supports the Ray object store")
    if args.update_weight_transfer_mode != "broadcast":
        raise ValueError("MOSS currently uses full NCCL broadcast weight updates")
    if args.use_dynamic_batch_size and not args.moss_local_packed_thd:
        raise ValueError("MOSS dynamic batching requires --moss-local-packed-thd")
    if args.micro_batch_size != 1 and not args.moss_local_packed_thd:
        raise ValueError("MOSS multi-sample replay requires packed THD")
    if args.use_dynamic_batch_size and cap is None:
        args.max_samples_per_microbatch = 2
    args.rollout_external = True
    args.rollout_num_gpus = len(args.sglang_omni_endpoints or [])
    args.rollout_num_gpus_per_engine = 1
    args.rollout_function_path = "miles.policies.moss_tts_local.rollout.generate_rollout"
    args.eval_function_path = args.rollout_function_path
    args.custom_rollout_log_function_path = "miles.policies.moss_tts_local.metrics.log_rollout"
    args.megatron_to_hf_mode = "raw"
    args.custom_pretrained_checkpoint_loader_path = (
        "miles.policies.moss_tts_local.pretrained_checkpoint.load_moss_tts_local_pretrained_checkpoint"
    )
    if args.eval_interval is not None or args.eval_num_gpus:
        raise ValueError("Use the dedicated MOSS held-out evaluator instead of the text eval fleet")
    if args.offload_rollout:
        raise ValueError("External Omni owns its memory lifecycle; disable rollout offload")
