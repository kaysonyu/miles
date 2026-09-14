"""Opt-in arguments and correctness boundaries for MOSS-TTS Local RL."""

import argparse
import math

from miles.policies.moss_tts_local import domain_distillation
from miles.policies.moss_tts_local.async_policy import validate_configuration
from miles.policies.moss_tts_local.reward_components import parse_components


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
        "--moss-local-old-policy-source",
        choices=["trainer_preupdate", "server_behavior", "trainer_behavior"],
        default="trainer_preupdate",
    )
    group.add_argument("--moss-local-async", action="store_true", help="Opt into one-batch-ahead Local WER training")
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
    group.add_argument("--moss-local-objective", choices=["wer_grpo", "mopd"], default="wer_grpo")
    group.add_argument(
        "--moss-local-reward-components",
        default="wer=1.0",
        help="Opt-in composite reward, e.g. 'wer=0.4 sim=0.4 rm=0.2'.",
    )
    group.add_argument("--moss-local-mopd-teachers", nargs="+")
    group.add_argument("--moss-local-student-score-endpoint")
    group.add_argument("--moss-local-mopd-default-domain", default=None)
    group.add_argument("--moss-local-mopd-score-batch-size", type=int, default=8)
    group.add_argument("--moss-local-mopd-batch-wait-ms", type=float, default=3.0)
    group.add_argument("--moss-local-mopd-concurrency", type=int, default=4)
    group.add_argument("--moss-local-mopd-timeout", type=float, default=120.0)
    group.add_argument("--moss-local-mopd-retries", type=int, default=2)
    group.add_argument("--moss-local-mopd-advantage-clip", type=float, default=5.0)
    group.add_argument(
        "--moss-local-mopd-estimator",
        choices=["sampled", "sampled_native", "dense_reverse", "dense_forward"],
        default="sampled",
        help="Opt-in full-vocabulary native-teacher distillation on student trajectories.",
    )
    group.add_argument(
        "--moss-local-native-teachers",
        nargs="+",
        help="Frozen native teachers as DOMAIN=/absolute/path/to/iter_XXXXXXX.",
    )
    group.add_argument("--moss-local-mopd-prefix-frames", type=int, default=0)
    group.add_argument("--moss-local-mopd-prefix-weight", type=float, default=4.0)
    group.add_argument("--moss-local-domain-loss-config")
    group.add_argument("--moss-local-mopd-codebook-weights", type=float, nargs=12)
    group.add_argument("--moss-local-mopd-decision-weight", type=float, default=1.0)
    group.add_argument(
        "--moss-local-replay-manifest", help="Audited frozen-teacher trajectory manifest for mixed GKD."
    )
    group.add_argument("--moss-local-replay-every-n-groups", type=int, default=5)
    group.add_argument("--moss-local-replay-mode", choices=["teacher", "prompt_only"], default="teacher")

    return parser


def validate_args(args):
    cap = args.max_samples_per_microbatch
    if cap is not None and (cap <= 0 or not args.use_dynamic_batch_size):
        raise ValueError("--max-samples-per-microbatch requires a positive value and dynamic batching")
    if args.policy_family != "moss_tts_local":
        return
    domain_specs = domain_distillation.load_specs(getattr(args, "moss_local_domain_loss_config", None))
    args.moss_local_domain_loss_specs = domain_specs
    if domain_specs and getattr(args, "moss_local_mopd_estimator", "sampled") not in {
        "dense_reverse",
        "dense_forward",
    }:
        raise ValueError("Per-domain loss settings require native full-distribution MOPD")
    prefix_frames = getattr(args, "moss_local_mopd_prefix_frames", 0)
    prefix_weight = getattr(args, "moss_local_mopd_prefix_weight", 4.0)
    replay = getattr(args, "moss_local_replay_manifest", None)
    books = getattr(args, "moss_local_mopd_codebook_weights", None)
    decision_weight = getattr(args, "moss_local_mopd_decision_weight", 1.0)
    if books is not None and (len(books) != 12 or any(not math.isfinite(v) or v <= 0 for v in books)):
        raise ValueError("Exactly twelve positive finite RVQ weights are required")
    if not math.isfinite(decision_weight) or decision_weight <= 0:
        raise ValueError("Decision weight must be positive and finite")
    if prefix_frames < 0 or not math.isfinite(prefix_weight) or prefix_weight <= 0:
        raise ValueError("MOPD prefix settings require nonnegative frames and a positive finite weight")
    if (prefix_frames or replay or books is not None or decision_weight != 1.0) and getattr(
        args, "moss_local_mopd_estimator", "sampled"
    ) not in {
        "dense_reverse",
        "dense_forward",
    }:
        raise ValueError("Prefix weighting and teacher replay require native full-distribution distillation")
    if replay and getattr(args, "moss_local_replay_every_n_groups", 5) < 2:
        raise ValueError("Mixed teacher replay requires at least one student group between replay groups")
    if replay and (args.moss_local_old_policy_source != "trainer_preupdate" or args.moss_local_async):
        raise ValueError("Mixed teacher replay supports only synchronous native KL with trainer_preupdate")
    if getattr(args, "moss_local_mopd_estimator", "sampled") != "sampled" and args.moss_local_objective != "mopd":
        raise ValueError("Native distillation estimators require --moss-local-objective mopd")
    parse_components(args.moss_local_reward_components)
    validate_configuration(args)
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

    if args.moss_local_objective == "mopd":
        from miles.policies.moss_tts_local.mopd_client import teacher_routes

        routes = teacher_routes(args.moss_local_mopd_teachers)
        if getattr(args, "moss_local_mopd_estimator", "sampled") != "sampled":
            from miles.policies.moss_tts_local.dense_mopd import native_teacher_sources

            sources = native_teacher_sources(getattr(args, "moss_local_native_teachers", None))
            if domain_specs and set(domain_specs) != set(sources):
                raise ValueError("Domain loss settings must cover exactly the routed teachers")
            if set(sources) != set(routes):
                raise ValueError("Dense MOPD requires exactly one native checkpoint for every routed domain")
            if args.moss_local_async:
                raise ValueError("Native dense MOPD currently requires synchronous training")
        if not args.moss_local_student_score_endpoint:
            raise ValueError("MOPD requires --moss-local-student-score-endpoint for matched prefill scoring")
        if not routes:
            raise ValueError("MOPD requires frozen teacher scoring endpoints")
        if args.moss_local_mopd_default_domain is not None and args.moss_local_mopd_default_domain not in routes:
            raise ValueError("MOPD default domain must identify a configured teacher")
        if not 1 <= args.moss_local_mopd_score_batch_size <= 64:
            raise ValueError("MOPD score batch size must be in [1,64]")
        if args.moss_local_mopd_concurrency < 1 or args.moss_local_mopd_timeout <= 0:
            raise ValueError("MOPD scoring concurrency/timeout must be positive")
        if (
            args.moss_local_mopd_retries < 0
            or args.moss_local_mopd_batch_wait_ms < 0
            or args.moss_local_mopd_advantage_clip <= 0
        ):
            raise ValueError("Invalid MOPD retry/wait/clip configuration")
        if args.group_rm:
            raise ValueError("Local MOPD supplies per-action advantages, not group rewards")
