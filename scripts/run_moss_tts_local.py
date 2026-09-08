"""Train mossLite MOSS-TTS Local with pre-launched SGLang-Omni endpoints.

Requires a model-only Megatron checkpoint, a converted Omni checkpoint and
JSONL prompts. Omni and the ASR reward service are externally managed.

Args:
  --model-dir / --model-name: Converted Omni checkpoint location.
  --pretrained-checkpoint: Native mossLite logical checkpoint root.
  --data-dir / --train-data: Prepared WER training prompts.
  --omni-endpoints: Comma-separated, already-running Omni HTTP endpoints.
  --asr-url / --asr-key-file: ASR endpoint and a file read only by reward workers.
  --num-gpus-per-node: Trainer GPUs; inference GPUs are outside this Ray allocation.

Example:
  python scripts/run_moss_tts_local.py --num-gpus-per-node 1 \
      --pretrained-checkpoint /models/moss-native --omni-endpoints http://127.0.0.1:18080
"""

import shlex
from dataclasses import dataclass, field

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    preserve_external_processes: bool = True
    model_dir: str = "/root/models"
    model_name: str = "MOSS-TTS-Local-mossLite-v0.1.1-iter20000-sglang"
    data_dir: str = "/root/datasets"
    train_data: str = "moss-wer-train.jsonl"
    pretrained_checkpoint: str = "/root/models/moss-native"
    megatron_path: str = "/root/Megatron-LM"
    omni_endpoints: str = "http://127.0.0.1:18080"
    asr_url: str = "http://127.0.0.1:8000/v1/chat/completions"
    asr_key_file: str = "/root/reward-secret.env"
    num_gpus_per_node: int = 1
    num_rollout: int = 32
    rollout_batch_size: int = 16
    samples_per_prompt: int = 4
    seed: int = 20260956
    rollout_concurrency: int = 2
    max_response_len: int = 128
    learning_rate: float = 3e-6
    save_interval: int = 32
    resume: bool = False
    extra_args: str = ""


def execute(args: ScriptArgs):
    q = shlex.quote
    checkpoint = (
        f"--hf-checkpoint {q(args.model_dir + '/' + args.model_name)} "
        f"--pretrained-checkpoint {q(args.pretrained_checkpoint)} "
        f"--save {q(args.output_dir + '/checkpoints')} --save-interval {args.save_interval} "
    )
    checkpoint += (
        f"--load {q(args.output_dir + '/checkpoints')} --use-checkpoint-opt-param-scheduler "
        if args.resume
        else "--start-rollout-id 0 "
    )
    rollout = (
        "--policy-family moss_tts_local --rollout-backend sglang_omni --model-name moss_tts_local "
        f"--sglang-omni-endpoints {' '.join(q(x) for x in args.omni_endpoints.split(','))} "
        f"--prompt-data {q(args.data_dir + '/' + args.train_data)} --input-key text --label-key text "
        "--metadata-key metadata --rollout-shuffle "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.samples_per_prompt} --rollout-seed {args.seed} "
        f"--rollout-max-response-len {args.max_response_len} --rollout-max-context-len 1024 "
        "--rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 "
        f"--sglang-server-concurrency {args.rollout_concurrency} "
        "--custom-rm-path miles.policies.moss_tts_local.wer_reward.reward_func "
    )
    training = (
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--global-batch-size {args.rollout_batch_size * args.samples_per_prompt} --micro-batch-size 1 "
        "--moss-local-packed-thd --use-dynamic-batch-size --max-tokens-per-gpu 1024 "
        "--max-samples-per-microbatch 2 --seq-length 1024 --max-position-embeddings 40960 "
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--update-weight-transfer-mode broadcast --object-store-backend ray "
        "--moss-local-old-policy-source trainer_preupdate --moss-local-reuse-train-forward "
        "--advantage-estimator grpo --disable-grpo-std-normalization --eps-clip 0.2 --eps-clip-high 0.2 "
        f"--optimizer adam --use-torch-adam --lr {args.learning_rate} --lr-decay-style constant "
        "--weight-decay 0 --clip-grad 0 --attention-dropout 0 --hidden-dropout 0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "
        "--transformer-impl local --no-persist-layer-norm --no-rope-fusion "
        "--no-gradient-accumulation-fusion --no-masked-softmax-fusion "
        "--log-interval 1 --use-tensorboard "
    )
    U.execute_train(
        checkpoint + rollout + training + U.get_default_wandb_args(__file__, run_id=args.run_id) + args.extra_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="moss-tts-local",
        megatron_path=args.megatron_path,
        config=args,
        extra_env_vars={
            "TENSORBOARD_DIR": args.output_dir + "/tensorboard",
            "NCCL_CUMEM_ENABLE": "0",
            "NCCL_NVLS_ENABLE": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MOSS_TTS_WER_ASR_URL": args.asr_url,
            "MOSS_TTS_WER_ASR_API_KEY_ENV": "INSPIRE_API_KEY",
            "MOSS_TTS_WER_ASR_API_KEY_FILE": args.asr_key_file,
            "MOSS_TTS_WER_ASR_MODEL": "qwen3-asr-1.7b",
            "MOSS_TTS_WER_ASR_REPEATS": "3",
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
