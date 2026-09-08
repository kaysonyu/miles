# MOSS-TTS Local RL with SGLang-Omni

This opt-in policy trains mossLite v0.1.1 MOSS-TTS Local with Miles/Megatron and externally launched SGLang-Omni pipelines. It replays structured text decisions and 12 audio codebooks per frame; the audio actions are never represented as text tokens.

The initial recipe is ported from the validated Slime MOSS experiment. It requires the native split-v1 model checkpoint, its converted Omni checkpoint, the audio codec, disjoint JSONL training/evaluation prompts, and an ASR chat-completions endpoint. Install Miles, Omni, SGLang and Megatron from their editable checkouts before launching.

## Training

Launch each Omni replica on a GPU outside the trainer Ray allocation, using Omni's `examples/configs/moss_tts_local.yaml` and the matching checkpoint/codec paths. Keep the vocoder CUDA graph disabled in the validated configuration. Then run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_moss_tts_local.py \
  --model-dir /models \
  --pretrained-checkpoint /models/moss-native \
  --megatron-path /sources/Megatron-LM \
  --data-dir /datasets --train-data train.jsonl \
  --omni-endpoints http://127.0.0.1:18080 \
  --asr-url https://asr.example/v1/chat/completions \
  --asr-key-file /secrets/reward-secret.env \
  --output-dir /runs/moss --num-gpus-per-node 1
```

The key file must define `INSPIRE_API_KEY`. Only its path is passed to reward workers; keep the secret outside checkpoints, source control and images. The launcher preserves external processes. Its default starts a Ray head; use `MILES_SCRIPT_EXTERNAL_RAY=1` for an existing trainer cluster.

The supported baseline uses TP/PP/CP/EP=1, full-model training, BF16, packed THD with at most two samples per microbatch, Ray object-store transport, and full NCCL refits. Scale training using data parallelism. Refit pauses generation, transfers global/local/audio weights, checks the new version and resumes generation. The reward is `1 - effective English WER`, with three ASR recognitions and median selection.

The `trainer_preupdate` mode is an explicitly labeled same-forward policy-gradient surrogate. It retains server/trainer logprob discrepancy metrics; a ratio of one does not establish behavior-policy PPO parity. The stricter `server_behavior` mode fails when its configured parity threshold is exceeded.

## Evaluation

Use `tools/moss_tts_local/evaluate_wer.py` against a fixed endpoint with `--concurrency 1`, a held-out JSONL dataset and repeated `--seed` values. Run a no-update repeat before comparing trained weights. `tools/moss_tts_local/compare_wer_eval.py` compares matching prompt/seed records using prompt-level bootstrap confidence intervals. Keep raw transcripts, effective/raw WER and invalid-language/quality failures. Training reward alone is not evidence of held-out improvement.

The default native model initialization is distinct from checkpoint resume: `--resume` loads the run's checkpoint and optimizer state. A new run starts at rollout zero. Store large optimizer checkpoints on a capacity tier with sufficient space.
