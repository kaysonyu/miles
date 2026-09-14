# MOSS-TTS Local RL with SGLang-Omni

This opt-in policy trains mossLite v0.1.1 MOSS-TTS Local with Miles/Megatron and externally launched SGLang-Omni pipelines. It replays structured text decisions and 12 audio codebooks per frame; the audio actions are never represented as text tokens.

The [adapter structure and compatibility contracts](../design/moss-tts-local-adapters.md) describe policy resolution, trainer snapshot ownership, structured transport and reward composition.

The initial recipe is ported from the validated Slime MOSS experiment. It requires the native split-v1 model checkpoint, its converted Omni checkpoint, the audio codec, disjoint JSONL training/evaluation prompts, and an ASR chat-completions endpoint. Install Miles, Omni, SGLang and Megatron from their editable checkouts before launching.

## Training

Launch each Omni replica on a GPU outside the trainer Ray allocation, using Omni's `examples/configs/moss_tts_local.yaml` and the matching checkpoint/codec paths. The successful long-run WER recipe explicitly uses the following settings:

```text
--tts_engine.engine.disable_radix_cache true
--tts_engine.engine.disable_cuda_graph false
--vocoder.factory.cuda_graph false
```

The vocoder `cuda_graph` option accelerates the streaming `_CodecStreamSession.step` path. Miles RL uses the non-streaming `/generate` route, whose vocoder runs `decode_codes_batch`; those requests do not replay the streaming graphs. Keep this option disabled for the RL recipe to avoid unused capture work and graph memory. Graph/eager equivalence for streaming must use identical chunk boundaries: full-sequence and streaming decode are not interchangeable waveform baselines.

Do not infer evaluation repeatability from `concurrency=1` or a fixed seed alone. Leaving Radix Cache enabled changed generated actions in a no-update control; re-establish the baseline after changing these settings. Then run:

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

The optional full Transformer Engine path has separate [numerical and performance experiments](moss-tts-local-te.md). The validated default remains `local` with TE packed attention; selecting full TE changes the global decoder kernels and is not a bitwise-equivalent switch.

## Evaluation

Use `tools/moss_tts_local/evaluate_wer.py` against a fixed endpoint with `--concurrency 1`, a held-out JSONL dataset and repeated `--seed` values. Run a no-update repeat before comparing trained weights. `tools/moss_tts_local/compare_wer_eval.py` compares matching prompt/seed records using prompt-level bootstrap confidence intervals. Keep raw transcripts, effective/raw WER and invalid-language/quality failures. Training reward alone is not evidence of held-out improvement.

The opt-in WER/SIM/RM composition, reference-audio metadata and Judge rubric contract are documented in [moss-tts-local-rewards.md](moss-tts-local-rewards.md). The default `wer=1.0` path remains unchanged.

The default native model initialization is distinct from checkpoint resume: `--resume` loads the run's checkpoint and optimizer state. A new run starts at rollout zero. Store large optimizer checkpoints on a capacity tier with sufficient space.

The optional [bounded asynchronous WER path](moss-tts-local-async.md) overlaps the next rollout with training and replays a matching old-policy snapshot. A matched four-H200 experiment completed 32 updates with 22.3% higher loop throughput and improved WER from the initial model; the document records quality uncertainty, memory cost and final-only checkpoint constraints.
