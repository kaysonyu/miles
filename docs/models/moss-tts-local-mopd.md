# MOSS-TTS Local MOPD implementation log

Status: end-to-end integration and throughput acceptance completed. No trained domain teachers are available yet.
The initial frozen teacher is the compatible Local base checkpoint; experiments
validate the pipeline and speed, and do not claim distillation quality gains.

## Acceptance plan

1. Add an Omni score-only pipeline consuming exact prompt rows, decisions and
   student codes. Use one causal global prefill and batched depth teacher forcing;
   return selected FP32 decision/code logprobs, input identity and teacher version.
2. Compare a fixed set of real trajectories, including stop/truncation and variable
   lengths. Audit scoring-vs-generation numerical differences and an identical-
   weight negative control before selecting the training score convention.
3. Add domain-routed frozen teachers, bounded concurrent scoring and per-action
   MOPD advantages/loss to Miles without changing the WER-GRPO path.
4. Run real update/refit/checkpoint/resume tests. Verify teacher immutability and
   student version progression. Base teachers are protocol fixtures, not experts.
5. Benchmark matched rollout/trainer workloads and the original WER-RL recipe.
   Record teacher scoring time, overlap, step throughput and GPU memory. Optimize
   bottlenecks without changing the chosen probability/mask/reduction semantics.

## Resource and provenance

Use the existing Inspire dev Notebook in CQ-科研驾驶舱. Stop gpu-occupy only for
leased GPUs and restore all four GPUs after each run. Keep secrets out of logs.
Editable roots remain under /inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable.
Run evidence: /inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909.
Baseline Miles: fbb0b8746; Omni: e621261b2; SGLang: 71de97b264.

## Decisions

- Existing public /generate does not implement supplied-action scoring. Add an
  explicit endpoint and isolated teacher pipeline; do not relax generation
  validation and silently treat generation logprobs as input scores.
- Preserve decision [D] and code [F,12] fields. A terminal stop has no code action.
- Keep frozen teacher endpoints separate from student refit clients.
- Keep Radix Cache off, student TTS CUDA Graph on, vocoder graph off.
- Treat serving/training and prefill/decode logprob differences as measured
  numerical issues. Do not claim same-model zero advantages without verifying it.

## Progress

- Read the current Delay MOPD routing, exact-token protocol, clipping/masks and
  Local/Miles score/loss boundaries. Existing reference worktrees remain unchanged.

### First scorer validation

- A score-only SGLang pipeline now accepts /score_actions batches. It executes
  global causal prefill and frame-batched forced local-depth scoring; generation
  APIs and the existing WER path remain distinct.
- Protocol validation: 7 tests passed. Miles policy regression plus new per-action
  MOPD math tests: 84 passed (including exact equal-score zero gradients, opposite
  per-channel advantages, inactive NaNs and detached teacher scores).
- First four real trajectories returned [66] decision and [65,12] code scores;
  first/cold request wall time 1.261 s. Prefill scoring vs original sampled logprob:
  code MAE around 0.05 and max around 0.52. This is an execution-path discrepancy,
  not evidence that an untrained base teacher is better. The existing Local
  same-forward trainer ratio remains explicitly labeled as a surrogate.
- Added frozen weight-update guards and a teacher weight SHA256, persisted by
  domain in checkpoint/mopd_teachers.json to detect teacher changes on resume.
- Early GPU integration failures (full hidden-state selection and request payload
  ownership) were fixed; artifacts retain the failed probes as well as success.

### Paired scoring convention

The first end-to-end smoke completed two MOPD steps and saved a checkpoint, but
base/base teacher-prefill minus student-generation scores produced nonzero
advantages (mean absolute about 0.045). It is not a valid zero-signal negative
control. Do not interpret those updates as teacher expertise.

The selected convention is now teacher_prefill minus student_prefill, with the
same exact actions, temperature and deterministic prefill microbatch grouping.
The original sampled logprobs remain available for audit. A separate updatable
student scoring replica receives every student refit and its version must match
the generating student's version. Only the teacher is frozen.

For throughput, teacher scoring runs on trainer GPU0 and student scoring on
trainer GPU1 during rollout; both finish before the trainer starts. This retains
the two original student generation GPUs. Each scorer uses a bounded KV pool,
loads no codec, and has no optimizer. This placement passed the real memory and overlap benchmark recorded below.

### Matched-score negative control

- Two real scoring pipelines on separate H200s returned bit-identical decision
  and code scores for the four saved test trajectories. Each result carries its
  prefill-batch fingerprint and local score chunk size; Miles rejects mismatches.
- The matched-score, co-resident-trainer experiment completed two updates with
  MOPD advantage=0, loss=0 and gradient norm=0, and saved its checkpoint. This
  replaces the earlier rollout-vs-prefill prototype as the accepted convention.
- The first harness's final teacher-check parser assumed `stages`, while that
  admin endpoint returns `results`; training and checkpointing succeeded, and
  the harness parser was fixed for subsequent frozen-teacher verification.
- Protocol/admission/coalescing tests: 30 passed. Miles policy/client tests:
  88 passed. Client tests cover domain routes, paired batching, input identity,
  failure without zero-score fallback, and rejecting changed teachers on resume.

## Current launch shape (four H200s)

- GPU0: trainer rank0 plus frozen teacher scoring (scoring only during rollout).
- GPU1: trainer rank1 plus updatable student scoring replica.
- GPU2/3: original student generation replicas; TTS CUDA graphs retained.
- All scoring completes before the optimizer phase. Scoring replicas have no
  codec and bounded KV pools. No teacher endpoint is included in student refits.
- The updatable scoring replica must not share the weight-transfer source GPU:
  NCCL's refit group must contain distinct physical GPUs. With the above DP2
  layout the sender is GPU0 and receivers are GPU2, GPU3 and GPU1.

The score signal is:

```
A_decision = clip(teacher_prefill_decision - student_prefill_decision, -5, 5)
A_code     = clip(teacher_prefill_code     - student_prefill_code,     -5, 5)
```

Per-action PPO terms retain the existing Local `trainer_preupdate` same-forward
ratio convention; this is explicitly a surrogate, not a claim of serving/trainer
bitwise policy parity. Teacher and student prefill scores are detached. The WER
branch remains unchanged.


## Completed integration checks

The accepted runs use paired prefill scoring throughout:

| Run | Workload | Result |
| --- | --- | --- |
| paired-smoke-r1 | DP2, 2 updates, GB4, identical weights | Exact zero advantages/loss/gradients; checkpoint iter 1 |
| paired-resume-r1 | Resume at step 2; 2 further updates | Exact zero signal; checkpoint iter 3; frozen teacher live checksum unchanged |
| positive-r1 | DP2, 4 updates, GB8, 2 domain routes | Nonzero gradients at every step; checkpoint iter 3; both teachers unchanged |
| benchmark-mopd-r1 | DP2, 8 updates, GB64, 2 domain routes, LR0 | All 512 trajectories passed identity/score audits; exact zero signal for equal weights |

Run roots are under
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909`.
Each contains `commands.json`, `info.json`, service/trainer logs and CPU-readable
rollout/train debug artifacts. `audit.json` checks DP partitions, sample coverage,
per-action advantages, teacher routing and live frozen-weight checks.

The positive-signal canary uses a copy of the previously WER-trained **student**
checkpoint with fresh optimizer state and two frozen **base** teacher fixtures.
It intentionally creates a measurable difference; these are not trained expert
teachers. Its gradient norms were 1.7210, 1.4253, 1.2767 and 1.9885. Both domains
were exercised (11 and 21 samples). The original WER checkpoint is untouched.

## Running with real teachers later

All teachers must use the same Local token/codebook contract as the student:
binary continue/stop decisions and 12 audio codebooks. This implementation does
not translate Delay tokens into Local tokens and does not support quantized
teacher checkpoints. Export each compatible teacher to the HF layout expected
by the existing MOSS-TTS Local SGLang model loader.

Start each frozen teacher with its own model directory and port:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang_omni.cli serve \
  --config /inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/sglang-omni/examples/configs/moss_tts_local_score.yaml \
  --model-path /absolute/path/to/teacher-hf --host 127.0.0.1 --port 31001
```

Start one updatable student scoring replica on GPU1, using
`moss_tts_local_student_score.yaml` and the student HF directory. Keep the two
existing generation pipelines on GPU2/3. Score pipelines have no codec/vocoder.
The `student-score` endpoint must join student refits; teacher endpoints must not.

Extend the existing `scripts/run_moss_tts_local.py` launch with:

```text
--objective mopd
--teachers domain_a=http://127.0.0.1:31001/score_actions,domain_b=http://127.0.0.1:31002/score_actions
--teacher-default-domain domain_a
--student-score-endpoint http://127.0.0.1:31003/score_actions
```

Keep `--omni-endpoints` set to the two generation services only. Put the desired
teacher domain in each JSONL record's `metadata.domain`; routing selects one
teacher per sample, not an average or ensemble of all teachers. Missing routes,
invalid response shapes/hashes, nonfinite scores, unequal teacher/student scoring
batch contexts and stale student versions abort the rollout rather than silently
creating zero rewards. MOPD does not call ASR during training; ASR remains the
independent quality evaluator.

Teacher identities are lazily pinned at first use and written by domain to
`checkpoints/mopd_teachers.json`. On resume, use the checkpoint root containing
that manifest. A teacher URL may change; its actual weight SHA256 and model
identity must match. An unused configured domain is checked when first exercised.
Both routes were exercised in the multi-teacher acceptance runs.

The repeatable Notebook driver is
`/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/run_mopd.py`.
It persists exact commands and run configuration, uses distinct ports, checks
teacher weights before/after training, and cleans up only its own servers/Ray.
Wrap it in the existing `gpu_lease.py` to stop and restore `gpu-occupy`.


## Throughput acceptance (2026-09-09)

The matched benchmark uses 4 H200s, DP2 training, 2 generation replicas,
16 prompts x 4 samples = GB64, concurrency 2 per generation endpoint, max 128
response frames, temperature 1, top-p 1, top-k -1, identical dataset order/seeds,
LR0 and no checkpoint saving. MOPD has two frozen base teacher processes and
one updatable student scorer; WER uses the existing remote ASR reward recipe.
Both retain debug output so its cost is present on both sides.

Each runs 8 updates; exclude the first 2 for warmup. The remaining 384 samples
contain exactly 23,617 audio frames in both runs.

| Metric | Existing WER-GRPO | Local MOPD |
| --- | ---: | ---: |
| Mean step seconds | 12.5215 | 12.3454 |
| Median step seconds | 12.2857 | 12.0779 |
| Samples/second | 5.1112 | 5.1841 |
| Audio frames/second | 314.3538 | 318.8360 |
| Mean trainer seconds | 2.5836 | 2.6444 |
| Mean train-wait seconds | 9.9379 | 9.7010 |
| Logged GPU0 maximum used GiB | 53.28 | 74.94 |
| Logged GPU1 maximum used GiB | 53.51 | 65.22 |

Observed MOPD mean step time is 1.41% lower. This small difference should be
interpreted as comparable throughput with no observed regression in this run,
not as a statistically established speedup or a guarantee for all future teacher
counts and workloads. The device-memory figures are maxima of lifecycle log
samples, not continuous profiler peak-memory measurements. Both trainer GPUs
have substantial headroom within the approximately 139.8 GiB reported capacity.

Teacher/student paired HTTP scoring handled 512 samples in 419 microbatches.
Its cumulative request-wall-time counter is 73.23 seconds over the full run;
requests overlap each other and generation, so this must not be added to step
latency. The scorer does additional work, but uses trainer GPUs during rollout
and replaces the ASR reward dependency in this objective. Model startup and
checkpoint I/O are excluded from these steady-state numbers.

Evidence: `benchmark-comparison.json` at the experiment root and each run's
TensorBoard events, trainer log and debug samples. The benchmark fixes weights
to keep the generated workload comparable; the separate positive-signal runs
exercise nonzero optimizer updates.

## Regression checks

Final focused Miles strategy/client/launcher checks: 90 passed. Omni Local plus
prefill admission/coalescing checks: 235 passed, 17 skipped (existing GPU-only
conditions in the CPU test invocation). New scoring was separately exercised
on actual H200s. A wider launcher run earlier had 506 passes and four existing
AMD GLM5.2 launcher errors caused by its use of an unavailable `U.exec_command`;
that unrelated launcher was not modified. Existing default MOSS launch snapshots
remain unchanged; two MOPD snapshots cover save and no-save modes.


## Nonzero optimizer-state resume

`positive-resume-r1` loads `positive-r1/checkpoints` without finetune or optimizer
reset, starts at rollout 4 and completes rollouts 4/5. Both teacher fixtures are
restarted at new URLs; the persisted domain/weight manifest validates them before
scoring. The final checkpoint is `positive-resume-r1/checkpoints/iter_0000005`.

Reading the distributed checkpoint byte metadata and deserializing only optimizer
and common scalar state confirms both Adam parameter-group steps advance from
4 to 6; scheduler consumed-sample count advances from 32 to 48; checkpoint
iteration advances from 3 to 5. This is an optimizer-state resume, not merely
loading model weights and starting over. The evidence is
`positive-resume-r1/checkpoint-state-audit.json`.


## Generation reproducibility boundary

A cross-run audit of all 512 benchmark samples found identical prompts and
continue/stop decisions for all samples, and identical codes, sampled logprobs
and audio SHA256 for 510/512. Samples 384 and 386 in rollout 6 had different
codes despite equal seeds, weight versions and frame lengths. Therefore this
benchmark does not establish bitwise-identical generated audio for every sample.
See `benchmark-output-comparison.json` and `benchmark-output-differences.json`.

A separate replay then started **only the existing generation pipeline**, with
no teacher, student scorer or trainer. It replayed the two saved inputs/seeds
five times with request concurrency 1/2. One concurrent execution produced
different codes/audio for both requests; the other four executions matched. Nominal
request concurrency alone does not fix the actual dynamic prefill/decode batch.
This demonstrates that the discrepancy can occur independently of MOPD scoring;
it does not establish the exact kernel responsible. The original generation
settings were preserved rather than forcing a potentially slower deterministic
batch policy. Evidence: `generation-replay.json`, `generation-replay.pt` and
`generation-replay-server.log` in the SSD run-evidence directory.

In contrast, the paired scoring pipelines explicitly match actual batch
fingerprints and depth chunk sizes. All 512 identical-weight MOPD samples had
exactly zero advantages and all eight updates had zero gradients. Different
teacher/student weights yielded nonzero signals; the two resumed updates had
gradient norms 0.8498 and 0.8003, both domain routes covered eight samples, and
both frozen-teacher checks passed again.

All owned experiment servers and Ray processes were cleaned up. The final GPU
lease restored `gpu-occupy` on GPUs 0,1,2,3. No environment package changes or new
image were required; this delivery uses the existing editable installs. Omni
implementation commit: `8b9264cc` on `moss-local-mopd`.
