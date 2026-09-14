# Bounded asynchronous MOSS-TTS Local WER training

Status: real 32-update async/sync WER comparison completed on four H200 GPUs.
The default synchronous recipe is unchanged. This is the one-batch-ahead
`train_async.py` driver, not the text fully-async producer/buffer implementation.

## Contract

Run with `--moss-local-async --moss-local-old-policy-source trainer_behavior
--keep-old-actor` and update_weights_interval=1. There must be exactly one
optimizer step per rollout batch. Only the WER objective is covered; MOPD scorer
co-residency and persistent fully-async generation need separate scheduling.
Offload/rematerialization and overlapping parameter gather are not supported by
this snapshot replay path.

The next batch generates while the current batch trains. The driver waits for
the whole next batch before publishing weights, so each utterance and batch
carry a single version. First-batch lag is zero, subsequent batch lag is one;
mixed, future, unpublished or older versions are rejected. CPU model snapshots
rotate with weight publication and their labels must match the behavior version.
Old/current weights are switched only around a no-grad scoring pass, with actor
restoration guaranteed before backward. The old denominator is not overwritten
when current train outputs are collected.

## Probability convention

The ratio uses current trainer logprob minus a replay under the trainer snapshot
whose weights produced the sample. Frame decisions and all 12 audio codebooks
retain their existing joint action/mask semantics. This corrects the weight-age
mismatch in the training implementation; it does not establish bitwise parity
with SGLang's decode distribution or exact behavior-policy importance sampling.
Raw serving logprobs remain available for auditing. Using current.detach() as
the denominator for stale samples is explicitly rejected by this opt-in path.

## Checkpoint boundary

Only final, drained checkpoints are currently supported: save_interval must be
at least num_rollout. External save triggers and early-exit controls are rejected
because an intermediate checkpoint would otherwise lose the prefetched batch
while advancing the data-source cursor. On restart, snapshots rebuild from the
loaded model and the first new batch has lag zero. This is not a claim of bitwise
identity to an uninterrupted pipeline with an already prefetched stale batch.

## Initial gates

- Four real LR0 batches, GB64, DP2 plus two generation GPUs: all 256 action
  denominators matched current trainer outputs exactly, ratio=1, clip fraction=0.
  Version-lag counts: 64 current, 192 one-version-old. Snapshot labels matched.
- Actual next-rollout and trainer time windows overlapped. The short smoke's
  middle iterations averaged about 9.94 s; this is preliminary, not a formal
  performance conclusion.
- Eight nonzero-learning-rate batches completed with finite gradients and
  nontrivial ratios/clipping, followed by checkpoint save.

The formal comparison matches synchronous cache cleanup in the async driver;
that cleanup runs while the next rollout is in flight. Both drivers log complete
loop/iteration wall time, including publication and checkpoint work. Actor
train timers include old-policy replay, while complete-loop timing also includes
CPU snapshot rotation and weight publication.

Evidence/scripts:
`/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-local-async/20260909`.

## Matched real experiment (2026-09-09)

Both arms used Local BF16, packed THD/Flash attention, DP2 training plus two
SGLang-Omni generation GPUs, LR 3e-6, 32 optimizer updates and GB64 (16 prompts
with four samples each). Training used 512 prompts; evaluation used 256 disjoint
held-out prompts with two fixed generation seeds (512 audios). Each arm repeated
64 initial examples with identical actions, audio hashes and WER against the
shared initial baseline. ASR was qwen3-asr-1.7b, three repeats with median scoring.
The image and SGLang-Omni implementation were unchanged between arms.

| Metric | Synchronous | Bounded async |
| --- | ---: | ---: |
| Complete training loop, including fill/drain, refits and final save | 430.124 s | 351.588 s |
| Samples per second over complete loop | 4.761 | 5.825 |
| Mean main-loop iteration, updates 2–30 | 12.571 s | 10.022 s |
| Mean trainer time, same updates, including old-policy replay | 2.521 s | 4.128 s |
| Generated training frames | 127,706 | 127,839 |
| Initial mean WER | 8.3886% | 8.3886% |
| Post-training mean WER | 7.4147% | 6.0693% |
| Post-training corpus WER | 7.1488% | 6.1809% |
| Post-training exact match | 51.9531% | 57.0313% |

The complete loop was 18.26% shorter, with 22.34% greater sample throughput.
These times exclude server startup and independent evaluation. This is one
32-step run per arm, not a repeated performance benchmark or an equal-wall-time
learning-curve comparison. Frame counts differ by only 0.10%, and mean evaluated
audio durations were 5.166 s (sync) and 5.163 s (async).

Async improved mean WER from its initial model by 2.3194 percentage points
(prompt-bootstrap 95% CI: -3.4939 to -1.1689 points). Async minus this synchronous
run was -1.3454 points (CI: -2.3932 to -0.3303). The synchronous change from
initial was -0.9740 points (CI: -2.1532 to +0.1712). Bootstrap units are prompts,
averaged over the two evaluation seeds, with 20,000 bootstrap draws. These
intervals quantify evaluation-prompt uncertainty for these checkpoints, not
training-run variability. Earlier Local synchronous runs reached 6.908% and
5.837% WER; consequently this study does not establish universally better async
sample efficiency or quality. It establishes useful overlap and real learning
in this configuration. WER and waveform sanity do not establish perceptual,
speaker-identity or style equivalence.

All 2,048 async samples had matching behavior snapshot labels: 64 had lag zero
and 1,984 had lag one. All 31 next-rollout/trainer windows overlapped (mean
intersection 4.484 s), all gradients were finite/nonzero, and publications were
versions 1 through 33. Lagged batches clipped approximately 47–52% of frames;
this warrants retaining the one-update lag bound. The maximum observed absolute
trainer-snapshot versus raw-serving joint logprob difference was 4.453, so the
probability convention above remains essential.

Each rank held three CPU parameter snapshots of 8,322,894,848 bytes each,
49,937,369,088 bytes (46.51 GiB) of tensor payload across DP2. This is the total
snapshot footprint, not a measured process RSS delta. Logged training GPU memory
peaks were about 53.6–53.8 GiB in both arms. The async cost includes an additional
no-grad old-policy forward and snapshot copying; the gain comes from overlap.

## Reproducing the opt-in path

Use the existing Local WER launcher/environment, select `train_async.py` in the
launcher, and add all three flags:

```text
--moss-local-async
--moss-local-old-policy-source trainer_behavior
--keep-old-actor
```

Keep `--update-weights-interval 1`, one optimizer step per rollout batch, and a
final-only save interval. Merely passing these flags to `train.py` is rejected.
The exact experiment launcher is `launcher_async.py` in the evidence directory;
`formal_async.py` runs the matched comparison, `audit_async.py` verifies sample
versions and overlap, and `compare_formal.py` produces `combined-results.json`.
The production synchronous launcher remains the default. Async MOPD, FP8,
unbounded lag, intermediate prefetched-state checkpoints, and full-TE async
performance have not been validated by this study.

## Real checkpoint continuation

The final async checkpoint at iteration 31 was loaded with optimizer/scheduler
state, then four new live WER batches trained at steps 32–35. Adam step advanced
from 32 to 36 and scheduler sample count from 2,048 to 2,304. All 256 new sample
indices were unique, snapshot labels matched (lag zero then one), all four
gradients were finite/nonzero, and five complete weight refits succeeded.
At the first and last resumed steps, both ranks passed all 330 combined parameter
shard/master checks. Checkpoint iteration 35 was saved successfully.

A paired 64-prompt/two-seed subset had mean WER 5.2011% before continuation and
4.8589% afterwards. The difference CI was -1.4289 to +0.7666 percentage points,
so the subset does not establish an additional quality improvement. It validates
a real resume/update/refit/evaluation path. It must not be compared directly with
the 512-audio full-set WER. Both models' subset scores use identical prompt/seed
keys. Evidence: `20260909-async-resume-r1/resume-audit.json` and
`resume-wer-comparison.json` under the shared `runs/miles-moss-rl` directory.
