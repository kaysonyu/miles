# Bounded asynchronous MOSS-TTS Local WER training

Status: implementation gates passed; formal WER/throughput evaluation in progress.
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
  nontrivial ratios/clipping, followed by checkpoint save. Formal evaluation
  must establish whether WER improves and how it compares with synchronous RL.

The formal comparison matches synchronous cache cleanup in the async driver;
that cleanup runs while the next rollout is in flight. Both drivers log complete
loop/iteration wall time, including publication and checkpoint work. Actor
train timers include old-policy replay, while complete-loop timing also includes
CPU snapshot rotation and weight publication.

Evidence/scripts:
`/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-local-async/20260909`.
