# MOSS-TTS Local: Transformer Engine experiments

The existing `--transformer-impl local` recipe already uses TE core attention for
packed THD. The `transformer_engine` option replaces the **global GPT decoder's**
linear and norm implementations; it does not convert the separate MOSS local
(depth) transformer or the audio embeddings/heads to TE.

## What changes

| Module | Existing local recipe | Full TE recipe |
| --- | --- | --- |
| Global core attention | TEDotProductAttention, flash configured | Same |
| Global QKV and FC1 | Local column linear and separate Torch RMSNorm | TELayerNormColumnParallelLinear |
| Global output projection and FC2 | Local row linear | TERowParallelLinear |
| Global Q/K and final norms | Torch RMSNorm | TE RMSNorm |
| MOSS depth transformer/audio heads | Existing MOSS implementation | Unchanged |

## Fixed-input experiment

Environment: 4 H200 dev Notebook; training uses DP2 on GPUs 0/1, other GPUs are
occupied separately during replay. Torch 2.13.0+cu130, TE 2.17.0, Megatron
8c1e05747, Miles baseline f7ba9e97f. No environment packages were changed.

Replay uses the same 512 real WER trajectories (8 steps x GB64), native base
weights, BF16, LR0, dynamic microbatches capped at 2 samples, packed THD, no
checkpoint I/O, no live generation or ASR. Exclude steps 0/1 from timing.
Parameter shapes/samples and exact DP/microbatch grouping were checked.

| Configuration | Mean trainer time, steps 2..7 |
| --- | ---: |
| Local baseline | 2.6305 s |
| Local repeated | 2.6438 s |
| Full TE, keeping disabled-fusion settings | 3.0324 s |
| Full TE repeated | 2.9822 s |
| Full TE + gradient accumulation/RoPE fusion candidate | 2.5099 s |

The fusion candidate also removes `--no-persist-layer-norm`; removing a flag
alone is not evidence of a changed TE kernel. The candidate changes several
settings, so this experiment does not attribute its improvement to one fusion.
All variants keep the same Torch Adam path and precision. It is not an FP8 test.

Full TE alone is about 13–15% slower here. The fusion candidate is about 5% faster
than local in this small replay benchmark. These are **trainer** timings, not
end-to-end RL throughput; previous MOPD iterations spent most of their wall time
waiting for rollout/scoring. Keep the original default; the later matched
real-training study has not established non-inferiority or an end-to-end speed
advantage.

## Numerical behavior

Local repeated runs returned bit-identical selected decision/code logprobs on
all 512 samples. First-step gradient sample cosine was 0.999993, showing small
backward execution variability even with identical forward outputs.

Plain TE vs local: selected code logprob MAE 0.0500, max absolute difference
2.1788. Decision logprob MAE 6.67e-6, max 0.0584. Fusion candidate vs local: code
MAE 0.0516, max 2.2064; decision MAE 1.18e-5, max 0.2166. These are selected-action
statistics, not full-vocabulary KL or an ASR/WER equivalence result.

The audit samples up to 4096 values from each of 329 trainable parameter tensors
and its accumulated gradient before the first optimizer step. Initial parameter
samples are identical after canonicalizing fused norm names; all gradients are
finite. Equal-size-per-parameter gradient sample cosines are 0.7809 (plain TE)
and 0.9389 (fusion candidate). Small norm vectors are overrepresented by that
sampling. Weighting per-parameter sample cosines by measured full gradient norms
gives approximate overall cosines 0.9327 and 0.9737. These are **estimates**, not
exact full-gradient cosine measurements. Individual norm gradients can differ
more strongly than the aggregate suggests.

Neither backend is a full-model FP32 oracle. These differences do not by
themselves establish a TE bug, but they do rule out calling the switch numerically
identical to the already validated local recipe. A speed result alone does not
justify claiming unchanged training quality or WER.

## Native checkpoint loading

The first direct TE attempt failed after strict DCP tensor-schema validation:
TE reported its runtime `_extra_state` entries missing. The mossLite native
checkpoint intentionally contains only model tensors. The loader already removes
runtime entries before DCP loading; its final PyTorch check now allows **only
those exact removed keys**. Missing weights, unlisted runtime state and unexpected
keys remain errors. This applies to model-only initialization, not permission to
discard optimizer state during checkpoint resume.

## Reproduction and evidence

Run evidence and drivers:
`/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-local-te/20260909`.
Large outputs:
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-local-te/20260909`.

- `run_replay.py`: uses the production launcher and saved real trajectories;
  `--impl local|transformer_engine`, optional `--audit` and `--fusions`.
- `te_audit.py`: first-step class/parameter/gradient audit. Audit overhead occurs
  before the steady timing window.
- `compare_replays.py`: checks grouping, compares selected-action outputs and
  sampled gradients, and records timing/memory summaries.
- Each run persists exact commands, train logs, debug outputs and comparison JSON.
- `launcher_te_fusions.py` is an experimental copy of the original launcher with
  the three disabled-fusion flags removed; production defaults are unchanged.

Use the existing `gpu_lease.py` around GPU experiments so occupancy is restored.
For a plain TE experiment through the production launcher, append
`--extra-args '--transformer-impl transformer_engine'`. The launcher retains its
original local default. Accepted live update/checkpoint results are recorded below and in the experiment report.

## Important: optimizer format when switching implementations

The existing local WER checkpoint uses `dp_reshardable` distributed optimizer
state. This layout depends on internal flattened parameter buffers. TE fuses
norms into linear modules and changes their parameter ordering. A direct
local-to-TE optimizer resume passed the framework's existing load checks but
corrupted weights on the first update; the subsequent MOPD signal jumped sharply.
That run and the queued continuation are explicitly rejected and marked INVALID.
Increasing optimizer step counts and a successful save are not sufficient proof
of a valid cross-backend restore.

Miles now rejects switching transformer implementations while restoring
bucket-layout optimizer state. Same-implementation resumes continue to work.
To preserve optimizer history, first load under the original implementation and
save with `--dist-ckpt-optim-fully-reshardable`, then restore into TE. A model-only
start with explicitly fresh optimizer state (`--finetune`) is a different option;
it must not be described as restoring the previous optimizer history.

The experiment also adds pre-step model/master-parameter alignment checks for
the migrated TE runs. These compare samples of every owned FP32 optimizer master
parameter shard, cast back to the model dtype, against the corresponding model
weights. Migration acceptance requires those checks as well as state-counter,
forward/backward, refit and save/reload checks.


## Accepted migration and live tests

The original local iteration-31 checkpoint was loaded under local and saved as
`local-canonical-checkpoint-r3/checkpoints/iter_0000031` with fully reshardable
optimizer state. This operation performed no forward/backward or optimizer
update. Adam step stayed 32 and scheduler sample count stayed 2048; all 145 global
norm checkpoint tensors were bit-identical. The conversion's trainer intentionally
stops after save with an explicit completion exception; the driver accepts only
the completion marker and complete checkpoint, not a generic failed job.

`te-live-canonical-r1` then restored that checkpoint into fused TE and completed
two real MOPD updates, GB64, two teacher routes and three student refit receivers.
The 128 samples covered fixture_a=44 and fixture_b=84. Mean absolute advantages
were 0.1411/0.1385 and gradient norms 1.1309/1.2110. Both frozen teachers remained
unchanged. Refits 1/2/3 each transferred 437 named inference tensors.

Before each update, both ranks checked samples from 330 owned model/master
parameter shards in total, with no mismatches. For 153 small parameter shards,
all 459 master/Adam-first-moment/Adam-second-moment tensors were compared in full
against the canonical checkpoint after restore and were bit-identical.

`te-resume-replay-r1` restored the new TE checkpoint under the same implementation
and replayed the saved real MOPD trajectories for two further updates. This is
an explicit fixed-trajectory resume test, not two additional live on-policy
rollouts or a quality experiment. Model/master checks passed before both steps.

| Checkpoint | Saved iteration | Adam step | Scheduler samples |
| --- | ---: | ---: | ---: |
| Original local | 31 | 32 | 2048 |
| Canonical conversion | 31 | 32 | 2048 |
| Accepted live TE | 33 | 34 | 2176 |
| Same-TE replay resume | 35 | 36 | 2304 |

The accepted live checkpoint uses TE's own `dp_reshardable` layout again, which
can resume under the same TE implementation. To switch it back to local while
preserving optimizer state, perform the analogous parameter-based conversion
under TE first. Setting the fully-reshardable flag only at load time does not
convert an existing bucket-based checkpoint.

Final focused policy/loader/resume/launcher regression checks: 109 passed. The
production local default and environment packages remain unchanged. These initial
short tests establish integration/state integrity. A subsequent
[matched real WER training study](moss-tts-local-te-training.md) compares 32-update
Local and full-TE runs on the held-out set; it does not establish equivalence.
All owned GPU experiments finished and occupancy was restored on GPUs 0/1/2/3.
