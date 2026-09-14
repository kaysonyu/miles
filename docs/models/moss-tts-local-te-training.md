# Full TE + fusions: real Local WER training

This follow-up tests real, nonzero-learning-rate WER optimization after the
[TE kernel and checkpoint experiments](moss-tts-local-te.md). The global decoder
uses Transformer Engine; the MOSS local depth transformer and audio heads retain
the existing implementation. Production launcher defaults remain local.

## Controlled setup

Both arms start from the same native mossLite iteration-20000 model with a new
optimizer. Each performs 32 updates, GB64 (16 prompts x 4 samples), LR3e-6, BF16,
Torch Adam and DP2, with two external generation replicas on the other two H200s.
Training uses 512 texts; evaluation uses 256 disjoint texts x two fixed seeds.
ASR recognizes each output three times; the median WER is used.

Environment: existing dev image, Torch 2.13.0+cu130, TE 2.17.0, Megatron 8c1e05747,
Omni 8b9264cc, Miles training code c97f0a795. No packages or model code were changed
for this follow-up. Thirty-two key parsed configuration fields match between the
arms. The intended differences are transformer_impl, gradient accumulation
fusion, RoPE fusion and the persistent-norm disable flag.

The first arm evaluates the initial model on all 512 samples. Each arm's newly
started inference service must then reproduce the same 64-sample subset before
training. Both controls reproduced every audio SHA, selected action sequence and
WER. The Local arm reuses the common initial baseline after passing its own
repeat gate; it does not reuse the TE trained model or its rewards.

Both arms retain the existing `trainer_preupdate` same-forward policy-gradient
surrogate. This study does not establish serving/trainer logprob parity or
change the algorithm into strict behavior-policy PPO.

## Held-out results after exactly 32 updates

WER below is mean effective English WER under the existing reward contract.
Raw mean WER and corpus/error counts are retained in the detailed artifacts.

| Metric | Initial model | Local control | Full TE + fusions |
| --- | ---: | ---: | ---: |
| Mean WER | 8.3886% | 5.8371% | 6.2369% |
| Corpus WER | 8.3103% | 5.8490% | 6.2500% |
| Total word errors / 7232 reference words | 601 | 423 | 452 |
| Exact transcription rate | 51.17% | 58.01% | 54.49% |
| ASR quality failures | 0 | 0 | 0 |
| Language mismatches | 1 | 0 | 0 |

TE improves mean WER by 2.1517 percentage points relative to the initial model
(25.65% relative reduction). Prompt-level paired bootstrap 95% CI for post minus
initial is [-3.2786, -1.0939] percentage points. Local improves by 2.5516 points
(30.42% relative reduction), with CI [-3.6320, -1.4918].

The direct TE-minus-Local difference is +0.3998 percentage points, with 95% CI
[-0.3857, +1.1953]. Local has the better point estimate in this run. The interval
includes zero, so this test does not establish a significant difference; it also
**does not establish equivalence or non-inferiority**. No non-inferiority margin
was predeclared. Each arm has one training run; prompt bootstrap covers evaluation
prompt uncertainty, not variation across repeated training runs.

## Stability and speed

Each arm completed all 32 updates with finite, nonzero gradients, 2048 unique
training sample indices, and 33 complete refits of 437 inference tensors. Initial
strict weight comparisons passed for both generation replicas. First/last-step
model-vs-FP32-master checks cover both ranks and pass without mismatches.

Timing summarizes steps 2..30, excluding the first two warmups and the final
step's audit/save boundary. These are descriptive measurements from real RL,
whose generated trajectories change as each policy learns.

| Metric | Local | TE + fusions |
| --- | ---: | ---: |
| Mean trainer seconds/step | 2.5300 | 2.4346 |
| Mean complete step seconds | 12.5914 | 12.6812 |
| Training audio frames/second | 1587.71 | 1647.92 |
| End-to-end audio frames/second | 319.02 | 316.37 |
| Total generated frames in timing window | 116490 | 116347 |
| Logged max GPU0 used GiB | 53.58 | 53.65 |
| Logged max GPU1 used GiB | 53.73 | 54.00 |

TE trainer time is about 3.8% lower, but complete step time is about 0.7% higher.
This run does not demonstrate an end-to-end RL speedup. Frame-normalized figures
lead to the same practical conclusion. Rollout/reward waiting dominates the
complete iteration. Memory figures are lifecycle-log samples, not continuous
profiler peak-memory measurements.

All 512 audio outputs per arm are 48 kHz and finite, with no zero-energy files.
Mean duration is 5.1598 seconds for Local and 5.1542 seconds for TE (initial:
5.1686 seconds). Near-full-scale sample fractions are small and recorded, but
these checks are not a perceptual, speaker-similarity or MOS evaluation.

## Reproduction and artifacts

SSD report/scripts:
`/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-local-te-training/20260909`.

Large outputs are under
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-rl`:

- `20260909-te-fused32-r2`: accepted TE run, initial/post eval and checkpoint.
- `20260909-local-control32-r1`: matched Local control and checkpoint.
- `20260909-te-fused-resume-r1`: separate real TE continuation test.

`formal_experiment.py` wraps the previously validated experiment harness and
persists exact commands. `launcher_te_fusions.py` is the experimental launcher
copy with the disabled-fusion flags removed. `formal_audit.py` checks master
alignment on first/last steps, including newly initialized optimizers before
Adam moments exist. `analyze_training.py`, `compare_formal.py` and
`audio_sanity.py` produce the training, configuration, paired WER and waveform
audits. Original/transformed audio and raw transcripts are retained.

Use the GPU lease wrapper so experiments run serially and restore gpu-occupy.
TE checkpoints can resume under the same implementation; changing implementations
while preserving optimizer state still requires the parameter-based conversion
described in the preceding TE guide.

## Real checkpoint continuation

`20260909-te-fused-resume-r1` restores the TE iteration-31 checkpoint and performs
four **new live WER rollouts/updates**, GB64, at steps 32..35. This follow-up uses
real generation and ASR rewards, unlike the earlier fixed-trajectory resume test.
All four gradients are finite/nonzero, 256 sample indices are unique, and the
five refits (initial plus four updates) each transfer 437 inference tensors.
First/last model/master checks pass on both ranks.

The checkpoint audit confirms both Adam groups advance from step 32 to 36 and
scheduler samples from 2048 to 2304; the saved iteration advances from 31 to 35.
No optimizer reset or cross-implementation restore is used.

A paired 64-prompt x 2-seed subset checks output after continuation. On that same
subset, mean WER is 5.3031% before continuation and 6.0794% after it; the difference
is +0.7763 percentage points, with CI [-0.8694, +2.4085]. The point estimate is
worse and the interval includes zero. This small subset does not establish that
more steps improve WER, or that the full held-out set regressed. Do not compare
its 6.0794% directly against the 512-sample 6.2369% from the main experiment.
ASR quality/language/repeat checks remain clean; all 128 waveforms are finite and
nonzero. The continuation checkpoint is an integrity-test artifact, not a newly
selected best-quality model.

## Decision

Full TE plus fusions passes real training, held-out improvement versus the initial
model, checkpoint save and real continuation checks on this environment. It has
not shown a quality or complete-RL-throughput advantage over the matched Local
control. Keep the validated Local default. The TE configuration remains an
experimental option; equivalence/non-inferiority and perceptual/speaker quality
would require additional evidence. All owned services and Ray processes were
cleaned up, and gpu-occupy was restored on GPUs 0/1/2/3.
