# MOSS-TTS Local: eight-H200 domain distillation results (2026-09-12)

The experiments establish early domain transfer from the base TTS checkpoint
at iteration 162500. At inference temperature 0.7, the 128-update MOPD student
improves both dialect and instruction attribute scores on a fresh held-out RL
test set. Its dialect CER also increases significantly, so the run does **not**
meet the registered joint attribute-improvement and CER-preservation criterion.

The method with these results uses frozen native teachers and exact
full-vocabulary reverse KL, with training temperature 0.7. These results do not
validate the original selected-action surrogate, which remains the default.

## Fresh final test

There are 192 prompts per domain. Scores below use inference temperature 0.7.
Judge is AnyAudio-Judge attribute p(yes), not human accuracy or MOS. CERcap is
the equal-sample mean of min(CER, 1), evaluated by an independent ASR model.

| Model | Additional optimizer updates | Dialect Judge | Dialect CERcap | Instruction Judge | Instruction CERcap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base 162500 | 0 | 0.2151 | 0.0720 | 0.5187 | 0.3690 |
| Dialect teacher 6000 | — | 0.4708 | 0.0423 | — | — |
| Instruction teacher 6000 | — | — | — | 0.7685 | 0.0859 |
| Dialect-only student | 64 | 0.2448 | 0.0883 | — | — |
| Instruction-only student | 64 | — | — | 0.5949 | 0.1631 |
| Two-teacher MOPD | 128 | 0.2835 | 0.1241 | 0.5855 | 0.1267 |

MOPD's dialect Judge gain is 0.0685, with paired 95% interval
[0.0396, 0.0975]. Dialect CERcap increases by 0.0521, with interval
[0.0272, 0.0795]. Instruction Judge improves by 0.0669, with interval
[0.0321, 0.1031], and instruction CERcap falls by 0.2423, with interval
[0.1860, 0.2984]. Instruction generation truncation remains 18.75%, compared
with 26.04% for the instruction-only student and 9.90% for its teacher at this
inference temperature.

The registered criterion requires both Judge improvement intervals to be above
zero and both CER improvement intervals to have lower bounds at least -0.05.
The dialect CER interval fails this requirement. At inference temperature 1,
dialect CER deteriorates further and the instruction attribute gain is not
supported; inference settings must be stated with the quality conclusion.

Shanghai is the largest dialect tradeoff in the exploratory 32-prompt buckets:
Judge rises from 0.1426 to 0.4474, while CERcap rises from 0.1595 to 0.4508.
The corresponding teacher CERcap is 0.1536. Dialect-only transfer is weaker:
its final-test Judge gain is small, and its validation-set attribute gain was
not clear. These details should accompany the aggregate result.

## Experimental and implementation controls

- All students descend from `local_pretrain_v0.0.0` iteration 162500. The joint
  student starts from that checkpoint directly. Neither VoiceClone 200000 nor
  a single-domain student is substituted as the joint initializer.
- Teachers are the supplied dialect/instruction `v0.0.1` iteration 6000 models.
  Training uses one routed teacher per sample and never averages their logits.
- Each domain has 2304 training prompts, 96 validation prompts and 192 fresh
  final-test prompts. No reference codes appear in training JSONL. Test text and
  recording-group hashes exclude all earlier RL selections; teacher-pretraining
  exclusion is not guaranteed.
- GB64 is 16 prompts times four student samples. Adam uses constant LR3e-6 and
  gradient clipping 1.0. Full model parameters are updated. Generation uses
  at most 192 frames, top-p 1 and no top-k truncation.
- MOPD processes 8192 trajectories: 4192 dialect and 4000 instruction. Each
  64-update single-domain run processes 4096. Every trajectory passed domain,
  teacher-route and frozen-serving-teacher identity checks.
- Native teacher parameters remain unchanged and match the standalone teacher
  runs. Serving teachers are unchanged. All 330 learned runtime tensors agree
  across the student scorer and four generation replicas.
- Checkpoint scalar state confirms 128 optimizer updates and 8192 samples.
  The instruction confirmation evaluation adds zero optimizer updates and
  reproduces its source checkpoint's 330 parameter checksums exactly.
- The actual image passes 135 MOSS Local tests. The same-weight negative
  control has exact zero dense KL and floating-point gradient residuals below
  1e-6. Native/serving score differences are separately measured, not assumed
  bitwise identical.
- The fresh MOPD student's initial generation matches the dedicated base
  deployment exactly for all 384 validation trajectories across both inference
  temperatures. Judge means match exactly; ASR mean differences are at most
  about 0.0017.

The experiment uses the user-designated `train` Notebook in CQ-科研驾驶舱 /
CQ项目, with eight H200s and image
`docker.sii.shaipower.online/inspire-studio/miles-moss-tts-local-env:20260908-cu130-v1`.
No additional GPU Jobs were created. Experiment GPU processes were cleaned up,
all eight `gpu-occupy` workers were restored and verified, and the CPU ASR
observer was stopped after all 9180 evaluation records had ASR and Judge scores.

## Artifacts and checkpoints

Artifact root:
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2`

Scripts and exact configuration:
`/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2`
(`joint-suite-r1.json`, `run_experiment.py`, `run_suite.py`, `lease8.py`).

Checkpoint paths relative to the artifact root:

- MOPD128: `mopd-reverse-t07-128-r1/checkpoints/iter_0000127`
- Dialect64: `confirm-dialect-reverse-t07-64-r1/checkpoints/iter_0000063`
- Instruction64: `pilot-instruction-reverse-t07-r1/checkpoints/iter_0000063`

These are Miles training checkpoints; directory iteration numbers start at
zero. The root's `REPORT.md`, `confirmation-assessment.json`, `run-audit.json`
and per-run `routing-audit.json` contain the evidence. `audio-review.html`
compares all completed models and temperatures. `audio-review-compact.html`
embeds 24 clips from eight fixed prompts for offline listening (about 43 MiB).

The original selected-action experiments are documented separately in
[the earlier domain OPD results](moss-tts-local-domain-opd-results.md).
