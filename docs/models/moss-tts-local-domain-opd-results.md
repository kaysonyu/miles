# MOSS-TTS Local domain OPD/MOPD pilot (2026-09-12)

The real-teacher training pipeline works, but this pilot did not establish a
student that improves both dialect and instruction fidelity while preserving
text accuracy. Instruction-only OPD improved text following and stopping;
dialect-only OPD and the tested MOPD checkpoints did not meet the combined
quality criterion. These are bounded empirical results for the current
selected-action, paired-prefill implementation, not a general impossibility
claim about on-policy distillation.

The student started from `local_pretrain_v0.0.0`, iteration **162500**, with fresh
RL optimizer state. Teachers were the user-provided dialect and instruction
iteration-6000 checkpoints. All 438 exported tensors per teacher matched the
specified HF checkpoints after layout conversion. Each domain used 2304 RL
training prompts, 96 validation prompts and 96 final evaluation prompts.

Each row below uses the same 96 final test prompts per domain. Audio Judge is
the mean target-attribute `p(yes)`, not human accuracy. Capped CER is
`min(CER, 1)`; raw CER, medians, transcripts and ASR failure diagnostics are
retained in the experiment evidence.

| Model | Dialect Judge ↑ | Dialect capped CER ↓ | Instruction Judge ↑ | Instruction capped CER ↓ |
| --- | ---: | ---: | ---: | ---: |
| Base 162500 | 0.2147 | 0.2051 | 0.6597 | 0.7941 |
| Dialect teacher | 0.4218 | 0.1779 | 0.6435 | 0.4650 |
| Instruction teacher | 0.2344 | 0.2185 | 0.7962 | 0.2131 |
| Dialect OPD, 128 updates, LR 3e-6 | 0.2667 | 0.4251 | 0.5593 | 0.8744 |
| Instruction OPD, 128 updates, LR 3e-6 | 0.2129 | 0.3267 | 0.6612 | 0.3583 |
| MOPD, 128 updates, LR 1e-6 | 0.2493 | 0.4829 | 0.5479 | 0.8207 |
| MOPD, 64 updates, LR 3e-6 | 0.2562 | 0.5271 | 0.5636 | 0.7531 |

Instruction OPD at 128 updates reduced generation truncation from 94.79% to
11.46%. Its instruction-attribute score change was approximately 0.0014 with
a paired-bootstrap 95% interval of [-0.0360, 0.0381], so the stopping/text gain
does not establish improved expressive instruction following.

At the user's request, all experiment jobs in the visual compute group were
stopped or allowed to finish at approximately 14:34 China time. The last
LR-3e-6 MOPD run logged 78 updates before stopping; its last complete saved and
evaluated checkpoint is **64 updates** (`iter_0000063`). It has no 128-update
result. No further jobs were submitted after the stop request.

The current scope is six Chinese dialect/accent buckets and Chinese short
instruction utterances, without reference audio or explicit duration control.
Evaluation is held out from this RL dataset, not necessarily from teacher
pretraining. This is a single-seed pilot with automated ASR and audio-Judge
proxies; no same-budget SFT comparison or human blind listening was performed.
The teachers also include a preceding 200000-update VoiceClone phase that the
base student does not have.

## Reproducibility assets

The stable environment and model/data paths are listed in [INSPIRE.md](../../INSPIRE.md).

Full experiment root:

```text
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912
```

It contains `REPORT.md`, `EVIDENCE.md`, `final-comparisons.json`,
`learning-curves.png`, the listening page `audio-review.html` with referenced WAV files,
`data/manifest.json`, model/prompt audits, `run-audit.json`, `user-stop.json`
and `shutdown-verification.json`. Each run retains exact commands, source
versions, configuration, checkpoints and per-sample evidence.

Experiment scripts:

```text
/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912
```

Several runs reduced the sampled teacher/student logprob difference while
making generated audio less likely under the teacher and degrading text
accuracy. This is why the pipeline/throughput acceptance in the original
MOPD implementation log must remain separate from domain-quality acceptance.
