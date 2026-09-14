# MOSS-TTS Local adapter structure

Local uses the existing Miles rollout transport, DP scheduling and Megatron
optimizer loop. Its extensions are resolved by policy family; Omni transport is
selected separately by `rollout_backend`.

| Module | Responsibility |
| --- | --- |
| `miles/policies/registry.py` | Resolve text defaults or Local codec, data adapter, workflow, model and weight adapter through lazy imports |
| `miles/policies/base.py` | Dependency-light interfaces and the explicit `TrainingContext` / `TrainingRuntime` contract |
| `miles/policies/media.py` | Shared audio artifact validation and serialization |
| `moss_tts_local/data_schema.py` | Authoritative action and paired-score tensor names, dtypes, CPU transport and device preparation |
| `moss_tts_local/provenance.py` | Student/teacher origin, behavior version, student scoring version and teacher identity |
| `moss_tts_local/rollout_data.py` | Sample conversion, masks and existing DP scheduling |
| `moss_tts_local/workflow.py` | Trainer-local workflow and native teacher pool; replay/advantage preparation and Megatron invocation |
| `moss_tts_local/objectives.py` | GRPO, sampled or native MOPD inputs, sample-summed loss and metrics |
| `moss_tts_local/training_loss.py` | Selected-score collection, parity and Megatron microbatch/DP scaling |
| `megatron_utils/policy_runtime.py` | Versioned snapshot leases, restoration of current weights and profiler/backup finalization |
| `megatron_utils/pretrained_loader.py` | Configured model-only import before a native Miles resume checkpoint exists |
| `megatron_utils/optimizer_adapters.py` | StatefulTorchAdam binding during optimizer construction |
| `moss_tts_local/reward_composite.py` | Concurrent scorer execution and one publication of composed diagnostics |

**Interfaces and lifecycle**

Resolving text defaults imports neither Local nor Megatron. Resolving Local's
rollout data adapter imports no Megatron/Transformer Engine runtime. The actor
creates one workflow during initialization, which lazily owns its native teacher
pool. Each actor has its own workflow and pool.

`TrainingContext` supplies the model, optimizer, scheduler, rollout ID and runtime
interface. It does not expose the Ray actor. A behavior replay leases a versioned
snapshot; current actor weights are restored even if replay raises. Snapshot
labels advance after the existing trainer snapshot rotation and successful
publication. Bounded async and final-drain checkpoint restrictions are unchanged.

The registry's data and serving-weight adapters are stateless. The rollout data
adapter implements conversion, raw DP split, device preparation, version
validation and disposal. Miles still owns object-store handles and release.
Legacy function-based rollout resource disposal remains supported.

**Data and compatibility**

The serialized Local trajectory remains schema v2, with exact prompt rows,
continue/stop decisions and twelve audio codebooks. Dictionary transport and the
`weight_versions` field remain compatible. For teacher replay, that legacy field
still carries the student training/scoring version; immutable `SampleProvenance`
makes this selection explicit. Replay shards and debug dumps additionally expose
`student_scoring_versions` alongside `behavior_weight_versions` and
`trajectory_origins`.

`Sample.from_dict` selects the concrete trajectory codec using its existing
`model_family`. `MediaArtifact` remains importable from
`miles.policies.moss_tts_local.types`; old pickle imports resolve to the shared
class. Local optimizer and checkpoint helper paths remain compatibility imports,
as do the workflow's previous loss/collection helper names.

The model, local transformer, selected-action probability implementation, GRPO
and MOPD mathematical losses, native teacher implementation and native checkpoint
tensor mapper are retained. Sample averaging, terminal geometry, independent
embeddings/heads and strict optimizer resume checks keep their original semantics.
Native MOPD still performs paired remote scoring for the existing protocol and
diagnostics; this refactor adds no option to omit it.

Omni discovery receives the resolved policy spec, and refit receives the policy
weight adapter's expected names. The standalone discovery call retains its Local
default. Full refit still pauses generation, validates the manifest and published
versions, then resumes generation.

**Reward behavior**

WER and RM expose `score_sample` returning `RewardScore(value, metadata)` without
mutating the input. Historical `reward_func` entrypoints merge diagnostics and
return the scalar. Composite scoring awaits all components, validates and
composes the entire batch, then merges each sample's diagnostics once. A failed
component or invalid score cannot leave partially published composite metadata.

Composite WER remains bounded to `[0, 1]`; historical plain WER continues to
return `1-WER`, including negative values. The launcher selects plain WER from
parsed values, so `wer=1`, `wer=1.0` and `wer=1,sim=0` select the same original
command and environment. Calling the composite entrypoint directly still
explicitly selects bounded composition.

Rollout SIM metrics accept composite `reward` and legacy fixed SIM/WER
`mixed_reward`. The metric name `moss_tts_local/mixed_reward_mean` is retained.

**Validation**

Local tests cover codec compatibility, dependency isolation, adapter conformance,
paired-score preparation, snapshot rotation/restoration, per-trainer teacher
reuse, concurrent reward diagnostics, failed composition and SIM metrics.
Objective integration tests compare packed and separate microbatch gradients for
all five objectives, including an immediate-stop sample.

Run with the supported Miles/Megatron/SGLang dependencies:

```bash
python -m pytest tests/fast/moss_tts_local tests/fast/launch_scripts tests/manual/launch_scripts
python -m pytest tests/fast/utils/test_weight_version.py tests/fast/utils/test_dp_schedule.py tests/fast/ray/rollout/test_train_data_conversion.py
```

CPU tests and synthetic loss equivalence do not replace GPU forward, refit,
checkpoint continuation or serving-parity validation for a deployment.
