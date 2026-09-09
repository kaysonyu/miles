# MOSS-TTS Local 复合奖励

MOSS-TTS Local 的奖励入口支持把 WER、参考音频相似度（SIM）和音频奖励模型（RM/Judge）组合成一个 `[0, 1]` 的标量，供 Miles 的 GRPO 使用。实现参考了 `/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/slime` 中的 MOSS-TTS delay 奖励协议，并按 Miles 的 `custom_rm_path` 和 group-RM 接口重新组织。

默认配置保持现有 Local WER 训练行为：

```text
--moss-local-reward-components wer=1.0
```

launcher 在这个默认值下继续加载 `miles.policies.moss_tts_local.wer_reward.reward_func`。只有显式配置多个组件时才加载 `miles.policies.moss_tts_local.reward_composite.reward_func`，因此原来的纯 WER 命令、数据格式和 ASR 服务不需要修改。

## 组件和公式

组件用空格或逗号分隔的 `name=weight` 表示，权重必须是有限非负数，且总和为 1：

```text
--moss-local-reward-components "wer=0.4 sim=0.4 rm=0.2"
```

支持的规范名称和兼容别名如下：

| 规范名称 | 兼容别名 | 计算 |
| --- | --- | --- |
| `wer` | 无 | `1 - effective English WER` |
| `sim` | `reference_similarity`、`timbre_sim` | 参考音频和生成音频的 speaker-SIM |
| `rm` | `judge` | 每个 rubric 的 yes/no 概率平均值 |

每个样本的 `reward_components` 会保存各组件的权重、原始值和归一化后的值；`reward_formula` 保存实际公式，便于从 rollout 数据重算和审计。所有活动组件都必须返回有限的 `[0, 1]` 分数，组件异常会让当前 reward 请求失败，不会静默转成高分或零分。

## WER

WER 仍由 `wer_reward.py` 负责。它调用已部署的 Qwen3-ASR OpenAI-compatible 服务，默认进行三次识别并取按有效 WER 排序后的中位结果。非英语语言、解码长度截断和明显不合理的词速会被视为有效 WER=1，同时保留原始转写和原始 WER 供审计。

launcher 通过以下变量把服务配置传给 reward worker：

```text
MOSS_TTS_WER_ASR_URL
MOSS_TTS_WER_ASR_MODEL=qwen3-asr-1.7b
MOSS_TTS_WER_ASR_API_KEY_ENV=INSPIRE_API_KEY
MOSS_TTS_WER_ASR_API_KEY_FILE=/path/to/reward-secret.env
MOSS_TTS_WER_ASR_REPEATS=3
```

secret 文件只应通过路径传入，不能写入 checkpoint、镜像或源码。

## SIM

SIM 使用 `sim_wer_reward.score_similarity_batch`，按参考音频分组并批量请求 SIM 服务；生成的 WAV 会以 artifact 的 SHA256 命名写入共享目录，服务只读取该目录中的文件。配置示例：

```text
--moss-local-reward-components "wer=0.4 sim=0.6"
--sim-url https://sim.example/v1/similarities
```

对应环境变量为：

```text
MOSS_TTS_SIM_URL
MOSS_TTS_SIM_CANDIDATE_DIR=/absolute/shared/path
MOSS_TTS_SIM_API_KEY_ENV=INSPIRE_API_KEY
MOSS_TTS_SIM_API_KEY_FILE=/path/to/reward-secret.env
MOSS_TTS_SIM_EXPECTED_MODEL=wavlm_large_ecapa_tdnn
```

Slime 的 reference audio uses 分为四个维度：

```text
timbre
accent
prosody
emotion
```

当前 Miles 与已部署 SIM 服务真实可用的是 `timbre`。`sim_dimensions` 会显式记录四个维度，其中 `timbre` 标记为 `implemented=true`，其余三个维度保留 `implemented=false` 和零占位；这不会把未部署的服务能力误报成已实现。样本必须在 `metadata` 中提供以下两种格式之一：

```json
{
  "reference_audio_path": "/shared/reference.wav"
}
```

或：

```json
{
  "reference_audios": [
    {
      "id": "speaker-1",
      "path": "/shared/reference.wav",
      "uses": ["timbre"]
    }
  ]
}
```

同一个样本的 `timbre` reference 必须唯一且为绝对路径。没有 reference audio 时，WER-only 仍然合法；启用 SIM 时会明确报配置错误。

## RM / Judge

RM 采用 Slime Judge 的基本语义：每条 rubric 独立发起确定性的音频+文本请求，要求模型只回答 yes/no，并通过 targeted token logprob 计算 `p(yes)`；样本 RM 是所有 rubric item 的平均值。配置示例：

```text
--moss-local-reward-components "wer=0.5 rm=0.5"
--rm-url https://judge.example/v1/chat/completions
```

环境变量：

```text
MOSS_TTS_RM_URL
MOSS_TTS_RM_MODEL=qwen3-omni-30b-a3b-thinker
MOSS_TTS_RM_TOKENIZER_PATH=/absolute/path/to/AnyAudio-Judge-30B
MOSS_TTS_RM_API_KEY_ENV=INSPIRE_API_KEY
MOSS_TTS_RM_API_KEY_FILE=/path/to/reward-secret.env
```

RM 需要 `Sample.metadata["rubric"]` 为非空数组，每项至少包含 `dimension` 和 `statement`：

```json
{
  "rubric": [
    {
      "dimension": "prosody",
      "statement": "The delivery sounds calm and steady."
    }
  ]
}
```

没有 rubric 时不应启用 RM。RM 结果和各 item 分数会写入 `rm_rubric_scores`、`rm_model` 和 `rm_reward`。

## 服务部署边界

Slime 中的服务编排位于：

```text
slime/examples/mosstts_rl/reward_grpo/serving/
```

Miles reward client 只负责在训练进程中调用服务，不会把 ASR、SIM 或 Judge 模型打进 Miles 镜像，也不会自动创建 Inspire Serving。生产运行需要分别准备：

1. Qwen3-ASR 服务；
2. speaker-SIM `/v1/similarities` 服务；
3. AnyAudio Judge 的 OpenAI-compatible `/v1/chat/completions` 服务。

服务地址、模型名、tokenizer 路径和密钥通过 launcher 的字段或环境变量注入。当前本次真实端到端验证只覆盖 WER；SIM/RM 已完成客户端、metadata 和组合接口的 CPU 级验证，但没有在本次 WER 实验中伪造服务结果。

## 性能和 group RM

启用多个组件时建议使用 `--group-rm`，让 Miles 一次把同一组样本交给 `reward_composite.reward_batch`。这样 SIM 可以按 reference 分组并批量发送，WER/RM 也能并行调度。若不启用 group RM，Miles 会逐样本调用兼容的 `reward_func`，功能相同但服务请求并发和批量效率较低。

推荐先以 `wer=1.0` 建立固定基线，再逐步比较：

```text
wer=1.0
wer=0.5 sim=0.5
wer=0.4 sim=0.4 rm=0.2
```

比较时固定 checkpoint、prompt/seed、Omni CUDA Graph/Radix Cache 配置和服务版本，并保存每个 component 的诊断字段。WER、SIM、RM 的量纲和噪声不同，不能只比较训练期平均 reward；至少要分别报告 held-out WER、SIM、RM，以及生成延迟和显存。
