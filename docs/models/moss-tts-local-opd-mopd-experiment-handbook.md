# MOSS-TTS Local OPD / MOPD 实验全记录：路径、配置、实验与结果

整理日期：2026-09-14。本文根据保存的`args.json`、`commands.json`、训练日志、检查点、评估结果及审计文件生成。覆盖工程验证、首轮采样代理损失、原生完整分布KL、单域/双域确认、A–L修复和自动评估。本文是文档整理，没有新增训练或改变既有结果。

**主要结论：原生反向KL支持从Base162500向同一学生迁移方言和指令能力，但方言内容质量存在代价。同提示约20%教师轨迹混合的H有两套ASR一致支持的修复收益；H属于混合GKD。上海话仍有残留问题，未通过联合确认筛选，未进行额外两个训练种子或新保留集确认。**

## 目录

- [1. 范围与计数规则](#1-范围与计数规则)
- [2. 环境和总路径](#2-环境和总路径)
- [3. 模型来源和预训练配置](#3-模型来源和预训练配置)
- [4. 数据与教师轨迹池](#4-数据与教师轨迹池)
- [5. 蒸馏方法与共同配置](#5-蒸馏方法与共同配置)
- [6. 分阶段实验总表](#6-分阶段实验总表)
- [7. 自动评估配置与覆盖](#7-自动评估配置与覆盖)
- [8. 结果与判定](#8-结果与判定)
- [9. 复现入口与证据定位](#9-复现入口与证据定位)
- [10. 状态、失败和未执行项目](#10-状态失败和未执行项目)
- [11. 每个实验的完整配置与产物](#11-每个实验的完整配置与产物)

## 1. 范围与计数规则

共索引 **63个运行记录**，其中包含训练、只评估、教师候选生成、基线及失败/中止尝试。它不是成功模型的数量。`steps`通常是累计目标计数，`start_step`是恢复起点，本次新增更新应看审计或日志。例：128→192只新增64次；`mode=evaluate,start_step=steps`新增0次。检查点`iter_0000127`代表128次后训练更新，目录采用零基计数。

P0仅验证工程链路，教师为兼容的Base占位模型，正信号学生来自已有WER训练模型副本，不能算作领域迁移证据。P1–P4的领域研究学生祖先均为指定Base162500。P1温度1/96样本、P2温度0.7/192样本、P3–P4温度0.7/288开发样本必须分别解释，不能跨表直接比较。

## 2. 环境和总路径

镜像：`docker.sii.shaipower.online/inspire-studio/miles-moss-tts-local-env:20260908-cu130-v1`。项目为CQ项目；按用户要求停止视觉组任务后，P2–P4使用CQ-科研驾驶舱中指定的`train`八卡H200交互式节点。P0是更早的`dev`四卡工程验证。这里记录历史运行资源，不表示Notebook当前状态或当前可用卡数。

| 阶段 | 产物根目录 | 驱动/日志根目录 |
| --- | --- | --- |
| P0 工程链路与吞吐验证 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909) | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909) |
| P1 真实领域教师：采样代理损失 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912) | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912) |
| P2 原生完整分布KL与单域/联合确认 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2) | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2) |
| P3 第一轮质量修复A/B/C | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality) | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality) |
| P4 全面对照D–L、父模型及自动探针 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive) | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive) |

下文使用P0–P4指代上述阶段。逐运行目录与每个检查点均给出可点击完整绝对路径。

| 资产 | 路径 |
| --- | --- |
| miles | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/miles](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/miles) |
| sglang-omni | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/sglang-omni](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/sglang-omni) |
| sglang | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/sglang](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/sglang) |
| Megatron-LM | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/Megatron-LM](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/Megatron-LM) |
| mossLite | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/mossLite](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/mossLite) |
| 占卡程序 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/bin/gpu-occupy](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/bin/gpu-occupy) |

P2–P4的一整节点分工：GPU0/1原生训练，GPU2–5学生生成，GPU6/7承载评分服务与两个AnyAudio-Judge工作进程；独立Whisper阶段改为每卡一个ASR工作进程。开始前释放`gpu-occupy`，suite退出后恢复0–7。`--num-gpus-per-node 2`在原生训练命令里指训练actor使用的两卡，不表示实验只分配两卡。

## 3. 模型来源和预训练配置

既有训练血缘为Qwen3-4B转换为MOSS Local（Global+Local decoder+12路RVQ），Base阶段取162500步，再经VC200000步训练两个领域教师。后训练学生直接从Base162500开始，VC只作为教师祖先与独立对照；指令v0.0.0直接训练分支没有用作本研究教师。以下预训练RUN是输入资产，不是本次由代理重新训练的任务。

### Base TTS

RUN：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0)

训练配方：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/config/recipe.sh](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/config/recipe.sh)

数据配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/config/data_config.jsonl](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/config/data_config.jsonl)

原生检查点：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints/iter_0162500](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints/iter_0162500)

原生模型DCP：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints/iter_0162500/model](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints/iter_0162500/model)

### VC父模型

RUN：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.0](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.0)

训练配方：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.0/config/recipe.sh](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.0/config/recipe.sh)

数据配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.0/config/data_config.jsonl](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.0/config/data_config.jsonl)

原生检查点：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.0/checkpoints/iter_0200000](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.0/checkpoints/iter_0200000)

原生模型DCP：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.0/checkpoints/iter_0200000/model](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.0/checkpoints/iter_0200000/model)

### 方言教师

RUN：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/xyzhang/MOSS-TTS-2_0/mosslite_local/RUNs/delivery/pretrain/local_pretrain_dialect_v0.0.1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/xyzhang/MOSS-TTS-2_0/mosslite_local/RUNs/delivery/pretrain/local_pretrain_dialect_v0.0.1)

训练配方：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/xyzhang/MOSS-TTS-2_0/mosslite_local/RUNs/delivery/pretrain/local_pretrain_dialect_v0.0.1/config/recipe.sh](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/xyzhang/MOSS-TTS-2_0/mosslite_local/RUNs/delivery/pretrain/local_pretrain_dialect_v0.0.1/config/recipe.sh)

数据配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/xyzhang/MOSS-TTS-2_0/mosslite_local/RUNs/delivery/pretrain/local_pretrain_dialect_v0.0.1/config/data_config.jsonl](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/xyzhang/MOSS-TTS-2_0/mosslite_local/RUNs/delivery/pretrain/local_pretrain_dialect_v0.0.1/config/data_config.jsonl)

原生检查点：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/xyzhang/MOSS-TTS-2_0/mosslite_local/RUNs/delivery/pretrain/local_pretrain_dialect_v0.0.1/checkpoints/iter_0006000](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/xyzhang/MOSS-TTS-2_0/mosslite_local/RUNs/delivery/pretrain/local_pretrain_dialect_v0.0.1/checkpoints/iter_0006000)

原生模型DCP：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/xyzhang/MOSS-TTS-2_0/mosslite_local/RUNs/delivery/pretrain/local_pretrain_dialect_v0.0.1/checkpoints/iter_0006000/model](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/xyzhang/MOSS-TTS-2_0/mosslite_local/RUNs/delivery/pretrain/local_pretrain_dialect_v0.0.1/checkpoints/iter_0006000/model)

### 指令教师

RUN：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/wchang/code/mossLite/RUNs/delivery/pretrain/local_instruction_v0.0.1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/wchang/code/mossLite/RUNs/delivery/pretrain/local_instruction_v0.0.1)

训练配方：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/wchang/code/mossLite/RUNs/delivery/pretrain/local_instruction_v0.0.1/config/recipe.sh](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/wchang/code/mossLite/RUNs/delivery/pretrain/local_instruction_v0.0.1/config/recipe.sh)

数据配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/wchang/code/mossLite/RUNs/delivery/pretrain/local_instruction_v0.0.1/config/data_config.jsonl](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/wchang/code/mossLite/RUNs/delivery/pretrain/local_instruction_v0.0.1/config/data_config.jsonl)

原生检查点：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/wchang/code/mossLite/RUNs/delivery/pretrain/local_instruction_v0.0.1/checkpoints/iter_0006000](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/wchang/code/mossLite/RUNs/delivery/pretrain/local_instruction_v0.0.1/checkpoints/iter_0006000)

原生模型DCP：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/wchang/code/mossLite/RUNs/delivery/pretrain/local_instruction_v0.0.1/checkpoints/iter_0006000/model](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/wchang/code/mossLite/RUNs/delivery/pretrain/local_instruction_v0.0.1/checkpoints/iter_0006000/model)

| HF/导出资产 | 路径 |
| --- | --- |
| 用户指定方言HF | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/zczhang/moss-eval-runs/automations/moss_checkpoint_watch/local_pretrain_dialect_v0.0.1/hf_ckpts/shared/0006000](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/zczhang/moss-eval-runs/automations/moss_checkpoint_watch/local_pretrain_dialect_v0.0.1/hf_ckpts/shared/0006000) |
| 原训练目录方言HF副本 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/xyzhang/MOSS-TTS-2_0/mosslite_local/ckpts/local_pretrain_dialect_v0.0.1_iter6000](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/xyzhang/MOSS-TTS-2_0/mosslite_local/ckpts/local_pretrain_dialect_v0.0.1_iter6000) |
| 用户指定指令HF（历史来源） | `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/wchang/code/mossLite/RUNs/delivery/pretrain/local_instruction_v0.0.1/hf/iter_0006000`（整理时不可访问） |
| 实际serving base-162500 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/base-162500](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/base-162500) |
| 实际serving dialect-6000 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/dialect-6000](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/dialect-6000) |
| 实际serving instruction-6000 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/instruction-6000](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/instruction-6000) |
| 实际serving VC父模型 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/vc-200000](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/vc-200000) |
| tokenizer/processor runtime来源 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-TTS-Local-Transformer-v1.5](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-TTS-Local-Transformer-v1.5) |

2026-09-14整理时，指令教师原HF目录不可访问；原生6000步检查点及本实验的serving导出仍可访问。历史[教师导出审计](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/teacher-export-audit.json)记录两个教师各438个张量与当时指定HF逐元素一致。不能将当前路径缺失误写为当时没有校验。每个serving导出中的`sglang_omni_conversion.json`记录native来源、iteration、模型合同及manifest哈希。

### 关键后训练检查点速查

| 模型 | 原生后训练检查点 | 用途/边界 |
| --- | --- | --- |
| 方言单OPD64 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/checkpoints/iter_0000063](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/checkpoints/iter_0000063) | 方言单域确认，原生反向KL，训练温度0.7 |
| 指令单OPD64 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/checkpoints/iter_0000063](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/checkpoints/iter_0000063) | 指令确认任务只加载本检查点评估 |
| 纯MOPD128 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints/iter_0000127](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints/iter_0000127) | A–L共享续训起点 |
| H混合GKD192 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/checkpoints/iter_0000191](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/checkpoints/iter_0000191) | 方言修复研究候选，未通过跨ASR确认筛选 |
| L分领域组合192 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/checkpoints/iter_0000191](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/checkpoints/iter_0000191) | 指令内容较好，方言门槛未通过 |

### 两个既有教师recipe中的关键参数

| 教师 | 迭代 | GB | micro | LR | WD | warmup | LR schedule | WSD decay迭代 | seq len | save interval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 方言教师 | 6000 | 128 | 1 | 1e-4 | 0.1 | 0.033 | WSD | 4800 | 20480 | 2500 |
| 指令教师 | 6000 | 32 | 1 | 2e-4 | 0.1 | 0.0 | WSD | 6000 | 20480 | 1000 |

两位教师都从VC200000步`START_MODE=pretrained`初始化；recipe使用bf16、TP/PP/CP=1、Local层chunk4096、block recompute8。通道权重为`[3,3,3,3,3,2,2,2,2,1,1,1,1]`。这里摘录的是整理时保存的recipe内容，完整历史启动仍以对应RUN记录为准。

## 4. 数据与教师轨迹池

| 阶段数据 | 目录 | 每域train | 每域validation/development | 每域final |
| --- | --- | --- | --- | --- |
| P1初始筛选 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/data](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/data) | 2304 | 96 | 96 |
| P2新保留集 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout) | 2304 | 96 | 192；当时已评估 |
| P3修复 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data) | 2304 | 288=此前96+192 | 192；封存 |
| P4全面对照 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data) | 2304 | 288 | 192；封存，未评估 |

初筛：文本8–65个规范化字符、单音频、参考38–110帧、12路codes合法；仅短单人、无参考音频输入。训练JSONL不含参考codes，评估JSONL保留参考codes。记录text_hash/source_group_hash、源Lance版本/行号、domain/bucket和tts_params。只保证本研究RL及评估抽样的隔离，不保证教师预训练未见，也不保证说话人隔离。

每个数据目录含`dialect-{train,validation,eval}.jsonl`、`instruction-{train,validation,eval}.jsonl`和对应`mixed-*.jsonl`。方言六类均衡，指令normal/denoise两类均衡。

| 领域/配置名 | 源Lance目录 | 固定版本 | 源记录数 |
| --- | --- | --- | --- |
| dialect/dialect/chuan/ge_3s_lt_10s_tc0 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-Train-2.0/Dialect/Base/Pack1/chuan/ge_3s_lt_10s](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-Train-2.0/Dialect/Base/Pack1/chuan/ge_3s_lt_10s) | 5 | 2463911 |
| dialect/accent_mandarin/chuan_mandarin/ge_3s_lt_10s_tc0 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-Train-2.0/Dialect/Base/Pack1/chuan_mandarin/ge_3s_lt_10s](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-Train-2.0/Dialect/Base/Pack1/chuan_mandarin/ge_3s_lt_10s) | 5 | 35623 |
| dialect/dialect/dongbei/ge_3s_lt_10s_tc0 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-Train-2.0/Dialect/Base/Pack1/dongbei/ge_3s_lt_10s](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-Train-2.0/Dialect/Base/Pack1/dongbei/ge_3s_lt_10s) | 5 | 41136 |
| dialect/accent_mandarin/dongbei_mandarin/ge_3s_lt_10s_tc0 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-Train-2.0/Dialect/Base/Pack1/dongbei_mandarin/ge_3s_lt_10s](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-Train-2.0/Dialect/Base/Pack1/dongbei_mandarin/ge_3s_lt_10s) | 5 | 12766 |
| dialect/dialect/guangdong_yue/ge_3s_lt_10s_tc0 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-Train-2.0/Dialect/Base/Pack1/guangdong_yue/ge_3s_lt_10s](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-Train-2.0/Dialect/Base/Pack1/guangdong_yue/ge_3s_lt_10s) | 3 | 8920183 |
| dialect/dialect/shanghai/ge_3s_lt_10s_tc0 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-Train-2.0/Dialect/Base/Pack1/shanghai/ge_3s_lt_10s](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-Train-2.0/Dialect/Base/Pack1/shanghai/ge_3s_lt_10s) | 5 | 658327 |
| instruction/Instruction/denoise-zh/ge_3s_lt_10s_tc0 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-Train-2.0/Instruction/denoise/zh/ge_3s_lt_10s](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-Train-2.0/Instruction/denoise/zh/ge_3s_lt_10s) | 3 | 280973 |
| instruction/Instruction/normal-zh/ge_3s_lt_10s_tc0 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-Train-2.0/Instruction/normal/zh/ge_3s_lt_10s](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-Train-2.0/Instruction/normal/zh/ge_3s_lt_10s) | 3 | 79864 |

[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/data/manifest.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/data/manifest.json)

[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/fresh-holdout-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/fresh-holdout-audit.json)

[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data/fresh-final-manifest.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data/fresh-final-manifest.json)

[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/final-isolation-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/final-isolation-audit.json)

[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/fresh-final-manifest.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/fresh-final-manifest.json)

### 教师轨迹池：B/J与H/I不同

| 池 | 路径 | 配置与用途 |
| --- | --- | --- |
| B/J筛选池 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-replay/manifest.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-replay/manifest.json) | 1536候选选384：方言每类32，指令每类96；自然停止、1..192帧、CER≤0.15、ASR完成、Judge非空，再按桶内Judge降序。B用教师轨迹；J仅取同样提示再由学生生成 |
| H/I精确位置池 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-replay/manifest.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-replay/manifest.json) | 820条，temperature0.7；selection=exact_sample；不做质量筛选；保留原始停止/截断状态；同一个缓存供H/I使用 |
| 原始续训位置表 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-schedule.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-schedule.json) | 4096个原始轨迹位置；按每5个提示组一组安排教师轨迹 |
| 教师候选提示 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/teacher-candidate-prompts.jsonl](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/teacher-candidate-prompts.jsonl) | 820个指定位置的原文本、指令、seed和采样index |
| J与B提示一致性 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/prompt-contrast-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/prompt-contrast-audit.json) | 4096条提示/指令一致，820个选择位置一致，J所有轨迹来自当前学生 |

H/I池由方言424条自然停止、指令352条自然停止+44条长度截断组成。H实际4096条续训轨迹中3276条为学生、820条为教师。它是混合GKD，教师行为版本不会伪装成当前学生版本。换完整训练seed时必须按真实新游标重新生成精确位置缓存，不能复用旧seed的820条。

## 5. 蒸馏方法与共同配置

| 实现 | 轨迹/评分 | 优化信号 | 结论边界 |
| --- | --- | --- | --- |
| sampled | 学生轨迹，serving teacher/student paired-prefill | 所选动作logprob差及裁剪代理损失 | P1未通过双域质量标准，仍是保留的默认路径 |
| sampled_native | 相同packed动作上的原生教师/学生 | 所选动作代理损失 | 用于区分评分实现与损失形式 |
| dense_reverse | 原生完整logits | KL(student &#124;&#124; teacher) | 温度0.7时有领域迁移证据 |
| dense_forward | 原生完整logits | KL(teacher &#124;&#124; student) | 在当前实验中更利指令内容，方言代价较大 |

每条样本按`metadata.domain`选择一个冻结教师，不平均两个教师的logits。动作包含continue/stop及12路RVQ，终止stop没有RVQ动作。标准OPD/MOPD用学生生成轨迹；B/H/I含明确教师轨迹。KL、CER和Judge的角色不同：训练优化KL/代理损失；CER/Judge用于评估，并用于B池的候选筛选，不是这些KL实验的在线奖励。

| 配置项 | P2正式单域/MOPD与P3/P4正式修复 |
| --- | --- |
| 训练范围 | full；仅pilot-dialect-reverse-local-r1是local_only |
| 优化器 | Adam/use-torch-adam，constant LR3e-6，weight decay0，clip-grad1 |
| 批量 | 16提示×4轨迹=GB64；动态微批，micro-batch-size1，max-samples-per-microbatch2 |
| 并行 | 2个训练GPU，TP=PP=CP=1；packed THD |
| 长度 | 最多192音频帧；训练上下文/seq-length1024，max-tokens-per-gpu1024 |
| 采样 | top-p1，top-k=-1；正式迁移/修复训练温度0.7 |
| seed | rollout-seed20260956；P3/P4显式训练seed1234；未显式保存的旧参数见附录，不凭当前默认值反推 |
| 评估温度 | P2主路径配置1.0，并另外评0.7；报告迁移结论用0.7；P3/P4统一0.7 |
| 修复起点 | P2/mopd-reverse-t07-128-r1/checkpoints/iter_0000127；连同优化器、调度器和数据游标恢复 |
| 修复终点 | start_step128，steps192，新增64次；每32次保存/评估，完整终点288条/域 |
| 推理副本 | 4个学生生成副本+1个当前学生评分副本；330学习参数张量同步核验 |

旧命令可能先包含默认`--rollout-temperature 1`和`--clip-grad 0`，再由extra-args追加0.7和1.0；附录按最后出现的参数记录。`mode=baseline/evaluate`中的steps/LR等默认字段不代表实际训练。

### P4分领域配置原文

[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-loss/K.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-loss/K.json)

```json
{
  "dialect": {
    "estimator": "dense_reverse"
  },
  "instruction": {
    "estimator": "dense_forward"
  }
}
```

[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-loss/L.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-loss/L.json)

```json
{
  "dialect": {
    "estimator": "dense_reverse",
    "prefix_frames": 0,
    "prefix_weight": 4,
    "codebook_weights": [
      4.0,
      4.0,
      4.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    "decision_weight": 1
  },
  "instruction": {
    "estimator": "dense_forward",
    "prefix_frames": 0,
    "prefix_weight": 4,
    "codebook_weights": null,
    "decision_weight": 1
  }
}
```

K/L由`--moss-local-domain-loss-config`选择，配置中的estimator覆盖每条对应domain的实际损失方向。L中的prefix_frames=0，因此prefix_weight=4不激活前缀加权；实际方言加权来自codebook_weights。码本权重按每样本归一化；决策权重按完整动作权重和归一化。分领域配置保存在`checkpoints/native_domain_loss.json`并在resume核验。

## 6. 分阶段实验总表

以下表格只给便于对比的参数。所有运行的完整args、实际命令、所有保存检查点和评估位置在第11节。空值表示该次记录未显式提供，不能理解为0。

### P0 工程链路与吞吐验证

| 运行 | 模式 | 起点→计划终点 | 实际新增更新 | LR / GB | 损失 / 训练温度 | 运行状态 |
| --- | --- | --- | --- | --- | --- | --- |
| [benchmark-mopd-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-mopd-r1) | train | 0→8 | 8 | 0.0 / 64 | sampled/工程对照 / 1 | 完成 |
| [benchmark-wer-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-wer-r1) | train | 0→8 | 8 | 0.0 / 64 | WER-GRPO / 1 | 完成 |
| [paired-resume-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-resume-r1) | train | 2→4 | 2 | 3e-06 / 4 | sampled/工程对照 / 1 | 完成 |
| [paired-smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-smoke-r1) | train | 0→2 | 2 | 3e-06 / 4 | sampled/工程对照 / 1 | 训练完成，收尾解析失败 |
| [positive-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-r1) | train | 0→4 | 4 | 3e-06 / 8 | sampled/工程对照 / 1 | 完成 |
| [positive-resume-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-resume-r1) | train | 4→6 | 2 | 3e-06 / 8 | sampled/工程对照 / 1 | 完成 |
| [refactor-mopd-r2](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/refactor-mopd-r2) | train | 0→4 | 4 | 3e-06 / 8 | sampled/工程对照 / 1 | 完成 |
| [smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/smoke-r1) | train | 0→2 | 2 | 3e-06 / 4 | sampled/工程对照 / 1 | 完成 |

### P1 真实领域教师：采样代理损失

| 运行 | 模式 | 起点→计划终点 | 实际新增更新 | LR / GB | 损失 / 训练温度 | 运行状态 |
| --- | --- | --- | --- | --- | --- | --- |
| [baselines8-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/baselines8-r1) | baseline | 0→64 | 0 | 不适用 | 无训练损失 | 完成 |
| [mopd-mixed128-lr1e6](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6) | train | 0→128 | 128 | 1e-06 / 64 | sampled/工程对照 / 1 | 完成 |
| [mopd-mixed128-lr3e6-r2](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2) | train | 0→128 | 78 | 3e-06 / 64 | sampled/工程对照 / 1 | 未完成/中止 |
| [mopd-mixed128-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-r1) | train | 0→128 | 0 | 3e-06 / 64 | sampled/工程对照 / 未显式记录 | 训练前取消 |
| [opd-dialect128-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1) | train | 64→128 | 64 | 3e-06 / 64 | sampled/工程对照 / 1 | 完成 |
| [opd-dialect64-lr1e6](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6) | train | 0→64 | 64 | 1e-06 / 64 | sampled/工程对照 / 1 | 完成 |
| [opd-dialect64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1) | train | 0→64 | 64 | 3e-06 / 64 | sampled/工程对照 / 1 | 完成 |
| [opd-instruction128-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r1) | train | 64→128 | 未记录 | 3e-06 / 64 | sampled/工程对照 / 未显式记录 | 未完成/中止 |
| [opd-instruction128-r2](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2) | train | 64→128 | 64 | 3e-06 / 64 | sampled/工程对照 / 1 | 完成 |
| [opd-instruction64-lr1e6](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6) | train | 0→64 | 64 | 1e-06 / 64 | sampled/工程对照 / 1 | 完成 |
| [opd-instruction64-lr3e7](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7) | train | 0→64 | 64 | 3e-07 / 64 | sampled/工程对照 / 1 | 完成 |
| [opd-instruction64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1) | train | 0→64 | 64 | 3e-06 / 64 | sampled/工程对照 / 1 | 完成 |
| [overlap8-smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/overlap8-smoke-r1) | train | 0→2 | 2 | 3e-06 / 64 | sampled/工程对照 / 1 | 完成 |
| [smoke8-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r1) | train | 0→2 | 0 | 3e-06 / 8 | sampled/工程对照 / 1 | 未完成/中止 |
| [smoke8-r2](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r2) | train | 0→2 | 2 | 3e-06 / 8 | sampled/工程对照 / 1 | 完成 |

### P2 原生完整分布KL与单域/联合确认

| 运行 | 模式 | 起点→计划终点 | 实际新增更新 | LR / GB | 损失 / 训练温度 | 运行状态 |
| --- | --- | --- | --- | --- | --- | --- |
| [baselines-fresh-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1) | baseline | 0→64 | 0 | 不适用 | 无训练损失 | 完成 |
| [confirm-dialect-reverse-t07-64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1) | train | 32→64 | 32 | 3e-06 / 64 | dense_reverse / 0.7 | 完成 |
| [confirm-instruction-reverse-t07-64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1) | evaluate | 64→64 | 0 | 不适用 | 无训练损失 | 完成 |
| [dense-zero-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1) | train | 0→2 | 2 | 0.0 / 8 | dense_reverse / 1 | 完成 |
| [mopd-reverse-t07-128-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1) | train | 0→128 | 128 | 3e-06 / 64 | dense_reverse / 0.7 | 完成 |
| [pilot-dialect-dense-forward-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1) | train | 0→32 | 32 | 3e-06 / 64 | dense_forward / 1 | 完成 |
| [pilot-dialect-dense-reverse-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1) | train | 0→32 | 32 | 3e-06 / 64 | dense_reverse / 1 | 完成 |
| [pilot-dialect-reverse-local-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1) | train | 0→32 | 32 | 3e-06 / 64 | dense_reverse / 1.0 | 完成 |
| [pilot-dialect-reverse-t07-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1) | train | 0→32 | 32 | 3e-06 / 64 | dense_reverse / 0.7 | 完成 |
| [pilot-dialect-sampled-native-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1) | train | 0→32 | 32 | 3e-06 / 64 | sampled_native / 1 | 完成 |
| [pilot-instruction-reverse-t07-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1) | train | 0→64 | 64 | 3e-06 / 64 | dense_reverse / 0.7 | 完成 |

### P3 第一轮质量修复A/B/C

| 运行 | 模式 | 起点→计划终点 | 实际新增更新 | LR / GB | 损失 / 训练温度 | 运行状态 |
| --- | --- | --- | --- | --- | --- | --- |
| [prefix-smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1) | train | 128→130 | 2 | 3e-06 / 8 | dense_reverse / 0.7 | 完成 |
| [repair-A-mopd64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1) | train | 128→192 | 64 | 3e-06 / 64 | dense_reverse / 0.7 | 完成 |
| [repair-B-replay64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1) | train | 128→192 | 64 | 3e-06 / 64 | dense_reverse / 0.7 | 完成 |
| [repair-C-prefix64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1) | train | 128→192 | 64 | 3e-06 / 64 | dense_reverse / 0.7 | 完成 |
| [replay-smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r1) | train | 128→130 | 0 | 3e-06 / 8 | dense_reverse / 0.7 | 未完成/中止 |
| [replay-smoke-r2](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2) | train | 128→130 | 2 | 3e-06 / 8 | dense_reverse / 0.7 | 完成 |
| [teacher-pool-candidates](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates) | teacher_pool | 不适用 | 0 | 不适用 | 无训练损失 | 未完成/中止 |
| [teacher-pool-candidates-r2](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates-r2) | teacher_pool | 不适用 | 0 | 不适用 | 无训练损失 | 完成 |

### P4 全面对照D–L、父模型及自动探针

| 运行 | 模式 | 起点→计划终点 | 实际新增更新 | LR / GB | 损失 / 训练温度 | 运行状态 |
| --- | --- | --- | --- | --- | --- | --- |
| [D-forward64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1) | train | 128→192 | 64 | 3e-06 / 64 | dense_forward / 0.7 | 完成 |
| [E-rvq1-64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1) | train | 128→192 | 64 | 3e-06 / 64 | dense_reverse / 0.7 | 完成 |
| [F-rvq123-64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1) | train | 128→192 | 64 | 3e-06 / 64 | dense_reverse / 0.7 | 完成 |
| [G-decision64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1) | train | 128→192 | 64 | 3e-06 / 64 | dense_reverse / 0.7 | 完成 |
| [H-matched-reverse64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1) | train | 128→192 | 64 | 3e-06 / 64 | dense_reverse / 0.7 | 完成 |
| [I-matched-forward64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1) | train | 128→192 | 64 | 3e-06 / 64 | dense_forward / 0.7 | 完成 |
| [J-selected-prompt64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1) | train | 128→192 | 64 | 3e-06 / 64 | dense_reverse / 0.7 | 完成 |
| [K-domain-direction64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1) | train | 128→192 | 64 | 3e-06 / 64 | 分领域KL，见K/L配置 / 0.7 | 完成 |
| [L-domain-composition64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1) | train | 128→192 | 64 | 3e-06 / 64 | 分领域KL，见K/L配置 / 0.7 | 完成 |
| [baseline-experts-probes](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes) | baseline | 0→64 | 0 | 不适用 | 无训练损失 | 完成 |
| [baseline-vc-parent](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-vc-parent) | baseline | 0→64 | 0 | 不适用 | 无训练损失 | 完成 |
| [domain-smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1) | train | 128→130 | 2 | 3e-06 / 8 | 分领域KL，见K/L配置 / 0.7 | 完成 |
| [matched-smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1) | train | 128→130 | 2 | 3e-06 / 64 | dense_forward / 0.7 | 完成 |
| [matched-teacher-cache-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-teacher-cache-r1) | teacher_pool | 不适用 | 0 | 不适用 | 无训练损失 | 完成 |
| [parent-vc128-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1) | train | 0→128 | 128 | 3e-06 / 64 | dense_reverse / 0.7 | 完成 |
| [probe-A](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A) | evaluate | 192→192 | 0 | 不适用 | 无训练损失 | 完成 |
| [probe-B](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B) | evaluate | 192→192 | 0 | 不适用 | 无训练损失 | 完成 |
| [probe-C](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C) | evaluate | 192→192 | 0 | 不适用 | 无训练损失 | 完成 |
| [probe-mopd128](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128) | evaluate | 128→128 | 0 | 不适用 | 无训练损失 | 完成 |
| [prompt-smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1) | train | 128→130 | 2 | 3e-06 / 8 | dense_reverse / 0.7 | 完成 |
| [weighting-smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1) | train | 128→130 | 2 | 3e-06 / 8 | dense_reverse / 0.7 | 完成 |

### 关键对照差异速查

| 组 | 唯一主要变化/用途 |
| --- | --- |
| repair-A-mopd64-r1 | A：原反向KL继续64次 |
| repair-B-replay64-r1 | B：筛选提示与教师轨迹混合20% |
| repair-C-prefix64-r1 | C：前8帧RVQ权重4，总RVQ权重归一化 |
| D-forward64-r1 | D：全前向KL，纯学生轨迹 |
| E-rvq1-64-r1 | E：RVQ1原始权重4，其余1 |
| F-rvq123-64-r1 | F：前3个RVQ原始权重4，其余1 |
| G-decision64-r1 | G：continue/stop权重4 |
| H-matched-reverse64-r1 | H：20%精确同提示教师轨迹，反向KL；混合GKD |
| I-matched-forward64-r1 | I：与H完全相同缓存和位置，改为前向KL |
| J-selected-prompt64-r1 | J：与B提示选择一致，全部轨迹由当前学生生成 |
| K-domain-direction64-r1 | K：方言反向KL，指令前向KL，统一权重 |
| L-domain-composition64-r1 | L：方言反向KL+前3码本加权，指令前向KL |
| parent-vc128-r1 | Base162500由共同VC200000父模型指导128次；不是从VC初始化学生 |

P4主筛选D–J固定后运行；K/L在看到领域取舍后追加，属于有时间戳记录的探索。704次是P4的9×64+128正式更新，不是整个项目所有阶段的总更新，也不能把中途checkpoint重复算成新训练。

## 7. 自动评估配置与覆盖

### 方言与指令属性：实际评测覆盖范围（9月14日补充）

本节直接核对Base、MOPD128、H、L和对应领域教师的逐条`-judge.json`记录，并逐项核验评分问题与生成指令对应。每个模型方言288条、指令288条均有完整评分；每个模型包含288项方言判断和1458项指令属性判断。没有新增模型运行或改变已有分数。

**方言：6类各48条，逐条判断目标方言/口音是否符合。指令：固定10类属性白名单，每条最多选6项，按白名单顺序优先选择。白名单中的缺失属性不会被补造为评分问题。** 因此“288条指令音频均有评分”不等于“每条指令的全部要求都被评过”。

| 属性 | 每模型实际项数 | Base | MOPD128 | H | L | 对应领域教师 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 口音/方言 | 288 | 0.2104 | 0.2712 | 0.2968 | 0.2847 | 0.4567 |
| 语速 | 115 | 0.3556 | 0.4595 | 0.4428 | 0.4861 | 0.4897 |
| 音高 | 93 | 0.5347 | 0.6564 | 0.6185 | 0.6495 | 0.6047 |
| 发声模式 | 102 | 0.4975 | 0.5225 | 0.4764 | 0.5285 | 0.7327 |
| 情绪与心理状态 | 235 | 0.7296 | 0.8042 | 0.7975 | 0.8321 | 0.8897 |
| 韵律变化 | 211 | 0.5030 | 0.6186 | 0.6286 | 0.6859 | 0.7792 |
| 音色明暗 | 141 | 0.4163 | 0.4866 | 0.4800 | 0.5143 | 0.6679 |
| 音色厚薄 | 95 | 0.4781 | 0.4610 | 0.4794 | 0.4934 | 0.7061 |
| 音色风格 | 82 | 0.4399 | 0.5960 | 0.5566 | 0.5933 | 0.8541 |
| 表演模式腔调 | 194 | 0.5266 | 0.5203 | 0.5441 | 0.5739 | 0.6953 |
| 性别 | 190 | 0.4996 | 0.5238 | 0.5632 | 0.5204 | 0.9392 |

表内为每一维度所有已评项目的平均p(yes)。此前的整体指令Judge先在每条音频内平均，再对音频平均；由于各条属性数不同，不能直接把本表10个维度均值再平均来复现整体分数。分数是自动模型输出，不是人工准确率，也不是已校准的成功概率。

**未覆盖或仅部分覆盖的字段：** 响度、年龄、节奏、声带紧张度、音域宽度、咬字清晰度等不在白名单内；即使在白名单内，也可能因为每条最多6项而未被选中。下面按同一批288条指令统计；未评数来自评分选择规则，不是评分进程漏跑。

| 指令字段 | 出现在输入中的次数 | 实际评测次数 | 未评次数 |
| --- | ---: | ---: | ---: |
| 交际行为 | 102 | 0 | 102 |
| 人格与性格特征 | 54 | 0 | 54 |
| 共鸣位置 | 62 | 0 | 62 |
| 发声模式 | 102 | 102 | 0 |
| 口音与方言 | 47 | 0 | 47 |
| 咬字清晰度 | 24 | 0 | 24 |
| 响度 | 86 | 0 | 86 |
| 声带紧张度 | 81 | 0 | 81 |
| 年龄 | 100 | 0 | 100 |
| 态度与表态 | 126 | 0 | 126 |
| 性别 | 263 | 190 | 73 |
| 情绪与心理状态 | 235 | 235 | 0 |
| 粗糙度 | 57 | 0 | 57 |
| 职业身份角色 | 78 | 0 | 78 |
| 节奏 | 109 | 0 | 109 |
| 表演模式腔调 | 230 | 194 | 36 |
| 表达策略 | 147 | 0 | 147 |
| 认知思考状态 | 19 | 0 | 19 |
| 语速 | 115 | 115 | 0 |
| 身体状态 | 25 | 0 | 25 |
| 适用场景场合 | 143 | 0 | 143 |
| 音域宽度 | 99 | 0 | 99 |
| 音色冷暖 | 36 | 0 | 36 |
| 音色厚薄 | 98 | 95 | 3 |
| 音色明暗 | 141 | 141 | 0 |
| 音色风格 | 91 | 82 | 9 |
| 音高 | 93 | 93 | 0 |
| 韵律变化 | 211 | 211 | 0 |

上述“口音与方言”是指令JSON中的额外字段，不属于这次指令白名单；方言分支的288项“口音/方言”判断则全部完成。

从逐维度结果看，MOPD128的语速、音高、情绪、韵律等点估计提高，音色厚薄和表演模式腔调没有提高；这些逐维度均值不等于每一维度都经过显著性检验。平均指令能力提升不能外推为所有指令能力提升。

[完整覆盖审计](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/judge-coverage-20260914.json)；[方言逐条评分示例](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/final-validation/0000-judge.json)；[指令逐条评分示例](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/final-validation/0288-judge.json)。


| 评估器/脚本 | 路径与设置 |
| --- | --- |
| AnyAudio-Judge30B权重 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/AnyAudio-Judge-30B](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/AnyAudio-Judge-30B) |
| AnyAudio源码 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/src/AnyAudio-Judge](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/src/AnyAudio-Judge) |
| 属性评分驱动 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/judge_audio.py](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/judge_audio.py)；实际音频输入，官方logits问题格式，yes/no候选各取最大logit再做二项softmax |
| 主CER | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/evaluate_cer.py](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/evaluate_cer.py)；Qwen3-ASR-1.7B服务，1次转写，NFKC+小写+zhconv简体+字母数字字符规范化 |
| 主音频生成/评分 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain_eval.py](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain_eval.py)；优先metadata.eval_seed，否则20270912+样本全局index；eval_concurrency4 |
| Whisper权重 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/whisper-large-v3](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/whisper-large-v3) |
| Whisper推理 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/whisper_worker.py](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/whisper_worker.py)；FP16/SDPA，batch8，greedy，num_beams1，max_new_tokens440，30s分块/5s重叠；粤语桶用cantonese，其余chinese |
| Whisper完整清单 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/whisper-inputs.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/whisper-inputs.json)；12000条，按audio SHA与固定解码配置缓存 |
| 同文本控制探针 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/counterfactual.jsonl](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/counterfactual.jsonl)；24文本×4条件=96条/模型，温度0.7，最多384帧；使用每条metadata.eval_seed |
| 声学度量 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/acoustic_features.py](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/acoustic_features.py)；mono16k，pyin50..800Hz，1024窗/320hop，voicing≥0.5、RMS门限、至少10有效F0帧 |
| 参考RVQ重建 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/decode_references.py](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/decode_references.py)；与生成相同vocoder，用于检查ASR误差水平 |

Whisper固定revision：`06f233fe06e710322aca913c1bc4249a0d71fce1`；model.safetensors SHA256：`a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`。主指标不因哪个识别器得分更好而替换。认证由评估器原有配置管理，本文不复制密钥或环境凭据。

### 辅助评估与接口检查记录

| 检查/运行 | 配置与结果 | 证据 |
| --- | --- | --- |
| 参考代码解码 | 576条参考音频；与既有波形相关系数0.999999989、MAE2.98e-5 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/reference-code-audio/decoder-check-0.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/reference-code-audio/decoder-check-0.json) |
| Whisper接口探针 | 中文/粤语短音频及35秒分块输入；完成，不代表识别准确率合格 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/whisper-probe.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/whisper-probe.json) |
| D结束后的探针状态 | 解码与Whisper探针returncode均0 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/automatic-probe-status.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/automatic-probe-status.json) |
| 声学测量自检 | 120/240/480Hz合成音F0检查通过 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/acoustic-self-test.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/acoustic-self-test.json) |
| automatic-evaluation-r1 | 8卡工作进程因缺zhconv启动失败；未产生转写结果 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/automatic-evaluation-r1-lease.json](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/automatic-evaluation-r1-lease.json) |
| automatic-evaluation-r2 | 修复依赖后8卡完成全部12000条；returncode0 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/automatic-evaluation-r2-lease.json](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/automatic-evaluation-r2-lease.json) |

统计：主CERcap逐条min(CER,1)后平均，另保留raw/micro-CER。属性分数不是人工准确率或MOS。2000次按来源组配对bootstrap，描述给定模型的提示不确定性，不是跨训练seed置信度。成对语速/音高使用固定24对与相同种子，成功还要求内容正确和自然停止。

## 8. 结果与判定

### P0工程吞吐与真实性

P0基准为4H200、DP2+2生成副本、GB64、最多128帧、温度1、LR0，两个流程各8次，剔除前2次。WER-GRPO平均步时12.5215秒，MOPD12.3454秒；1.41%差异只支持该次未见明显吞吐退化，不是统计确立的加速。占位教师结果不支持领域质量结论。详细[P0报告](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/REPORT.md)。

### P1最初的采样代理实验

P1使用温度1、每域96条最终评估。

**实验已按用户要求停止。LR 3e-6 MOPD 只保留第64步完整测试结果，未完成128步。总体结论见 [REPORT.md](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/REPORT.md)。**

学生固定从基础 TTS iter 162500 初始化。教师分别为方言/指令 iter 6000。测试集每领域 96 条，与本次 RL 训练数据隔离。

以下表格只纳入完整生成、ASR 和音频 Judge 均结束的测试点。Judge 为自动属性评分 p(yes)，不是人工准确率或 MOS。封顶 CER 使用 min(CER,1)；原始 CER 与中位数保存在 JSON。

| 模型 / checkpoint | 方言 Judge ↑ | 方言封顶 CER ↓ | 指令 Judge ↑ | 指令封顶 CER ↓ | 指令生成截断率 ↓ |
|---|---:|---:|---:|---:|---:|
| Base 162500 | 0.2147 | 0.2051 | 0.6597 | 0.7941 | 0.9479 |
| Dialect teacher | 0.4218 | 0.1779 | 0.6435 | 0.4650 | 0.3646 |
| Instruction teacher | 0.2344 | 0.2185 | 0.7962 | 0.2131 | 0.0000 |
| OPD dialect64 / 3e-6 | 0.2774 | 0.5638 | 0.5663 | 0.9042 | 0.9688 |
| OPD instruction64 / 3e-6 | 0.1771 | 0.3265 | 0.6243 | 0.4579 | 0.2083 |
| OPD instruction64 / 1e-6 | 0.1839 | 0.4402 | 0.5593 | 0.7131 | 0.8229 |
| OPD instruction64 / 3e-7 | 0.1893 | 0.2675 | 0.6520 | 0.7420 | 0.8958 |
| OPD dialect128 / 3e-6 | 0.2667 | 0.4251 | 0.5593 | 0.8744 | 0.9062 |
| OPD instruction128 / 3e-6 | 0.2129 | 0.3267 | 0.6612 | 0.3583 | 0.1146 |
| OPD dialect64 / 1e-6 | 0.2183 | 0.5117 | 0.5980 | 0.8666 | 0.9375 |
| MOPD64 / 1e-6 | 0.2242 | 0.4455 | 0.5697 | 0.7984 | 0.9271 |
| MOPD128 / 1e-6 | 0.2493 | 0.4829 | 0.5479 | 0.8207 | 0.9062 |
| MOPD64 / 3e-6 | 0.2562 | 0.5271 | 0.5636 | 0.7531 | 0.7812 |



### 前序实验：单域OPD到双域MOPD

前序阶段已完成同权重控制、方言单教师64更新、指令单教师64更新，以及从Base重新开始的双教师MOPD128。以下是前序每域192条最终测试的结果；这些样本后来已纳入本轮288条开发集，**不能将两张表的数值直接作为新的独立测试重复证据**。

| 模型 | 新训练更新数 | 方言Judge↑ | 方言CERcap↓ | 指令Judge↑ | 指令CERcap↓ | 指令截断率↓ |
|---|---:|---:|---:|---:|---:|---:|
| Base 162500 | 0 | 0.2151 | 0.0720 | 0.5187 | 0.3690 | 94.79% |
| 方言教师6000 | — | 0.4708 | 0.0423 | — | — | — |
| 指令教师6000 | — | — | — | 0.7685 | 0.0859 | 9.90% |
| 方言单教师OPD | 64 | 0.2448 | 0.0883 | — | — | — |
| 指令单教师OPD | 64 | — | — | 0.5949 | 0.1631 | 26.04% |
| 两教师MOPD | 128 | **0.2835** | **0.1241** | **0.5855** | **0.1267** | **18.75%** |

前序结论：指令单教师迁移证据较强；方言单教师结果较弱且存在验证/最终划分差异。双教师在同一学生上同时提高了领域属性，但方言内容有代价。完整配对区间、温度对照、实现验证见[单OPD/MOPD阶段报告](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/../20260912-notebook-v2/REPORT.md)。

其后A/B/C分别测试纯续训、筛选教师轨迹混合、前8帧加权，均未通过当时质量门槛，见[第一轮修复报告](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/../20260913-quality/REPORT.md)。本报告主体覆盖进一步D–L机制对照、VC父模型对照、独立ASR和声学探针。


### 全部主评估结果

每个模型每域288条。Judge 为 AnyAudio-Judge30B 的自动属性 p(yes) 平均值，越高越好；不是人工准确率或MOS。CERcap 为逐条 `min(CER,1)` 后取均值，越低越好。所有模型使用同温度、同提示、同评估种子。原始CER、micro-CER及逐条结果同时保存。


| 模型 | 方言Judge↑ | 方言CERcap↓ | 指令Judge↑ | 指令CERcap↓ | 指令截断率↓ |
| --- | --- | --- | --- | --- | --- |
| Base 162500 | 0.2104 | 0.0730 | 0.5220 | 0.3668 | 95.1% |
| 对应领域教师 | 0.4567 | 0.0463 | 0.7633 | 0.0883 | 9.4% |
| 纯 MOPD128 | 0.2712 | 0.1162 | 0.5852 | 0.1193 | 17.7% |
| A 原反向 KL 续训 | 0.2873 | 0.1089 | 0.5881 | 0.1229 | 17.0% |
| B 筛选提示及教师轨迹 | 0.3053 | 0.1045 | 0.5837 | 0.1471 | 20.1% |
| C 前8帧加权 | 0.3019 | 0.1153 | 0.6062 | 0.1181 | 18.4% |
| D 全前向 KL | 0.2995 | 0.1431 | 0.6240 | 0.0835 | 6.2% |
| E RVQ1加权 | 0.2899 | 0.1026 | 0.5913 | 0.0964 | 15.3% |
| F RVQ1–3加权 | 0.2807 | 0.0940 | 0.6010 | 0.1191 | 17.0% |
| G 停止/继续加权 | 0.2913 | 0.0985 | 0.6013 | 0.1266 | 18.4% |
| H 同提示教师轨迹混合 | 0.2968 | 0.0890 | 0.5869 | 0.1245 | 19.1% |
| I 同提示混合+前向 KL | 0.2812 | 0.1600 | 0.6259 | 0.0843 | 5.2% |
| J 仅更换筛选提示 | 0.2822 | 0.1067 | 0.5850 | 0.1156 | 15.6% |
| K 分领域 KL 方向 | 0.2893 | 0.1152 | 0.6176 | 0.0934 | 10.4% |
| L 分领域组合 | 0.2847 | 0.1094 | 0.6094 | 0.0776 | 6.6% |
| 仅VC父模型指导128步 | 0.1929 | 0.1143 | 0.4488 | 0.2881 | 99.7% |
| 直接VC父模型 | 0.2194 | 0.1018 | 0.4790 | 0.3841 | 94.4% |


![开发集质量与属性权衡](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/development-tradeoffs.png)

图中 H 为同提示轨迹混合，L 为分领域组合，灰点为其他对照。对应教师行是两个不同专家各自领域的结果；所有学生行均为同一个学生同时承担两域。


### 哪些机制得到支持

**同提示教师轨迹改善方言内容。** H 对 MOPD128 的主评估方言 CERcap 从0.1162降至0.0890，Judge从0.2712升至0.2968。与单纯再训练64次的A对照相比，也有内容改善。下面的差值均为“正数表示改善”，区间来自2000次来源组配对bootstrap。


| 方言CER改善 | Qwen主评估：差值 [95%CI] | Whisper：差值 [95%CI] |
| --- | --- | --- |
| H vs MOPD128 | +0.0272 [+0.0093, +0.0450] | +0.0209 [+0.0032, +0.0408] |
| H vs A同预算续训 | +0.0198 [+0.0041, +0.0363] | +0.0181 [+0.0002, +0.0368] |


H 的指令Judge保持在0.5869；CERcap为0.1245，相对MOPD128的0.1193没有改善，截断率仍有19.10%。H 并非两个领域所有指标都最优。其方言/指令原始平均CER分别为0.0890/0.1245；完整原始值和micro-CER见[CSV](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/development-results.csv)。Base和MOPD128有少量ASR长转写离群值，原始CER可能显著高于CERcap，报告没有丢弃这些记录。

**KL方向存在明显领域取舍。** D的全前向KL使指令CERcap降至0.0835，但方言升至0.1431。H→I在相同提示、相同教师轨迹缓存上改用前向KL，方言进一步恶化而指令改善，支持该取舍并非仅来自采样提示差异。

**粗RVQ加权有方言改善信号，但不等价于完整修复。** F的前3码本权重为4，其余为1，并保持每样本RVQ总权重归一化；方言CERcap降至0.0940，仍未达到0.09。E只强化RVQ1，G强化停止/继续动作，均未通过全部标准。

**各域最优设置不能简单组合成联合最优。** K使用方言反向KL、指令前向KL；L再加入方言前3码本加权。L达到本轮最低指令主CERcap0.0776，指令Judge0.6094，截断率6.60%；但方言CERcap0.1094，未保留F的最佳方言点估计。K/L是看到前面对照结果后追加的有界探索，已在运行前记录协议补充；不是原始预注册的独立验证。

**筛选提示本身不足以解释轨迹混合收益。** J 与 A 的方言CER改善仅0.0022，区间跨0；B对J的方言Judge改善0.0231，区间[0.0052,0.0408]，但方言CER差异区间跨0。H保持原始提示后仍优于A，提供了更干净的教师轨迹贡献证据。[J/B提示一致性审计](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/prompt-contrast-audit.json)及[机制配对结果](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/comprehensive-analysis.json)保留细节。

**专家教师确实提供了父模型没有的领域能力。** 仅用共同VC200000父模型指导Base128次，得到方言/指令Judge0.1929/0.4488，显著低于专家MOPD128的0.2712/0.5852；指令CERcap0.2881也明显更差。这支持领域专家的价值。该对照只覆盖当前无参考音频输入的任务，不能推断VC父模型在VoiceClone任务上的表现。

![方言与指令内容变化的配对区间](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/development-cer-effects.png)


### 独立ASR结果及其可靠性

独立评估对12000条音频全部完成转写，包含所有主要对照、17组成对指令探针及576条参考codes重建音频。使用固定 [Whisper large-v3 官方模型](https://huggingface.co/openai/whisper-large-v3)，revision `06f233fe06e710322aca913c1bc4249a0d71fce1`，本地八卡推理，固定greedy、30秒分块/5秒重叠、中文/粤语语言提示；未向识别器提供待识别文本。主指标仍为Qwen，没有因结果而替换识别器。

参考音频由评估记录的真实RVQ代码通过同一vocoder重建。独立解码实现与已有生成波形核验：相关系数0.999999989，平均绝对差2.98e-5。它是识别误差水平的对照，仍包含codec重建损失，不是人工听感真值。


| 模型/音频 | 方言Qwen | 方言Whisper | 指令Qwen | 指令Whisper |
| --- | --- | --- | --- | --- |
| Base 162500 | 0.0730 | 0.1553 | 0.3668 | 0.4046 |
| 对应领域教师 | 0.0463 | 0.2708 | 0.0883 | 0.1339 |
| 纯 MOPD128 | 0.1162 | 0.2378 | 0.1193 | 0.1570 |
| A 原反向 KL 续训 | 0.1089 | 0.2350 | 0.1229 | 0.1641 |
| H 同提示教师轨迹混合 | 0.0890 | 0.2169 | 0.1245 | 0.1789 |
| F RVQ1–3加权 | 0.0940 | 0.2420 | 0.1191 | 0.1597 |
| L 分领域组合 | 0.1094 | 0.2417 | 0.0776 | 0.1288 |
| 直接VC父模型 | 0.1018 | 0.1893 | 0.3841 | 0.4054 |
| 真实参考codes重建 | 0.0553 | 0.3164 | 0.0861 | 0.1536 |


Whisper的全方言结果不能直接被当作内容质量真值：真实参考音频的CERcap0.3164，专家0.2708，反而高于Base0.1553。逐类定位尤其明显：上海话参考0.7228、教师0.6913、H0.5820、Base0.2208。较标准普通话的输出可能比真实方言更容易被它识别。这个解释有参考音频对照支持，但不能据此断言学生不存在真实发音或文本错误。 **主评估也保留了上海话短板：Qwen下Base为0.1539、教师0.1673、MOPD128为0.4118、H为0.3251。H修复了一部分问题，但上海话仍明显落后于Base和教师；全方言平均0.0890掩盖了这一类别差异。**

![六类方言的识别器差异](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/dialect-asr-reliability.png)

看到分歧后，追加了明确标为探索性的CPU分析：仅按照参考音频是否CER≤0.2选择诊断子集，不依据某个学生得分筛选。两套ASR都通过的方言子集为125/288条，构成为四川14、川普27、东北32、东北普通话42、粤语10，**上海话0条**。


| 参考可靠子集模型 | Qwen CERcap | Whisper CERcap |
| --- | --- | --- |
| Base 162500 | 0.0369 | 0.0772 |
| 纯 MOPD128 | 0.0186 | 0.0727 |
| H 同提示教师轨迹混合 | 0.0149 | 0.0630 |
| 对应领域教师 | 0.0095 | 0.0773 |


H在该子集两套ASR的点估计均优于Base；Qwen改善区间为[0.0090,0.0383]，Whisper为[-0.0152,0.0468]，后者仍跨0。该子集不能外推至上海话或所有方言。另做“以同识别器对参考音频的转写作为比较文本”的敏感性分析，完整记录于[ASR可靠性分析](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/asr-reliability-analysis.json)；识别器转写不是ground truth，未替换主结果或验收条件。


### 同文本成对指令测试

固定24个文本（normal/denoise各12），分别施加单字段“语速:低/高”“音高:低/高”，配对使用同一个生成种子。该探针温度0.7、最多384帧，与主测试192帧上限不同，不能直接合并均值。

合格条件同时要求：两端自然停止、两端内容CER≤0.2、声学测量有效，且低速/高速有效语音时长比≥1.05，或高音/低音的中位F0比≥1.10。F0由pyin计算，RMS/voicing过滤及最少有效帧要求固定；120/240/480Hz合成音检查通过。分母始终为24，未把无效测量从成功率分母移除。此测试只覆盖语速/音高的受控变化，不是整体指令遵循准确率。


| 模型 | Qwen合格语速对 | Qwen合格音高对 | Whisper合格语速对 | Whisper合格音高对 |
| --- | --- | --- | --- | --- |
| Base | 0/24 | 2/24 | 0/24 | 2/24 |
| 指令教师 | 13/24 | 8/24 | 11/24 | 7/24 |
| MOPD128 | 4/24 | 8/24 | 3/24 | 6/24 |
| A续训 | 9/24 | 5/24 | 8/24 | 5/24 |
| G停止/继续加权 | 11/24 | 6/24 | 9/24 | 2/24 |
| H同提示轨迹混合 | 7/24 | 4/24 | 7/24 | 4/24 |
| L分领域组合 | 7/24 | 5/24 | 7/24 | 5/24 |


语速控制有改善，但音高控制没有稳定随平均指令Judge提升。H相对MOPD128的音高合格对反而减少；24对的样本量也较小。全部17组的原始比值、失效/截断计数和配对区间见[主成对分析](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/counterfactual-analysis.json)及[独立ASR成对分析](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/whisper-counterfactual-analysis.json)。


### 验收与未完成的科学确认

主验收保持不变：方言CERcap≤0.09且较MOPD128改善；两域相对Base的属性增益至少保留80%；两域Judge增益区间下界>0；CER对Base的退化区间上界≤0.05；指令对MOPD128的CER退化区间上界≤0.05。**H是本轮唯一通过全部主门槛的配置。**

独立ASR筛选在全量Whisper结果之前定义：任一领域对Base的Whisper CER点退化>0.05，且95%配对区间完全指向退化，则不进入新种子质量确认。H方言退化0.0616，区间[0.0341,0.0883]，因此被筛除。其余候选也没有同时通过主/相对门槛及独立ASR筛选。[冻结判定](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/confirmation-decision.json)记录了全部候选及输入分析哈希。

**本轮没有质量确认候选，没有训练额外两个种子，也没有打开新最终测试。** ASR诊断没有被用于事后放宽门槛或宣布验收成功。未来若修订方言评测方案，应先独立制定和冻结校准方案，再进行新种子及保留集确认。当前证据支持“领域迁移存在、同提示轨迹混合能修复一部分方言内容问题”，不支持“已证明全方言稳定有效”或“已达到教师水平”。


## 9. 复现入口与证据定位

复现按保存的配置、命令和源码快照进行。每个训练目录通常包含`args.json`（上层参数）、`commands.json`（服务/训练实际argv）、`eval-config.json`（评估路径与地址）、`train.log`、`experiment-scripts/`、`policy-source/`、`policy-source.patch`、`shared-source/`和`source-versions.json`。不同阶段快照范围不同，以实际存在项为准。不能只凭当前git HEAD重建当时的未提交补丁。

### 服务配置与Codec路径

| 角色 | H运行实际命令所引用的配置 |
| --- | --- |
| 冻结教师评分 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/sglang-omni/examples/configs/moss_tts_local_score.yaml](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/sglang-omni/examples/configs/moss_tts_local_score.yaml) |
| 可更新学生评分 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/sglang-omni/examples/configs/moss_tts_local_student_score.yaml](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/sglang-omni/examples/configs/moss_tts_local_student_score.yaml) |
| 学生生成 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/sglang-omni/examples/configs/moss_tts_local.yaml](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/sglang-omni/examples/configs/moss_tts_local.yaml) |
| 生成预处理/声码器Codec | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-Audio-Tokenizer-v2](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-Audio-Tokenizer-v2) |

H的生成命令显式设定`vocoder.factory.cuda_graph=false`、`tts_engine.engine.disable_radix_cache=true`、`tts_engine.engine.disable_cuda_graph=false`；模型路径分别指向P4/models中的base/dialect/instruction。具体端口与服务argv保存在各运行commands.json，YAML的历史版本须结合source-versions核对。

| 入口 | 文件 |
| --- | --- |
| 统一训练launcher | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/miles/scripts/run_moss_tts_local.py](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/miles/scripts/run_moss_tts_local.py) |
| P2正式单域/联合序列 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/joint-suite-r1.json](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/joint-suite-r1.json) |
| P2原生损失小试 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-suite.json](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-suite.json) |
| P3完整修复序列 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-complete-suite.json](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-complete-suite.json) |
| P4主序列 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/comprehensive-suite.json](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/comprehensive-suite.json) |
| P4分领域后续 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-followup-suite.json](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-followup-suite.json) |
| 单次实验驱动 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/run_experiment.py](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/run_experiment.py) |
| 顺序执行与清理 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/run_suite.py](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/run_suite.py) |
| 整节点lease | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/lease8.py](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/lease8.py) |
| 主指标汇总 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/analyze_comprehensive.py](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/analyze_comprehensive.py) |
| 独立ASR汇总 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/analyze_whisper.py](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/analyze_whisper.py) |
| 属性覆盖补充 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/supplement_report_20260914.py](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/supplement_report_20260914.py) |
| 本手册生成脚本 | [/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/documentation/build_handbook.py](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/documentation/build_handbook.py) |

复现实验应使用新的RUN名称和输出位置，并先核对模型、数据和节点。历史`start_*.sh`保留固定旧RUN名称，直接重复执行不会创建独立复验。本文仅保存可复现入口，未重新执行这些GPU任务。

### 已验证H配置的完整原始参数

以下是已经执行过的配置记录，`steps=192,start_step=128`表示新增64次。它由suite同时管理服务、训练、Judge和清理；单独裸跑train.py不能替代整个服务链路。

```json
{
  "name": "H-matched-reverse64-r1",
  "mode": "train",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 192,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "final_validation_per_domain": 288,
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-replay/manifest.json",
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

### 从产物定位问题

| 需要检查的内容 | 证据位置 |
| --- | --- |
| 教师身份/冻结 | teacher-before.json、teacher-after.json、checkpoints/mopd_teachers.json及原生教师记录 |
| 学生生成端同步 | student-weight-audit.json：330学习参数张量；resume-student-audit.json核验续训起点 |
| 域路由/真实行为来源 | routing-audit.json或quality-run-audit.json，debug/rollout_<id>.pt |
| 优化器与实际更新数 | quality-run-audit.json、阶段run-audit.json、train.log及原生DCP标量 |
| 评估逐条音频/问题 | 评估目录NNNN.json、NNNN-<domain>.wav、NNNN-scores.json、NNNN-cer.json、NNNN-judge.json |
| 全量指标/区间 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/comprehensive-analysis.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/comprehensive-analysis.json)、[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/whisper-analysis.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/whisper-analysis.json) |
| 保留集未使用/最终状态 | [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/confirmation-decision.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/confirmation-decision.json)、[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/COMPLETION.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/COMPLETION.json) |

## 10. 状态、失败和未执行项目

P1的mopd-mixed128-lr3e6-r2按用户指令停止，最终日志78次更新、最后完整保存并评估64次，没有128次结果。P1的smoke8-r1遇到错误的文本评估路径，修复后r2完成。P3的replay-smoke-r1触发未发布default版本校验，修复显式教师行为来源后r2完成。教师池未完成尝试保留在索引中，不被当作完整池。

P4独立ASR r1因缺少zhconv在转写前退出；隔离依赖与主CER保持一致并核验后，r2完成12000条。原失败日志见[/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/asr-r1-failure](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/asr-r1-failure)；依赖来源见[/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/asr-deps/provenance.json](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/asr-deps/provenance.json)。

已有prepare_confirmation_suite.py、prepare_confirmation_whisper.py等后续确认工具，但选择结果为null，没有创建两个新训练seed的正式确认运行，也没有打开新192条/域保留测试。存在工具文件不代表对应实验已执行。没有人工听评；已有音频试听页面的存在也不代表有人工评分。

P4正式训练704次、实现烟雾8次、独立ASR12000条；实际Notebook167项MOSS+44项相关通用测试通过。训练结果仍不能外推到ASMR、长文本、其他语言或有参考输入的VoiceClone。

历史收尾核验：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/final-occupancy-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/final-occupancy-audit.json)、[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/idle-process-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/idle-process-audit.json)。记录当时8卡恢复gpu-occupy、无本研究残留进程；本文未查询或改变Notebook当前状态。

### 路径核对与原始资产保护

本文引用原始报告和机器结果，没有覆盖原REPORT.md、COMPLETION.json或历史模型。指令教师原HF当前不可访问已显式标注；其原生与实验导出仍可访问。其余路径可用性以本文随附路径清单的整理时间为准。

机器索引：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/experiment-inventory-20260914.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/experiment-inventory-20260914.json)。每个配置记录保存SHA256；共享产物复制件为`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/EXPERIMENTS-AND-CONFIGS-20260914.md`；核对清单为`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/experiment-handbook-manifest-20260914.json`。

## 11. 每个实验的完整配置与产物

### P0 / benchmark-mopd-r1

固定权重MOPD吞吐对照

状态：完成；模式 `train`；起始计数 `0`；计划终点 `8`；本次实际新增更新 **8**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-mopd-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-mopd-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-mopd-r1/info.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-mopd-r1/info.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-mopd-r1/train.log](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-mopd-r1/train.log)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-mopd-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-mopd-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

本目录没有保存训练检查点。

审计/源码：[audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-mopd-r1/audit.json)

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "0.0",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "8",
  "start-rollout-id": "0",
  "rollout-max-response-len": "128",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-mopd-r1/data/train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.1/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-TTS-Local-mossLite-v0.1.1-iter20000-sglang"
}
```

</details>

### P0 / benchmark-wer-r1

固定权重WER-GRPO吞吐基线

状态：完成；模式 `train`；起始计数 `0`；计划终点 `8`；本次实际新增更新 **8**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-wer-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-wer-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-wer-r1/info.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-wer-r1/info.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-wer-r1/train.log](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-wer-r1/train.log)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-wer-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-wer-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

本目录没有保存训练检查点。

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "0.0",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "8",
  "start-rollout-id": "0",
  "rollout-max-response-len": "128",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/benchmark-wer-r1/data/train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.1/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-TTS-Local-mossLite-v0.1.1-iter20000-sglang"
}
```

</details>

### P0 / paired-resume-r1

零信号检查点恢复

状态：完成；模式 `train`；起始计数 `2`；计划终点 `4`；本次实际新增更新 **2**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-resume-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-resume-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-resume-r1/info.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-resume-r1/info.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-resume-r1/train.log](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-resume-r1/train.log)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-resume-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-resume-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-resume-r1/checkpoints/iter_0000003](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-resume-r1/checkpoints/iter_0000003)

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "4",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "4",
  "start-rollout-id": "2",
  "save-interval": "4",
  "rollout-max-response-len": "128",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-local-tts-rl/20260903/moss-local-split-gb128-c12-long128-r6-20260903-0341/data/train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-smoke-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.1/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-TTS-Local-mossLite-v0.1.1-iter20000-sglang"
}
```

</details>

### P0 / paired-smoke-r1

教师/学生相同权重的paired-prefill零信号对照；训练完成，收尾解析错误KeyError: stages，后续脚本修复

状态：训练完成，收尾解析失败；模式 `train`；起始计数 `0`；计划终点 `2`；本次实际新增更新 **2**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-smoke-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-smoke-r1/info.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-smoke-r1/info.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-smoke-r1/train.log](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-smoke-r1/train.log)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-smoke-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-smoke-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-smoke-r1/checkpoints/iter_0000001](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/paired-smoke-r1/checkpoints/iter_0000001)

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "4",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "2",
  "start-rollout-id": "0",
  "save-interval": "2",
  "rollout-max-response-len": "128",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-local-tts-rl/20260903/moss-local-split-gb128-c12-long128-r6-20260903-0341/data/train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.1/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-TTS-Local-mossLite-v0.1.1-iter20000-sglang"
}
```

</details>

### P0 / positive-r1

有权重差异的功能正信号测试；教师为占位Base，非领域质量实验

状态：完成；模式 `train`；起始计数 `0`；计划终点 `4`；本次实际新增更新 **4**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-r1/info.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-r1/info.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-r1/train.log](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-r1/train.log)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-r1/checkpoints/iter_0000003](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-r1/checkpoints/iter_0000003)

审计/源码：[audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-r1/audit.json)

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "8",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "4",
  "start-rollout-id": "0",
  "save-interval": "4",
  "rollout-max-response-len": "128",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-r1/data/train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-rl/20260908-wer32-r2/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.1/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-TTS-Local-mossLite-v0.1.1-iter20000-sglang"
}
```

</details>

### P0 / positive-resume-r1

非零Adam与调度器状态恢复

状态：完成；模式 `train`；起始计数 `4`；计划终点 `6`；本次实际新增更新 **2**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-resume-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-resume-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-resume-r1/info.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-resume-r1/info.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-resume-r1/train.log](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-resume-r1/train.log)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-resume-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-resume-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-resume-r1/checkpoints/iter_0000005](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-resume-r1/checkpoints/iter_0000005)

审计/源码：[audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-resume-r1/audit.json)

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "8",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "6",
  "start-rollout-id": "4",
  "save-interval": "6",
  "rollout-max-response-len": "128",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-resume-r1/data/train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/positive-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.1/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-TTS-Local-mossLite-v0.1.1-iter20000-sglang"
}
```

</details>

### P0 / refactor-mopd-r2

拆分/重构后真实更新回归

状态：完成；模式 `train`；起始计数 `0`；计划终点 `4`；本次实际新增更新 **4**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/refactor-mopd-r2](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/refactor-mopd-r2)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/refactor-mopd-r2/info.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/refactor-mopd-r2/info.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/refactor-mopd-r2/train.log](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/refactor-mopd-r2/train.log)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/refactor-mopd-r2/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/refactor-mopd-r2/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/refactor-mopd-r2/checkpoints/iter_0000003](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/refactor-mopd-r2/checkpoints/iter_0000003)

审计/源码：[audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/refactor-mopd-r2/audit.json)

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "8",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "4",
  "start-rollout-id": "0",
  "save-interval": "4",
  "rollout-max-response-len": "128",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/refactor-mopd-r2/data/train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-rl/20260909-async-refactor-wer32-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.1/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-TTS-Local-mossLite-v0.1.1-iter20000-sglang"
}
```

</details>

### P0 / smoke-r1

早期直接比较prefill/decode分数的探索；同权重伪信号，未作为正确蒸馏验收

状态：完成；模式 `train`；起始计数 `0`；计划终点 `2`；本次实际新增更新 **2**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/smoke-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/smoke-r1/info.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/smoke-r1/info.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/smoke-r1/train.log](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/smoke-r1/train.log)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/smoke-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/smoke-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/smoke-r1/checkpoints/iter_0000001](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/miles-moss-mopd/20260909/smoke-r1/checkpoints/iter_0000001)

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "4",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "1",
  "num-gpus-per-node": "1",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "2",
  "start-rollout-id": "0",
  "save-interval": "2",
  "rollout-max-response-len": "128",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-local-tts-rl/20260903/moss-local-split-gb128-c12-long128-r6-20260903-0341/data/train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.1.1/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-TTS-Local-mossLite-v0.1.1-iter20000-sglang"
}
```

</details>

### P1 / baselines8-r1

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `baseline`；起始计数 `0`；计划终点 `64`；本次实际新增更新 **0**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/baselines8-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/baselines8-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/baselines8-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/baselines8-r1/args.json)

更新数来源：模式无优化器更新

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/baselines8-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/baselines8-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

本目录没有保存训练检查点。

已有评估汇总：

- [base-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/baselines8-r1/base-eval/summary.json)
- [base-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/baselines8-r1/base-validation/summary.json)
- [dialect-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/baselines8-r1/dialect-eval/summary.json)
- [dialect-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/baselines8-r1/dialect-validation/summary.json)
- [instruction-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/baselines8-r1/instruction-eval/summary.json)
- [instruction-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/baselines8-r1/instruction-validation/summary.json)

审计/源码：[teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/baselines8-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/baselines8-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/baselines8-r1/source-versions.json)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "baselines8-r1",
  "mode": "baseline",
  "domain": "dialect",
  "steps": 64,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 16,
  "eval_per_domain": 96,
  "max_frames": 192,
  "debug": false
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{}
```

</details>

### P1 / mopd-mixed128-lr1e6

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `0`；计划终点 `128`；本次实际新增更新 **128**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/checkpoints/iter_0000031](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/checkpoints/iter_0000031)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/checkpoints/iter_0000063](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/checkpoints/iter_0000063)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/checkpoints/iter_0000095](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/checkpoints/iter_0000095)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/checkpoints/iter_0000127](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/checkpoints/iter_0000127)

已有评估汇总：

- [final-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/final-eval/summary.json)
- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/final-validation/summary.json)
- [step64-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/step64-eval/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/validation/point-rollout-00000/summary.json)
- [validation/point-rollout-00032/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/validation/point-rollout-00032/summary.json)
- [validation/point-rollout-00064/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/validation/point-rollout-00064/summary.json)
- [validation/point-rollout-00096/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/validation/point-rollout-00096/summary.json)

审计/源码：[teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/source-versions.json) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr1e6/experiment-scripts)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "mopd-mixed128-lr1e6",
  "mode": "train",
  "domain": "mixed",
  "steps": 128,
  "batch": 16,
  "samples": 4,
  "lr": 1e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": false,
  "resume_from": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "1e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "4",
  "num-gpus-per-node": "4",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "128",
  "start-rollout-id": "0",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/data/mixed-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/base-162500"
}
```

</details>

### P1 / mopd-mixed128-lr3e6-r2

按用户要求中止；记录78次更新，仅64次检查点完成测试

状态：未完成/中止；模式 `train`；起始计数 `0`；计划终点 `128`；本次实际新增更新 **78**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/train.log](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/train.log)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/checkpoints/iter_0000031](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/checkpoints/iter_0000031)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/checkpoints/iter_0000063](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/checkpoints/iter_0000063)

已有评估汇总：

- [step64-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/step64-eval/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/validation/point-rollout-00000/summary.json)
- [validation/point-rollout-00032/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/validation/point-rollout-00032/summary.json)
- [validation/point-rollout-00064/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/validation/point-rollout-00064/summary.json)

审计/源码：[teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/teacher-before.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/source-versions.json) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-lr3e6-r2/experiment-scripts)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "mopd-mixed128-lr3e6-r2",
  "mode": "train",
  "domain": "mixed",
  "steps": 128,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": false,
  "resume_from": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "4",
  "num-gpus-per-node": "4",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "128",
  "start-rollout-id": "0",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/data/mixed-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/base-162500"
}
```

</details>

### P1 / mopd-mixed128-r1

首个LR3e-6联合任务，训练前取消，无完整训练结果

状态：训练前取消；模式 `train`；起始计数 `0`；计划终点 `128`；本次实际新增更新 **0**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/EVIDENCE.md](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/EVIDENCE.md)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

本目录没有保存训练检查点。

审计/源码：[source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-r1/source-versions.json) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/mopd-mixed128-r1/experiment-scripts)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "mopd-mixed128-r1",
  "mode": "train",
  "domain": "mixed",
  "steps": 128,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": false,
  "resume_from": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{}
```

</details>

### P1 / opd-dialect128-r1

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `64`；计划终点 `128`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/checkpoints/iter_0000095](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/checkpoints/iter_0000095)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/checkpoints/iter_0000127](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/checkpoints/iter_0000127)

已有评估汇总：

- [final-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/final-eval/summary.json)
- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/final-validation/summary.json)
- [step64-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/step64-eval/summary.json)
- [validation/point-rollout-00064/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/validation/point-rollout-00064/summary.json)
- [validation/point-rollout-00096/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/validation/point-rollout-00096/summary.json)

审计/源码：[teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/source-versions.json) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect128-r1/experiment-scripts)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "opd-dialect128-r1",
  "mode": "train",
  "domain": "dialect",
  "steps": 128,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": false,
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/checkpoints",
  "start_step": 64
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "4",
  "num-gpus-per-node": "4",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "128",
  "start-rollout-id": "64",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/data/dialect-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/base-162500"
}
```

</details>

### P1 / opd-dialect64-lr1e6

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `0`；计划终点 `64`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/checkpoints/iter_0000015](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/checkpoints/iter_0000015)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/checkpoints/iter_0000031](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/checkpoints/iter_0000031)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/checkpoints/iter_0000047](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/checkpoints/iter_0000047)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/checkpoints/iter_0000063](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/checkpoints/iter_0000063)

已有评估汇总：

- [final-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/final-eval/summary.json)
- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/final-validation/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/validation/point-rollout-00000/summary.json)
- [validation/point-rollout-00016/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/validation/point-rollout-00016/summary.json)
- [validation/point-rollout-00032/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/validation/point-rollout-00032/summary.json)
- [validation/point-rollout-00048/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/validation/point-rollout-00048/summary.json)

审计/源码：[teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/source-versions.json) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-lr1e6/experiment-scripts)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "opd-dialect64-lr1e6",
  "mode": "train",
  "domain": "dialect",
  "steps": 64,
  "batch": 16,
  "samples": 4,
  "lr": 1e-06,
  "eval_interval": 16,
  "eval_per_domain": 96,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": false,
  "resume_from": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "1e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "4",
  "num-gpus-per-node": "4",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "64",
  "start-rollout-id": "0",
  "save-interval": "16",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/data/dialect-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/base-162500"
}
```

</details>

### P1 / opd-dialect64-r1

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `0`；计划终点 `64`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/checkpoints/iter_0000015](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/checkpoints/iter_0000015)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/checkpoints/iter_0000031](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/checkpoints/iter_0000031)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/checkpoints/iter_0000047](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/checkpoints/iter_0000047)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/checkpoints/iter_0000063](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/checkpoints/iter_0000063)

已有评估汇总：

- [final-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/final-eval/summary.json)
- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/final-validation/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/validation/point-rollout-00000/summary.json)
- [validation/point-rollout-00016/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/validation/point-rollout-00016/summary.json)
- [validation/point-rollout-00032/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/validation/point-rollout-00032/summary.json)
- [validation/point-rollout-00048/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/validation/point-rollout-00048/summary.json)

审计/源码：[teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/source-versions.json) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-dialect64-r1/experiment-scripts)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "opd-dialect64-r1",
  "mode": "train",
  "domain": "dialect",
  "steps": 64,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 16,
  "eval_per_domain": 96,
  "max_frames": 192,
  "debug": false,
  "resume_from": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "4",
  "num-gpus-per-node": "4",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "64",
  "start-rollout-id": "0",
  "save-interval": "16",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/data/dialect-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/base-162500"
}
```

</details>

### P1 / opd-instruction128-r1

未完成的续训尝试，后由r2替代

状态：未完成/中止；模式 `train`；起始计数 `64`；计划终点 `128`；本次实际新增更新 **未从记录提取**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r1/args.json)

更新数来源：`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r1/train.log`（整理时不可访问）

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/checkpoints)。

本目录没有保存训练检查点。

审计/源码：[source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r1/source-versions.json) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r1/experiment-scripts)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "opd-instruction128-r1",
  "mode": "train",
  "domain": "instruction",
  "steps": 128,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": false,
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/checkpoints",
  "start_step": 64
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{}
```

</details>

### P1 / opd-instruction128-r2

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `64`；计划终点 `128`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/checkpoints/iter_0000095](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/checkpoints/iter_0000095)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/checkpoints/iter_0000127](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/checkpoints/iter_0000127)

已有评估汇总：

- [final-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/final-eval/summary.json)
- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/final-validation/summary.json)
- [step64-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/step64-eval/summary.json)
- [validation/point-rollout-00064/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/validation/point-rollout-00064/summary.json)
- [validation/point-rollout-00096/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/validation/point-rollout-00096/summary.json)

审计/源码：[student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/source-versions.json) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction128-r2/experiment-scripts)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "opd-instruction128-r2",
  "mode": "train",
  "domain": "instruction",
  "steps": 128,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": false,
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/checkpoints",
  "start_step": 64
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "4",
  "num-gpus-per-node": "4",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "128",
  "start-rollout-id": "64",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/data/instruction-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/base-162500"
}
```

</details>

### P1 / opd-instruction64-lr1e6

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `0`；计划终点 `64`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/checkpoints/iter_0000015](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/checkpoints/iter_0000015)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/checkpoints/iter_0000031](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/checkpoints/iter_0000031)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/checkpoints/iter_0000047](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/checkpoints/iter_0000047)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/checkpoints/iter_0000063](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/checkpoints/iter_0000063)

已有评估汇总：

- [final-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/final-eval/summary.json)
- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/final-validation/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/validation/point-rollout-00000/summary.json)
- [validation/point-rollout-00016/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/validation/point-rollout-00016/summary.json)
- [validation/point-rollout-00032/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/validation/point-rollout-00032/summary.json)
- [validation/point-rollout-00048/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/validation/point-rollout-00048/summary.json)

审计/源码：[teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/source-versions.json) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr1e6/experiment-scripts)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "opd-instruction64-lr1e6",
  "mode": "train",
  "domain": "instruction",
  "steps": 64,
  "batch": 16,
  "samples": 4,
  "lr": 1e-06,
  "eval_interval": 16,
  "eval_per_domain": 96,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": false,
  "resume_from": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "1e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "4",
  "num-gpus-per-node": "4",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "64",
  "start-rollout-id": "0",
  "save-interval": "16",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/data/instruction-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/base-162500"
}
```

</details>

### P1 / opd-instruction64-lr3e7

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `0`；计划终点 `64`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/checkpoints/iter_0000015](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/checkpoints/iter_0000015)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/checkpoints/iter_0000031](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/checkpoints/iter_0000031)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/checkpoints/iter_0000047](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/checkpoints/iter_0000047)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/checkpoints/iter_0000063](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/checkpoints/iter_0000063)

已有评估汇总：

- [final-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/final-eval/summary.json)
- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/final-validation/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/validation/point-rollout-00000/summary.json)
- [validation/point-rollout-00016/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/validation/point-rollout-00016/summary.json)
- [validation/point-rollout-00032/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/validation/point-rollout-00032/summary.json)
- [validation/point-rollout-00048/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/validation/point-rollout-00048/summary.json)

审计/源码：[teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/source-versions.json) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-lr3e7/experiment-scripts)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "opd-instruction64-lr3e7",
  "mode": "train",
  "domain": "instruction",
  "steps": 64,
  "batch": 16,
  "samples": 4,
  "lr": 3e-07,
  "eval_interval": 16,
  "eval_per_domain": 96,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": false,
  "resume_from": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-07",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "4",
  "num-gpus-per-node": "4",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "64",
  "start-rollout-id": "0",
  "save-interval": "16",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/data/instruction-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/base-162500"
}
```

</details>

### P1 / opd-instruction64-r1

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `0`；计划终点 `64`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/checkpoints/iter_0000015](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/checkpoints/iter_0000015)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/checkpoints/iter_0000031](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/checkpoints/iter_0000031)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/checkpoints/iter_0000047](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/checkpoints/iter_0000047)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/checkpoints/iter_0000063](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/checkpoints/iter_0000063)

已有评估汇总：

- [final-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/final-eval/summary.json)
- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/final-validation/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/validation/point-rollout-00000/summary.json)
- [validation/point-rollout-00016/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/validation/point-rollout-00016/summary.json)
- [validation/point-rollout-00032/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/validation/point-rollout-00032/summary.json)
- [validation/point-rollout-00048/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/validation/point-rollout-00048/summary.json)

审计/源码：[teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/source-versions.json) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/opd-instruction64-r1/experiment-scripts)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "opd-instruction64-r1",
  "mode": "train",
  "domain": "instruction",
  "steps": 64,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 16,
  "eval_per_domain": 96,
  "max_frames": 192,
  "debug": false
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "4",
  "num-gpus-per-node": "4",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "64",
  "start-rollout-id": "0",
  "save-interval": "16",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/data/instruction-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/base-162500"
}
```

</details>

### P1 / overlap8-smoke-r1

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `0`；计划终点 `2`；本次实际新增更新 **2**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/overlap8-smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/overlap8-smoke-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/overlap8-smoke-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/overlap8-smoke-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/overlap8-smoke-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/overlap8-smoke-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/overlap8-smoke-r1/checkpoints/iter_0000001](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/overlap8-smoke-r1/checkpoints/iter_0000001)

已有评估汇总：

- [final-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/overlap8-smoke-r1/final-eval/summary.json)
- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/overlap8-smoke-r1/final-validation/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/overlap8-smoke-r1/validation/point-rollout-00000/summary.json)

审计/源码：[teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/overlap8-smoke-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/overlap8-smoke-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/overlap8-smoke-r1/source-versions.json) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/overlap8-smoke-r1/experiment-scripts)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "overlap8-smoke-r1",
  "mode": "train",
  "domain": "mixed",
  "steps": 2,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 2,
  "eval_per_domain": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "resume_from": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "4",
  "num-gpus-per-node": "4",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "2",
  "start-rollout-id": "0",
  "save-interval": "2",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/data/mixed-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/base-162500"
}
```

</details>

### P1 / smoke8-r1

首次八卡烟雾检查；误入文本eval路径而失败

状态：未完成/中止；模式 `train`；起始计数 `0`；计划终点 `2`；本次实际新增更新 **0**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r1/train.log](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r1/train.log)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

本目录没有保存训练检查点。

审计/源码：[teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r1/teacher-before.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r1/source-versions.json)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "smoke8-r1",
  "mode": "train",
  "domain": "dialect",
  "steps": 2,
  "batch": 4,
  "samples": 2,
  "lr": 3e-06,
  "eval_interval": 2,
  "eval_per_domain": 4,
  "max_frames": 192,
  "debug": true
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "8",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "4",
  "num-gpus-per-node": "4",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "2",
  "start-rollout-id": "0",
  "save-interval": "2",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/data/dialect-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/base-162500"
}
```

</details>

### P1 / smoke8-r2

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `0`；计划终点 `2`；本次实际新增更新 **2**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r2](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r2)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r2/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r2/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r2/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r2/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r2/checkpoints/iter_0000001](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r2/checkpoints/iter_0000001)

已有评估汇总：

- [final-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r2/final-eval/summary.json)
- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r2/final-validation/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r2/validation/point-rollout-00000/summary.json)

审计/源码：[teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r2/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r2/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/smoke8-r2/source-versions.json)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "smoke8-r2",
  "mode": "train",
  "domain": "dialect",
  "steps": 2,
  "batch": 4,
  "samples": 2,
  "lr": 3e-06,
  "eval_interval": 2,
  "eval_per_domain": 4,
  "max_frames": 192,
  "debug": true
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "8",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "4",
  "num-gpus-per-node": "4",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "2",
  "start-rollout-id": "0",
  "save-interval": "2",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/data/dialect-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912/models/base-162500"
}
```

</details>

### P2 / baselines-fresh-r1

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `baseline`；起始计数 `0`；计划终点 `64`；本次实际新增更新 **0**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/args.json)

更新数来源：模式无优化器更新

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

本目录没有保存训练检查点。

已有评估汇总：

- [base-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/base-eval/summary.json)
- [base-eval-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/base-eval-t0.7/summary.json)
- [base-pilot-validation-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/base-pilot-validation-t0.7/summary.json)
- [base-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/base-validation/summary.json)
- [base-validation-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/base-validation-t0.7/summary.json)
- [dialect-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/dialect-eval/summary.json)
- [dialect-eval-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/dialect-eval-t0.7/summary.json)
- [dialect-pilot-validation-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/dialect-pilot-validation-t0.7/summary.json)
- [dialect-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/dialect-validation/summary.json)
- [dialect-validation-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/dialect-validation-t0.7/summary.json)
- [instruction-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/instruction-eval/summary.json)
- [instruction-eval-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/instruction-eval-t0.7/summary.json)
- [instruction-pilot-validation-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/instruction-pilot-validation-t0.7/summary.json)
- [instruction-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/instruction-validation/summary.json)
- [instruction-validation-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/instruction-validation-t0.7/summary.json)

审计/源码：[teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/baselines-fresh-r1/policy-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "baselines-fresh-r1",
  "mode": "baseline",
  "domain": "dialect",
  "steps": 64,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 16,
  "eval_per_domain": 96,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 2,
  "debug": false,
  "skip_final_test": false,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 1.0,
  "eval_temperature": 1.0,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": null,
  "expected_weight_audit": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{}
```

</details>

### P2 / confirm-dialect-reverse-t07-64-r1

从方言32次学生恢复，再做32次，累计64次

状态：完成；模式 `train`；起始计数 `32`；计划终点 `64`；本次实际新增更新 **32**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/checkpoints/iter_0000063](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/checkpoints/iter_0000063)

已有评估汇总：

- [final-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/final-eval/summary.json)
- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/final-validation/summary.json)
- [secondary-eval-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/secondary-eval-t0.7/summary.json)
- [secondary-validation-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/secondary-validation-t0.7/summary.json)
- [validation/point-rollout-00032/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/validation/point-rollout-00032/summary.json)
- [validation/point-rollout-00032-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/validation/point-rollout-00032-t0.7/summary.json)

审计/源码：[student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-dialect-reverse-t07-64-r1/policy-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "confirm-dialect-reverse-t07-64-r1",
  "mode": "train",
  "domain": "dialect",
  "steps": 64,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": false,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 1.0,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/checkpoints",
  "expected_weight_audit": null,
  "start_step": 32
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "64",
  "start-rollout-id": "32",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout/dialect-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/models/base-162500"
}
```

</details>

### P2 / confirm-instruction-reverse-t07-64-r1

只加载已有指令64次检查点做扩大评估；新增更新0次

状态：完成；模式 `evaluate`；起始计数 `64`；计划终点 `64`；本次实际新增更新 **0**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1/args.json)

更新数来源：模式无优化器更新

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/checkpoints)。

本项不保存新的训练检查点；使用恢复来源检查点。

已有评估汇总：

- [final-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1/final-eval/summary.json)
- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1/final-validation/summary.json)
- [secondary-eval-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1/secondary-eval-t0.7/summary.json)
- [secondary-validation-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1/secondary-validation-t0.7/summary.json)

审计/源码：[evaluation-checkpoint-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1/evaluation-checkpoint-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/confirm-instruction-reverse-t07-64-r1/policy-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "confirm-instruction-reverse-t07-64-r1",
  "mode": "evaluate",
  "domain": "instruction",
  "steps": 64,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": false,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 1.0,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/student-weight-audit.json",
  "start_step": 64
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "64",
  "start-rollout-id": "64",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout/instruction-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/models/base-162500"
}
```

</details>

### P2 / dense-zero-r1

Base162500同时充当教师和学生，LR0，完整KL为0

状态：完成；模式 `train`；起始计数 `0`；计划终点 `2`；本次实际新增更新 **2**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/checkpoints/iter_0000001](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/checkpoints/iter_0000001)

已有评估汇总：

- [final-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/final-eval/summary.json)
- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/final-validation/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/validation/point-rollout-00000/summary.json)

审计/源码：[student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/policy-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "dense-zero-r1",
  "mode": "train",
  "domain": "dialect",
  "steps": 2,
  "batch": 4,
  "samples": 2,
  "lr": 0.0,
  "eval_interval": 2,
  "eval_per_domain": 2,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "teacher_control": true,
  "estimator": "dense_reverse",
  "resume_from": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "8",
  "micro-batch-size": "1",
  "lr": "0.0",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "2",
  "start-rollout-id": "0",
  "save-interval": "2",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data/dialect-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/models/base-162500"
}
```

</details>

### P2 / mopd-reverse-t07-128-r1

从Base162500重新开始，两个专家按领域路由，纯学生轨迹反向KL128次

状态：完成；模式 `train`；起始计数 `0`；计划终点 `128`；本次实际新增更新 **128**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints/iter_0000063](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints/iter_0000063)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints/iter_0000127](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints/iter_0000127)

已有评估汇总：

- [final-eval/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/final-eval/summary.json)
- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/final-validation/summary.json)
- [secondary-eval-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/secondary-eval-t0.7/summary.json)
- [secondary-validation-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/secondary-validation-t0.7/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/validation/point-rollout-00000/summary.json)
- [validation/point-rollout-00000-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/validation/point-rollout-00000-t0.7/summary.json)
- [validation/point-rollout-00064/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/validation/point-rollout-00064/summary.json)
- [validation/point-rollout-00064-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/validation/point-rollout-00064-t0.7/summary.json)

审计/源码：[student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json) · [routing-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/routing-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/policy-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "mopd-reverse-t07-128-r1",
  "mode": "train",
  "domain": "mixed",
  "steps": 128,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 64,
  "eval_per_domain": 96,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": false,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 1.0,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": null,
  "expected_weight_audit": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "128",
  "start-rollout-id": "0",
  "save-interval": "64",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout/mixed-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/models/base-162500"
}
```

</details>

### P2 / pilot-dialect-dense-forward-r1

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `0`；计划终点 `32`；本次实际新增更新 **32**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/checkpoints/iter_0000015](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/checkpoints/iter_0000015)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/checkpoints/iter_0000031](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/checkpoints/iter_0000031)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/final-validation/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/validation/point-rollout-00000/summary.json)
- [validation/point-rollout-00016/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/validation/point-rollout-00016/summary.json)

审计/源码：[student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-forward-r1/policy-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "pilot-dialect-dense-forward-r1",
  "mode": "train",
  "domain": "dialect",
  "steps": 32,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 16,
  "eval_per_domain": 24,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": false,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout",
  "teacher_control": false,
  "estimator": "dense_forward",
  "resume_from": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_forward",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "32",
  "start-rollout-id": "0",
  "save-interval": "16",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout/dialect-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/models/base-162500"
}
```

</details>

### P2 / pilot-dialect-dense-reverse-r1

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `0`；计划终点 `32`；本次实际新增更新 **32**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/checkpoints/iter_0000015](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/checkpoints/iter_0000015)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/checkpoints/iter_0000031](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/checkpoints/iter_0000031)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/final-validation/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/validation/point-rollout-00000/summary.json)
- [validation/point-rollout-00016/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/validation/point-rollout-00016/summary.json)

审计/源码：[student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-dense-reverse-r1/policy-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "pilot-dialect-dense-reverse-r1",
  "mode": "train",
  "domain": "dialect",
  "steps": 32,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 16,
  "eval_per_domain": 24,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": false,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "resume_from": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "32",
  "start-rollout-id": "0",
  "save-interval": "16",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout/dialect-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/models/base-162500"
}
```

</details>

### P2 / pilot-dialect-reverse-local-r1

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `0`；计划终点 `32`；本次实际新增更新 **32**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/checkpoints/iter_0000015](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/checkpoints/iter_0000015)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/checkpoints/iter_0000031](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/checkpoints/iter_0000031)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/final-validation/summary.json)
- [secondary-validation-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/secondary-validation-t0.7/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/validation/point-rollout-00000/summary.json)
- [validation/point-rollout-00016/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/validation/point-rollout-00016/summary.json)

审计/源码：[student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-local-r1/policy-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "pilot-dialect-reverse-local-r1",
  "mode": "train",
  "domain": "dialect",
  "steps": 32,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 16,
  "eval_per_domain": 24,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 1.0,
  "eval_temperature": 1.0,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "local_only",
  "resume_from": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1.0",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "local_only",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "32",
  "start-rollout-id": "0",
  "save-interval": "16",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout/dialect-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/models/base-162500"
}
```

</details>

### P2 / pilot-dialect-reverse-t07-r1

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `0`；计划终点 `32`；本次实际新增更新 **32**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/checkpoints/iter_0000015](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/checkpoints/iter_0000015)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/checkpoints/iter_0000031](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/checkpoints/iter_0000031)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/final-validation/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/validation/point-rollout-00000/summary.json)
- [validation/point-rollout-00016/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/validation/point-rollout-00016/summary.json)

审计/源码：[student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-reverse-t07-r1/policy-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "pilot-dialect-reverse-t07-r1",
  "mode": "train",
  "domain": "dialect",
  "steps": 32,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 16,
  "eval_per_domain": 24,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 1.0,
  "trainable_scope": "full",
  "resume_from": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "32",
  "start-rollout-id": "0",
  "save-interval": "16",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout/dialect-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/models/base-162500"
}
```

</details>

### P2 / pilot-dialect-sampled-native-r1

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `0`；计划终点 `32`；本次实际新增更新 **32**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/checkpoints/iter_0000015](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/checkpoints/iter_0000015)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/checkpoints/iter_0000031](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/checkpoints/iter_0000031)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/final-validation/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/validation/point-rollout-00000/summary.json)
- [validation/point-rollout-00016/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/validation/point-rollout-00016/summary.json)

审计/源码：[student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-dialect-sampled-native-r1/policy-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "pilot-dialect-sampled-native-r1",
  "mode": "train",
  "domain": "dialect",
  "steps": 32,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 16,
  "eval_per_domain": 24,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": false,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout",
  "teacher_control": false,
  "estimator": "sampled_native",
  "resume_from": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "1",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "sampled_native",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "32",
  "start-rollout-id": "0",
  "save-interval": "16",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout/dialect-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/models/base-162500"
}
```

</details>

### P2 / pilot-instruction-reverse-t07-r1

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `0`；计划终点 `64`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/checkpoints/iter_0000031](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/checkpoints/iter_0000031)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/checkpoints/iter_0000063](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/checkpoints/iter_0000063)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/final-validation/summary.json)
- [secondary-validation-t0.7/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/secondary-validation-t0.7/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/validation/point-rollout-00000/summary.json)
- [validation/point-rollout-00032/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/validation/point-rollout-00032/summary.json)

审计/源码：[student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/pilot-instruction-reverse-t07-r1/policy-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "pilot-instruction-reverse-t07-r1",
  "mode": "train",
  "domain": "instruction",
  "steps": 64,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 24,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 1.0,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "64",
  "start-rollout-id": "0",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/data-fresh-holdout/instruction-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/models/base-162500"
}
```

</details>

### P3 / prefix-smoke-r1

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `128`；计划终点 `130`；本次实际新增更新 **2**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/checkpoints/iter_0000129](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/checkpoints/iter_0000129)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/validation/point-rollout-00128/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/prefix-smoke-r1/policy-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "prefix-smoke-r1",
  "mode": "train",
  "domain": "mixed",
  "steps": 130,
  "batch": 2,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 128,
  "eval_per_domain": 4,
  "final_validation_per_domain": 4,
  "prefix_frames": 8,
  "prefix_weight": 4.0,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "8",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "130",
  "start-rollout-id": "128",
  "save-interval": "128",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/models/base-162500"
}
```

</details>

### P3 / repair-A-mopd64-r1

A：原反向KL继续64次

状态：完成；模式 `train`；起始计数 `128`；计划终点 `192`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/checkpoints/iter_0000159](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/checkpoints/iter_0000159)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/checkpoints/iter_0000191](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/checkpoints/iter_0000191)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/validation/point-rollout-00128/summary.json)
- [validation/point-rollout-00160/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/validation/point-rollout-00160/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "repair-A-mopd64-r1",
  "mode": "train",
  "domain": "mixed",
  "steps": 192,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "final_validation_per_domain": 288,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "192",
  "start-rollout-id": "128",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/models/base-162500"
}
```

</details>

### P3 / repair-B-replay64-r1

B：筛选提示与教师轨迹混合20%

状态：完成；模式 `train`；起始计数 `128`；计划终点 `192`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/checkpoints/iter_0000159](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/checkpoints/iter_0000159)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/checkpoints/iter_0000191](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/checkpoints/iter_0000191)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/validation/point-rollout-00128/summary.json)
- [validation/point-rollout-00160/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/validation/point-rollout-00160/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "repair-B-replay64-r1",
  "mode": "train",
  "domain": "mixed",
  "steps": 192,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "final_validation_per_domain": 288,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-replay/manifest.json",
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "192",
  "start-rollout-id": "128",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/models/base-162500"
}
```

</details>

### P3 / repair-C-prefix64-r1

C：前8帧RVQ权重4，总RVQ权重归一化

状态：完成；模式 `train`；起始计数 `128`；计划终点 `192`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/checkpoints/iter_0000159](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/checkpoints/iter_0000159)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/checkpoints/iter_0000191](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/checkpoints/iter_0000191)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/validation/point-rollout-00128/summary.json)
- [validation/point-rollout-00160/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/validation/point-rollout-00160/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "repair-C-prefix64-r1",
  "mode": "train",
  "domain": "mixed",
  "steps": 192,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "final_validation_per_domain": 288,
  "prefix_frames": 8,
  "prefix_weight": 4.0,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "192",
  "start-rollout-id": "128",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/models/base-162500"
}
```

</details>

### P3 / replay-smoke-r1

教师轨迹来源/版本校验失败；后修复显式教师行为版本处理

状态：未完成/中止；模式 `train`；起始计数 `128`；计划终点 `130`；本次实际新增更新 **0**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r1/train.log](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r1/train.log)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

本目录没有保存训练检查点。

审计/源码：[teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r1/teacher-before.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r1/policy-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "replay-smoke-r1",
  "mode": "train",
  "domain": "mixed",
  "steps": 130,
  "batch": 2,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 128,
  "eval_per_domain": 4,
  "final_validation_per_domain": 4,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-replay/manifest.json",
  "replay_every_n_groups": 2,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "8",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "130",
  "start-rollout-id": "128",
  "save-interval": "128",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/models/base-162500"
}
```

</details>

### P3 / replay-smoke-r2

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `train`；起始计数 `128`；计划终点 `130`；本次实际新增更新 **2**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/checkpoints/iter_0000129](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/checkpoints/iter_0000129)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/validation/point-rollout-00128/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/replay-smoke-r2/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "replay-smoke-r2",
  "mode": "train",
  "domain": "mixed",
  "steps": 130,
  "batch": 2,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 128,
  "eval_per_domain": 4,
  "final_validation_per_domain": 4,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-replay/manifest.json",
  "replay_every_n_groups": 2,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "8",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "130",
  "start-rollout-id": "128",
  "save-interval": "128",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/models/base-162500"
}
```

</details>

### P3 / teacher-pool-candidates

未完成候选池尝试，以r2完整候选池为准

状态：未完成/中止；模式 `teacher_pool`；起始计数 `0`；计划终点 `不适用`；本次实际新增更新 **0**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates/args.json)

更新数来源：模式无优化器更新

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

本目录没有保存训练检查点。

审计/源码：[source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates/policy-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "teacher-pool-candidates",
  "mode": "teacher_pool"
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{}
```

</details>

### P3 / teacher-pool-candidates-r2

生成1536条候选，供384条质量筛选教师池使用

状态：完成；模式 `teacher_pool`；起始计数 `0`；计划终点 `不适用`；本次实际新增更新 **0**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates-r2](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates-r2)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates-r2/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates-r2/args.json)

更新数来源：模式无优化器更新

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates-r2/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates-r2/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

本目录没有保存训练检查点。

审计/源码：[source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates-r2/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates-r2/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates-r2/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-pool-candidates-r2/policy-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "teacher-pool-candidates-r2",
  "mode": "teacher_pool"
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{}
```

</details>

### P4 / D-forward64-r1

D：全前向KL，纯学生轨迹

状态：完成；模式 `train`；起始计数 `128`；计划终点 `192`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/checkpoints/iter_0000159](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/checkpoints/iter_0000159)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/checkpoints/iter_0000191](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/checkpoints/iter_0000191)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/validation/point-rollout-00128/summary.json)
- [validation/point-rollout-00160/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/validation/point-rollout-00160/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/D-forward64-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "D-forward64-r1",
  "mode": "train",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 192,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "final_validation_per_domain": 288,
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_forward",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_forward",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "192",
  "start-rollout-id": "128",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / E-rvq1-64-r1

E：RVQ1原始权重4，其余1

状态：完成；模式 `train`；起始计数 `128`；计划终点 `192`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/checkpoints/iter_0000159](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/checkpoints/iter_0000159)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/checkpoints/iter_0000191](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/checkpoints/iter_0000191)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/validation/point-rollout-00128/summary.json)
- [validation/point-rollout-00160/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/validation/point-rollout-00160/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/E-rvq1-64-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "E-rvq1-64-r1",
  "mode": "train",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 192,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "final_validation_per_domain": 288,
  "codebook_weights": [
    4.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  ],
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "192",
  "start-rollout-id": "128",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / F-rvq123-64-r1

F：前3个RVQ原始权重4，其余1

状态：完成；模式 `train`；起始计数 `128`；计划终点 `192`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/checkpoints/iter_0000159](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/checkpoints/iter_0000159)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/checkpoints/iter_0000191](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/checkpoints/iter_0000191)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/validation/point-rollout-00128/summary.json)
- [validation/point-rollout-00160/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/validation/point-rollout-00160/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/F-rvq123-64-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "F-rvq123-64-r1",
  "mode": "train",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 192,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "final_validation_per_domain": 288,
  "codebook_weights": [
    4.0,
    4.0,
    4.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  ],
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "192",
  "start-rollout-id": "128",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / G-decision64-r1

G：continue/stop权重4

状态：完成；模式 `train`；起始计数 `128`；计划终点 `192`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/checkpoints/iter_0000159](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/checkpoints/iter_0000159)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/checkpoints/iter_0000191](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/checkpoints/iter_0000191)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/validation/point-rollout-00128/summary.json)
- [validation/point-rollout-00160/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/validation/point-rollout-00160/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/G-decision64-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "G-decision64-r1",
  "mode": "train",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 192,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "final_validation_per_domain": 288,
  "codebook_weights": null,
  "decision_weight": 4.0,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "192",
  "start-rollout-id": "128",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / H-matched-reverse64-r1

H：20%精确同提示教师轨迹，反向KL；混合GKD

状态：完成；模式 `train`；起始计数 `128`；计划终点 `192`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/checkpoints/iter_0000159](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/checkpoints/iter_0000159)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/checkpoints/iter_0000191](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/checkpoints/iter_0000191)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/validation/point-rollout-00128/summary.json)
- [validation/point-rollout-00160/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/validation/point-rollout-00160/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/H-matched-reverse64-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "H-matched-reverse64-r1",
  "mode": "train",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 192,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "final_validation_per_domain": 288,
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-replay/manifest.json",
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "192",
  "start-rollout-id": "128",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / I-matched-forward64-r1

I：与H完全相同缓存和位置，改为前向KL

状态：完成；模式 `train`；起始计数 `128`；计划终点 `192`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/checkpoints/iter_0000159](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/checkpoints/iter_0000159)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/checkpoints/iter_0000191](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/checkpoints/iter_0000191)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/validation/point-rollout-00128/summary.json)
- [validation/point-rollout-00160/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/validation/point-rollout-00160/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/I-matched-forward64-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "I-matched-forward64-r1",
  "mode": "train",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 192,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "final_validation_per_domain": 288,
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-replay/manifest.json",
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_forward",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_forward",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "192",
  "start-rollout-id": "128",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / J-selected-prompt64-r1

J：与B提示选择一致，全部轨迹由当前学生生成

状态：完成；模式 `train`；起始计数 `128`；计划终点 `192`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/checkpoints/iter_0000159](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/checkpoints/iter_0000159)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/checkpoints/iter_0000191](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/checkpoints/iter_0000191)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/validation/point-rollout-00128/summary.json)
- [validation/point-rollout-00160/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/validation/point-rollout-00160/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/J-selected-prompt64-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "J-selected-prompt64-r1",
  "mode": "train",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 192,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "final_validation_per_domain": 288,
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "prompt_only",
  "teacher_parent_control": false,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-replay/manifest.json",
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "192",
  "start-rollout-id": "128",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / K-domain-direction64-r1

K：方言反向KL，指令前向KL，统一权重

状态：完成；模式 `train`；起始计数 `128`；计划终点 `192`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/checkpoints/iter_0000159](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/checkpoints/iter_0000159)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/checkpoints/iter_0000191](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/checkpoints/iter_0000191)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/validation/point-rollout-00128/summary.json)
- [validation/point-rollout-00160/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/validation/point-rollout-00160/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/K-domain-direction64-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "K-domain-direction64-r1",
  "mode": "train",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 192,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "final_validation_per_domain": 288,
  "domain_loss_config": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-loss/K.json",
  "audit_schedule_file": null,
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "192",
  "start-rollout-id": "128",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / L-domain-composition64-r1

L：方言反向KL+前3码本加权，指令前向KL

状态：完成；模式 `train`；起始计数 `128`；计划终点 `192`；本次实际新增更新 **64**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/checkpoints/iter_0000159](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/checkpoints/iter_0000159)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/checkpoints/iter_0000191](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/checkpoints/iter_0000191)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/validation/point-rollout-00128/summary.json)
- [validation/point-rollout-00160/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/validation/point-rollout-00160/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/L-domain-composition64-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "L-domain-composition64-r1",
  "mode": "train",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 192,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 32,
  "eval_per_domain": 96,
  "final_validation_per_domain": 288,
  "domain_loss_config": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-loss/L.json",
  "audit_schedule_file": null,
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "192",
  "start-rollout-id": "128",
  "save-interval": "32",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / baseline-experts-probes

Base/专家成对控制探针；4条/域小型验证不作为主288样本结果

状态：完成；模式 `baseline`；起始计数 `0`；计划终点 `64`；本次实际新增更新 **0**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes/args.json)

更新数来源：模式无优化器更新

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

本目录没有保存训练检查点。

已有评估汇总：

- [base-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes/base-validation/summary.json)
- [dialect-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes/dialect-validation/summary.json)
- [instruction-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes/instruction-validation/summary.json)

审计/源码：[teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-experts-probes/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "baseline-experts-probes",
  "mode": "baseline",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 64,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 16,
  "eval_per_domain": 4,
  "final_validation_per_domain": 288,
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": null,
  "expected_weight_audit": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{}
```

</details>

### P4 / baseline-vc-parent

直接评VC父模型；目录instruction-validation包含两个domain，不是指令专家

状态：完成；模式 `baseline`；起始计数 `0`；计划终点 `64`；本次实际新增更新 **0**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-vc-parent](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-vc-parent)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-vc-parent/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-vc-parent/args.json)

更新数来源：模式无优化器更新

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-vc-parent/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-vc-parent/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

本目录没有保存训练检查点。

已有评估汇总：

- [instruction-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-vc-parent/instruction-validation/summary.json)

审计/源码：[teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-vc-parent/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-vc-parent/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-vc-parent/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-vc-parent/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-vc-parent/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-vc-parent/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/baseline-vc-parent/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "baseline-vc-parent",
  "mode": "baseline",
  "baseline_models": [
    "instruction"
  ],
  "domain": "mixed",
  "steps": 64,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 16,
  "eval_per_domain": 288,
  "final_validation_per_domain": 288,
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": true,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": null,
  "expected_weight_audit": null,
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{}
```

</details>

### P4 / domain-smoke-r1

L分领域配置的2次更新检查

状态：完成；模式 `train`；起始计数 `128`；计划终点 `130`；本次实际新增更新 **2**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/checkpoints/iter_0000129](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/checkpoints/iter_0000129)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/validation/point-rollout-00128/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-smoke-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "domain-smoke-r1",
  "mode": "train",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 130,
  "batch": 2,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 128,
  "eval_per_domain": 4,
  "final_validation_per_domain": 4,
  "domain_loss_config": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/domain-loss/L.json",
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": false,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "8",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "130",
  "start-rollout-id": "128",
  "save-interval": "128",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / matched-smoke-r1

精确位置轨迹+前向KL实现检查

状态：完成；模式 `train`；起始计数 `128`；计划终点 `130`；本次实际新增更新 **2**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/checkpoints/iter_0000129](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/checkpoints/iter_0000129)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/validation/point-rollout-00128/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-smoke-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "matched-smoke-r1",
  "mode": "train",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 130,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 128,
  "eval_per_domain": 4,
  "final_validation_per_domain": 4,
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": false,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-replay/manifest.json",
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_forward",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_forward",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "130",
  "start-rollout-id": "128",
  "save-interval": "128",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / matched-teacher-cache-r1

按原始A续训位置和种子生成820条教师轨迹，不进行质量筛选

状态：完成；模式 `teacher_pool`；起始计数 `0`；计划终点 `不适用`；本次实际新增更新 **0**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-teacher-cache-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-teacher-cache-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-teacher-cache-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-teacher-cache-r1/args.json)

更新数来源：模式无优化器更新

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-teacher-cache-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-teacher-cache-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

本目录没有保存训练检查点。

审计/源码：[source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-teacher-cache-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-teacher-cache-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-teacher-cache-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-teacher-cache-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/matched-teacher-cache-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "matched-teacher-cache-r1",
  "mode": "teacher_pool"
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{}
```

</details>

### P4 / parent-vc128-r1

Base162500由共同VC200000父模型指导128次；不是从VC初始化学生

状态：完成；模式 `train`；起始计数 `0`；计划终点 `128`；本次实际新增更新 **128**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

初始化：原生Base162500，`--ckpt-step 162500`；`dense-zero-r1`另用同一Base作教师。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/checkpoints/iter_0000063](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/checkpoints/iter_0000063)
- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/checkpoints/iter_0000127](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/checkpoints/iter_0000127)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/final-validation/summary.json)
- [validation/point-rollout-00000/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/validation/point-rollout-00000/summary.json)
- [validation/point-rollout-00064/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/validation/point-rollout-00064/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/parent-vc128-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "parent-vc128-r1",
  "mode": "train",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 128,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 64,
  "eval_per_domain": 96,
  "final_validation_per_domain": 288,
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": true,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": null,
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/dense-zero-r1/student-weight-audit.json",
  "start_step": 0
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "128",
  "start-rollout-id": "0",
  "save-interval": "64",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / probe-A

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `evaluate`；起始计数 `192`；计划终点 `192`；本次实际新增更新 **0**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A/args.json)

更新数来源：模式无优化器更新

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/checkpoints)。

本项不保存新的训练检查点；使用恢复来源检查点。

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A/final-validation/summary.json)

审计/源码：[evaluation-checkpoint-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A/evaluation-checkpoint-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-A/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "probe-A",
  "mode": "evaluate",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 192,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 16,
  "eval_per_domain": 4,
  "final_validation_per_domain": 4,
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/student-weight-audit.json",
  "start_step": 192
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "192",
  "start-rollout-id": "192",
  "save-interval": "16",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / probe-B

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `evaluate`；起始计数 `192`；计划终点 `192`；本次实际新增更新 **0**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B/args.json)

更新数来源：模式无优化器更新

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/checkpoints)。

本项不保存新的训练检查点；使用恢复来源检查点。

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B/final-validation/summary.json)

审计/源码：[evaluation-checkpoint-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B/evaluation-checkpoint-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-B/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "probe-B",
  "mode": "evaluate",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 192,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 16,
  "eval_per_domain": 4,
  "final_validation_per_domain": 4,
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/student-weight-audit.json",
  "start_step": 192
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "192",
  "start-rollout-id": "192",
  "save-interval": "16",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / probe-C

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `evaluate`；起始计数 `192`；计划终点 `192`；本次实际新增更新 **0**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C/args.json)

更新数来源：模式无优化器更新

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/checkpoints)。

本项不保存新的训练检查点；使用恢复来源检查点。

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C/final-validation/summary.json)

审计/源码：[evaluation-checkpoint-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C/evaluation-checkpoint-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-C/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "probe-C",
  "mode": "evaluate",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 192,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 16,
  "eval_per_domain": 4,
  "final_validation_per_domain": 4,
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/student-weight-audit.json",
  "start_step": 192
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "192",
  "start-rollout-id": "192",
  "save-interval": "16",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / probe-mopd128

固定配置的基线/试验；详见本项参数和阶段结果

状态：完成；模式 `evaluate`；起始计数 `128`；计划终点 `128`；本次实际新增更新 **0**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128/args.json)

更新数来源：模式无优化器更新

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

本项不保存新的训练检查点；使用恢复来源检查点。

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128/final-validation/summary.json)

审计/源码：[evaluation-checkpoint-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128/evaluation-checkpoint-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/probe-mopd128/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "probe-mopd128",
  "mode": "evaluate",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 128,
  "batch": 16,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 16,
  "eval_per_domain": 4,
  "final_validation_per_domain": 4,
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": true,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "64",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "128",
  "start-rollout-id": "128",
  "save-interval": "16",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / prompt-smoke-r1

prompt_only模式实现检查，烟雾比例为每2组一次

状态：完成；模式 `train`；起始计数 `128`；计划终点 `130`；本次实际新增更新 **2**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/checkpoints/iter_0000129](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/checkpoints/iter_0000129)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/validation/point-rollout-00128/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/prompt-smoke-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "prompt-smoke-r1",
  "mode": "train",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 130,
  "batch": 2,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 128,
  "eval_per_domain": 4,
  "final_validation_per_domain": 4,
  "codebook_weights": null,
  "decision_weight": 1,
  "replay_mode": "prompt_only",
  "teacher_parent_control": false,
  "counterfactual": false,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-replay/manifest.json",
  "replay_every_n_groups": 2,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "8",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "130",
  "start-rollout-id": "128",
  "save-interval": "128",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>

### P4 / weighting-smoke-r1

RVQ/决策权重联合实现检查

状态：完成；模式 `train`；起始计数 `128`；计划终点 `130`；本次实际新增更新 **2**。运行完成不等于质量验收通过。

运行目录：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1)

原始配置：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/args.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/args.json)

更新数来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/quality-run-audit.json)

实际命令与服务启动参数：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/commands.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/commands.json)。历史localhost端口只是该次运行的地址，不代表当前仍有服务。

恢复来源：[/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints)。

实际保存的检查点：

- [/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/checkpoints/iter_0000129](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/checkpoints/iter_0000129)

已有评估汇总：

- [final-validation/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/final-validation/summary.json)
- [validation/point-rollout-00128/summary.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/validation/point-rollout-00128/summary.json)

审计/源码：[quality-run-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/quality-run-audit.json) · [student-weight-audit.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/student-weight-audit.json) · [teacher-before.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/teacher-before.json) · [teacher-after.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/teacher-after.json) · [source-versions.json](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/source-versions.json) · [policy-source.patch](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/policy-source.patch) · [experiment-scripts](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/experiment-scripts) · [policy-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/policy-source) · [shared-source](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/weighting-smoke-r1/shared-source)

<details>
<summary>展开该次运行保存的完整 args.json</summary>

```json
{
  "name": "weighting-smoke-r1",
  "mode": "train",
  "baseline_models": [
    "base",
    "dialect",
    "instruction"
  ],
  "domain": "mixed",
  "steps": 130,
  "batch": 2,
  "samples": 4,
  "lr": 3e-06,
  "eval_interval": 128,
  "eval_per_domain": 4,
  "final_validation_per_domain": 4,
  "codebook_weights": [
    4.0,
    4.0,
    4.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  ],
  "decision_weight": 4.0,
  "replay_mode": "teacher",
  "teacher_parent_control": false,
  "counterfactual": false,
  "prefix_frames": 0,
  "prefix_weight": 4,
  "replay_manifest": null,
  "replay_every_n_groups": 5,
  "seed": 1234,
  "rollout_seed": 20260956,
  "eval_concurrency": 4,
  "max_frames": 192,
  "generation_replicas": 4,
  "debug": true,
  "skip_final_test": true,
  "test_per_domain": 192,
  "data_dir": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data",
  "teacher_control": false,
  "estimator": "dense_reverse",
  "temperature": 0.7,
  "eval_temperature": 0.7,
  "secondary_eval_temperature": 0.7,
  "trainable_scope": "full",
  "resume_from": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "expected_weight_audit": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/student-weight-audit.json",
  "start_step": 128
}
```

</details>

<details>
<summary>展开命令/日志中可核验的训练参数（重复参数取最后一次）</summary>

```json
{
  "rollout-temperature": "0.7",
  "rollout-top-p": "1",
  "rollout-top-k": "-1",
  "rollout-seed": "20260956",
  "seed": "1234",
  "global-batch-size": "8",
  "micro-batch-size": "1",
  "lr": "3e-06",
  "lr-decay-style": "constant",
  "weight-decay": "0",
  "clip-grad": "1.0",
  "attention-dropout": "0",
  "hidden-dropout": "0",
  "tensor-model-parallel-size": "1",
  "pipeline-model-parallel-size": "1",
  "context-parallel-size": "1",
  "actor-num-gpus-per-node": "2",
  "num-gpus-per-node": "2",
  "seq-length": "1024",
  "max-tokens-per-gpu": "1024",
  "max-samples-per-microbatch": "2",
  "moss-local-mopd-estimator": "dense_reverse",
  "moss-local-trainable-scope": "full",
  "moss-local-old-policy-source": "trainer_preupdate",
  "num-rollout": "130",
  "start-rollout-id": "128",
  "save-interval": "128",
  "rollout-max-response-len": "192",
  "prompt-data": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/data/mixed-train.jsonl",
  "load": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/mopd-reverse-t07-128-r1/checkpoints",
  "pretrained-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite_local/RUNs/delivery/pretrain/local_pretrain_v0.0.0/checkpoints",
  "hf-checkpoint": "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/models/base-162500"
}
```

</details>
