# MOSS TTS Local 双域 MOPD：质量修复实验记录（2026-09-13）

本轮三组正式实验全部完成并通过工程审计。结论是：**完整词表 reverse-KL MOPD 已表现出方言与指令属性迁移，但方言内容准确率问题仍未解决；三种修复方案均未通过事先规定的质量门槛。** 教师回放与前缀加权对部分属性有正向信号，不能据此宣称两域内容质量同时达标。

学生的祖先仍是用户指定的 `local_pretrain_v0.0.0 / iter_0162500`。上轮先从该 Base 完成128次双域MOPD更新；本轮A/B/C分别从同一个MOPD128 checkpoint及优化器状态继续64次更新，各自累计192次后训练更新。两个教师均为用户指定的方言/指令 `v0.0.1 / iter_0006000`，其父模型为VoiceClone阶段200000步模型。

本轮是上一轮验证之后的质量修复研究。上轮单域OPD与MOPD的独立评估保存在 [上一轮报告](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260912-notebook-v2/REPORT.md)；本报告的288条/域由上轮96条验证和192条最终测试合并而成，现全部视为开发数据。不要将本轮结果标为新的独立最终测试。

三组共同设置为：全参数训练、native完整词表KL(student‖teacher)、温度0.7、LR 3e-6、梯度裁剪1、每步16个提示×4条轨迹=64条、最多192音频帧；推理评估温度也为0.7。每条提示按domain路由到对应冻结教师。B明确称为混合GKD，A/C保持学生轨迹的on-policy采样。

- A：原MOPD继续64步。
- B：每五个提示组有一个使用同domain、同来源桶的质量筛选教师轨迹；其余使用当前学生生成。保存真实教师生成版本、回放条目、原提示槽和当前学生评分版本。
- C：每条音频前8帧的RVQ KL系数为4，其余为1；归一化后保持总RVQ系数及continue/stop项原有权重。

最终比较使用完整288条/域。Judge是AnyAudio-Judge属性问题的平均p_yes，越高越好，**不是人工MOS或属性准确率**。CER主指标为逐句 `min(CER,1)` 的均值，越低越好；所有样本保留。未截断CER、micro-CER、ASR截断率及非性别属性均同时导出。ASR采用现有Qwen3-ASR-1.7B服务，文本做NFKC、简繁归一和字母数字字符规范化。

| 模型 | 方言 Judge ↑ | 方言 capped-CER ↓ | 指令 Judge ↑ | 指令 capped-CER ↓ | 指令生成截断率 ↓ |
|---|---:|---:|---:|---:|---:|
| Base 162500 | 0.2104 | 0.0730 | 0.5220 | 0.3668 | 95.14% |
| 对应领域教师 6000 | 0.4567 | 0.0463 | 0.7633 | 0.0883 | 9.38% |
| MOPD128 | 0.2712 | 0.1162 | 0.5852 | 0.1193 | 17.71% |
| A：原 MOPD +64 | 0.2873 | 0.1089 | 0.5881 | 0.1229 | 17.01% |
| B：20% 教师回放 +64 | 0.3053 | 0.1045 | 0.5837 | 0.1471 | 20.14% |
| C：前8帧加权 +64 | 0.3019 | 0.1153 | 0.6062 | 0.1181 | 18.40% |

主要发现如下。

1. **增加64步尚不足以修复方言内容。** A相对MOPD128的方言CER改善0.0073，95%区间为[-0.0062, 0.0218]；没有形成可靠的内容改善证据。
2. **教师回放增强了方言属性，但没有证明内容质量同步改善。** B相对MOPD128的方言Judge增加0.0342，95%区间[0.0180, 0.0519]；方言CER改善0.0117的区间仍跨0。B的指令CER点估计增加0.0279，退化区间[-0.0029, 0.0572]，尚不能断言真实退化，但未通过保留已学指令内容能力的非劣检查。B相对同预算A的各项主要差异区间均跨0，不能直接宣称B优于A。
3. **前8帧加权没有解决整体方言错误。** C相对MOPD128的方言Judge增加0.0307，区间[0.0096, 0.0530]；方言CER仅改善0.0009，区间[-0.0183, 0.0198]。指令Judge点估计增加0.0210，其区间[-0.0004, 0.0431]仍跨0。
4. **当前证据支持属性迁移，未支持双域质量验收。** 三组相对Base的两域Judge增加区间均为正，指令CER改善也很明确；方言CER相对Base仍有代价。

以下区间来自2000次按源录音组的配对bootstrap，比较同一提示及采样种子。它们描述开发集不确定性，不代表多训练种子确认。CER列为“新模型−Base”，正数表示退化。

| 方案/领域 | Judge 增加 [95%区间] | capped-CER 变化 [95%区间] |
|---|---:|---:|
| A/方言 | +0.0769 [+0.0532～+0.1000] | +0.0359 [+0.0144～+0.0565] |
| A/指令 | +0.0661 [+0.0374～+0.0953] | -0.2439 [-0.2891～-0.1993] |
| B/方言 | +0.0949 [+0.0697～+0.1216] | +0.0315 [+0.0105～+0.0531] |
| B/指令 | +0.0617 [+0.0354～+0.0899] | -0.2197 [-0.2684～-0.1701] |
| C/方言 | +0.0915 [+0.0648～+0.1180] | +0.0424 [+0.0219～+0.0628] |
| C/指令 | +0.0843 [+0.0557～+0.1145] | -0.2487 [-0.2916～-0.2037] |

预设门槛要求两域至少保留MOPD128相对Base的80%属性增益，方言capped-CER≤0.09且比起点改善，两域Judge增加区间下界>0，CER相对Base的退化区间上界≤0.05。A完成、B/C结果尚未产生时增加了一项保守排除条件：指令CER相对MOPD128的退化区间上界也须≤0.05，以免高错误率的Base掩盖已学内容能力的回退。该补充及时间记录见 [实验协议](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/PROTOCOL.md)。

**A、B、C均未通过方言CER目标及相对Base的非劣检查；B还未通过新增的指令保护检查。** 因此没有选出验收候选，没有追加两个训练种子，也没有运行新最终测试。另备的192条/域（共384条）新最终测试仍保留，检查到与此前所有已使用文本哈希和源录音组哈希均无重叠。它不保证排除教师预训练暴露。

逐桶结果揭示了整体均值背后的取舍。下面每个方言桶有48条开发样本，数值均为capped-CER。

| 方言/口音桶 | Base | 教师 | MOPD128 | A | B | C |
|---|---:|---:|---:|---:|---:|---:|
| chuan | 0.0604 | 0.0457 | 0.1300 | 0.1070 | 0.1014 | 0.1264 |
| chuan_mandarin | 0.0257 | 0.0092 | 0.0105 | 0.0182 | 0.0377 | 0.0316 |
| dongbei | 0.0354 | 0.0173 | 0.0214 | 0.0417 | 0.0246 | 0.0196 |
| dongbei_mandarin | 0.0389 | 0.0081 | 0.0189 | 0.0125 | 0.0050 | 0.0044 |
| guangdong_yue | 0.1236 | 0.0298 | 0.1046 | 0.0785 | 0.0791 | 0.1401 |
| shanghai | 0.1539 | 0.1673 | 0.4118 | 0.3953 | 0.3792 | 0.3700 |

上海话仍是主要难点：MOPD128为0.4118，三种方案为0.3700～0.3953，远高于Base/教师约0.154/0.167。C在上海话的点估计有所改善，但粤语桶从0.1046升到0.1401，其他桶也存在取舍，整体CER没有净改善证据。不能把少数方言的改善等同于整个方言域修复。

对上轮32条上海话和32条四川话的自动错例分析显示：上海话平均每句替换数由Base的3.34增至MOPD128的8.72；相邻三字重复和末尾静音并未同步增加。参考文本前5字和后续部分都出现错误增加。它支持继续检查文本与发音对应关系，但文本编辑位置并非音频强制对齐，也不能替代方言母语者听评。

数据准备与回放审计已经完成：保留已筛选的2304条RL提示/域；从训练提示生成1536条真实教师候选，正常stop、1～192帧、原始CER≤0.15、ASR完整返回、属性问题非空后按桶内Judge排序，选出方言每桶32条、指令每来源96条，共384条。所有入选和淘汰记录均保留。候选的属性分数没有设绝对及格线，桶内最高分也不保证方言绝对质量，例如东北普通话口音桶入选均值仍低。

B实际使用820条教师轨迹和3276条学生轨迹，教师比例20.0195%，覆盖335条不同教师轨迹；方言回放424条、指令396条。三组总领域曝光均为方言2000条、指令2096条，数据游标一致。B同时包含轨迹来源变化和质量筛选提示分布变化，不能单独识别这两种因素的因果贡献。它没有使用数据集真实音频codes做监督训练。

验证与资源记录如下。

- 实际Notebook环境的153项MOSS Local测试及44项通用版本检查全部通过，共197项。两个有效小步校验各2次更新，验证前缀梯度、真实教师来源、当前学生版本、权重同步及优化器连续性。
- 三组各64次更新、4096条轨迹，合计192次正式更新和12288条训练轨迹；加上有效小步校验，本轮共196次实际优化器更新。一次回放接入探针在更新前被版本检查拒绝，修复后重跑成功；未将该失败探针计入效果。
- 每组恢复时330个学生运行时学习参数与MOPD128完全匹配；训练结束后四个生成器和学生评分器的330个参数同步一致。冻结教师审计通过，逐条领域/教师路由、优化器step192、调度累计12288条、提示组游标3072均核对一致。
- A恢复后重新生成的192段评估音频及codes与上一轮同权重/同种子结果全部一致，Judge一致；ASR的平均CER有约0.0005量级波动。极端转写会明显影响未截断CER，因此主判据使用预先约定的capped-CER，原始数据始终保留。
- 每组评估包含训练前96条/域、中途96条/域、结束288条/域，共960段；三组2880段均有完整ASR与Judge结果。另有1536条教师候选和32段有效小步评估音频。
- 仅使用CQ项目/CQ-科研驾驶舱现有train Notebook的8张H200。正式训练2卡，生成4卡，评分与两路Judge共用2卡。镜像为 `docker.sii.shaipower.online/inspire-studio/miles-moss-tts-local-env:20260908-cu130-v1`。

| 方案 | 新更新/轨迹 | 训练循环分钟 | 整个方案分钟 | 中位每步秒 |
|---|---:|---:|---:|---:|
| A | 64/4096 | 17.01 | 22.83 | 12.40 |
| B | 64/4096 | 15.95 | 21.84 | 11.24 |
| C | 64/4096 | 17.08 | 22.83 | 12.41 |

整个方案时间包含服务/模型初始化、训练、评估及权重核验，不包含随后清理；三组共约67.49分钟，按8卡占用折算约9.00 GPU小时。B还需教师候选生成与筛选的前置开销，该开销不能从样本效率比较中忽略。步数与教师6000步训练不是等算力预算，不能直接宣称几十倍训练加速。

训练和评估工作负载已退出，CPU ASR观察器已结束。`gpu-occupy`已恢复并独立核验覆盖全部8卡，无其他GPU进程遗留；核验文件为 [占卡恢复记录](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/occupancy-quality-study-final.json)。

可直接复查的产物如下。

- [完整开发集对照图](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/full-development-comparison.png) / [PDF](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/full-development-comparison.pdf)；[学习曲线](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-development-curves.png)。曲线仅使用固定96条/域子集，最终判定使用288条/域。
- [模型与领域指标CSV](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/development-metrics.csv)；[逐桶CSV](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/development-buckets.csv)；[完整统计与置信区间](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/quality-analysis.json)；[同预算方案配对比较](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-paired-comparisons.json)。
- [新旧六模型匿名试听：16组96段](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-listening/index.html)；[原始上海/四川错例盲听：64组192段](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/dialect-error-review/index.html)。均未产生人工评分；模型身份保存在各目录独立的blind-key.json中。
- [教师回放manifest](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-replay/manifest.json)；[全部候选筛选审计](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/teacher-replay/selection-audit.json)；[新测试隔离审计](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/data/final-isolation-audit.json)；[确认阶段未启动原因](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/confirmation-decision.json)。
- [驱动脚本目录](/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality)；[测试日志和源码快照](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/validation-source)。每个run也独立保存命令、版本、实际策略源码和审计。

三组模型均为Miles原生checkpoint，目录从0开始编号：

- [A：累计192次更新](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/checkpoints/iter_0000191)；[该组审计](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-A-mopd64-r1/quality-run-audit.json)。
- [B：累计192次更新](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/checkpoints/iter_0000191)；[该组审计](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-B-replay64-r1/quality-run-audit.json)。
- [C：累计192次更新](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/checkpoints/iter_0000191)；[该组审计](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-quality/repair-C-prefix64-r1/quality-run-audit.json)。

下一轮最有价值的工作是先完成固定盲听，区分真实发音错误和方言ASR偏差；然后围绕内容约束做有界消融，例如RVQ1/粗码本的独立权重或明确的文本正确性目标。若继续研究教师回放，应固定相同提示比较学生/教师轨迹，并区分内容过滤和轨迹来源的作用。当前三组都缺乏稳定内容修复证据，扩大领域前应先解决这一问题。

上述是自动评测和单训练种子开发实验的结论。没有完成新的独立测试、多训练种子确认或人工母语者评审，也没有评估长文本、单独Base通用集或VoiceClone保持性。没有设置共同VoiceClone200000父模型作为单独教师的对照，因而不能分离其通用TTS知识与6000步领域微调知识各自的贡献。它不构成对所有MOPD实现、初始化或超参数的否定。
