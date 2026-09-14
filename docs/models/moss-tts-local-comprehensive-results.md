# MOSS-TTS Local：全面自动对照结果（2026-09-13）

本轮在用户指定的 CQ `train` 八卡 H200 节点完成704次正式更新、8次实现烟雾检查，以及12000条音频的独立Whisper复核，没有人工听评或人工标注。

**最明确的新进展是同提示教师轨迹混合：在纯MOPD128之后，以约20%的对应教师轨迹、80%的当前学生轨迹继续64次更新，方言内容错误相对MOPD128和同预算纯续训对照，在两套ASR中均下降。但尚未完成相对Base的全方言质量保持、跨训练种子稳定性及新保留集确认。该候选属于混合GKD，不是纯on-policy MOPD。**

完整协议、全部对照、原始CER/micro-CER、配对区间、自动声学探针、识别器可靠性诊断及检查点链接见[完整实验报告（2026-09-14补充版）](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/REPORT-20260914.md)。补充版包含前序单域OPD结果、10类指令属性的逐项分数，以及已评和未评字段清单；原始报告和冻结实验结果保留不变。

属性覆盖核验：Base、MOPD128、H、L和对应教师各有288项方言判断、288条指令音频对应的1458项属性判断。指令评分使用固定10类白名单，每条最多选择6项，因此没有覆盖输入中的全部字段。完整覆盖审计见报告内的`judge-coverage-20260914.json`链接。

各阶段的模型/数据路径、63条运行记录、完整参数、实际更新数、全部保存检查点、服务配置和复现入口汇总在[实验全记录与配置手册](moss-tts-local-opd-mopd-experiment-handbook.md)。

快速查看可打开[离线音频对比页](/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive/quick-demo-20260914/index.html)：包含全量指标、3组同句5模型对比，以及一组高低语速控制样例，共23段音频。音频为既有结果的无损FLAC封装；样例按已记录指标筛选，包含改善和上海话残留问题，选择规则随页面保存，不代表随机抽样或新增独立评估。

## 主要结果

开发集每域288条；各模型同提示、同生成种子、温度0.7。Judge是自动属性p(yes)均值，不是人工准确率。CERcap是逐条`min(CER,1)`的均值。

| 模型 | 方言Judge↑ | 方言主CERcap↓ | 指令Judge↑ | 指令主CERcap↓ |
| --- | --- | --- | --- | --- |
| Base162500 | 0.2104 | 0.0730 | 0.5220 | 0.3668 |
| 对应领域教师 | 0.4567 | 0.0463 | 0.7633 | 0.0883 |
| 纯MOPD128 | 0.2712 | 0.1162 | 0.5852 | 0.1193 |
| A：再做64次纯MOPD | 0.2873 | 0.1089 | 0.5881 | 0.1229 |
| H：同提示教师轨迹混合64次 | 0.2968 | 0.0890 | 0.5869 | 0.1245 |
| L：分领域KL方向和码本加权64次 | 0.2847 | 0.1094 | 0.6094 | 0.0776 |

H相对MOPD128的方言CER改善为：Qwen0.0272，95%区间[0.0093,0.0450]；Whisper0.0209，区间[0.0032,0.0408]。相对A的改善分别为0.0198，[0.0041,0.0363]，以及0.0181，[0.0002,0.0368]。区间是单训练种子下、按来源组配对bootstrap得到的提示不确定性。

L的指令主CERcap0.0776、Whisper CERcap0.1288、截断率6.60%，但方言主CERcap0.1094未达到0.09门槛。语速成对测试有改善，音高控制没有稳定跟随平均属性分数提升。共同VC200000父模型单教师128次对照明显弱于两个领域专家，支持专家教师带来的领域知识价值；学生初始化仍严格为Base162500。

## 验收状态与识别器局限

H是唯一通过全部主质量门槛的配置，但它的Whisper全方言CERcap0.2169，高于Base0.1553；按事前冻结的跨ASR规则，未进入额外两个完整训练种子的确认。其余候选也没有同时通过主/相对门槛和独立ASR筛选。**新最终集192条/域继续封存，没有进行最终集评估。**

独立ASR自身存在明显的方言识别偏差：真实参考codes重建音频的Whisper方言CERcap为0.3164，方言教师为0.2708；上海话参考音频高达0.7228。不能将这些错误全部归因于学生内容质量。

同时，Qwen主评估中上海话CERcap仍为：Base0.1539、教师0.1673、MOPD128为0.4118、H为0.3251。H修复了一部分内容问题，但上海话依然落后于Base和教师；全方言平均值掩盖了这一类别短板。

参考音频在两套ASR上均CER≤0.2的诊断子集只有125/288条，不含上海话。该子集上H的Qwen/Whisper CERcap为0.0149/0.0630，Base为0.0369/0.0772；Whisper改善区间仍跨0。此项是事后探索性可靠性诊断，未替换主指标或放宽门槛。

## 可复现资产

- 产物根目录：`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive`
- 驱动/分析根目录：`/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/runs/moss-domain-mopd/20260913-comprehensive`
- H研究候选：`H-matched-reverse64-r1/checkpoints/iter_0000191`；L指令内容对照：`L-domain-composition64-r1/checkpoints/iter_0000191`。
- `confirmation-decision.json`记录未选择确认候选及完整输入哈希；`asr-reliability-analysis.json`保留所有诊断子集规模与分布。
- 所有正式训练和烟雾检查的330学生参数同步、教师冻结、轨迹来源、实际更新数及优化器/数据游标审计通过。实际Notebook上167项MOSS+44项相关通用测试通过。

本轮新增实现的码本/决策加权、精确位置教师轨迹、仅提示选择和分领域KL配置见[实现文档](moss-tts-local-mopd.md)。当前结论不覆盖ASMR、带参考音频的VoiceClone、长文本、多语言或人工听感。
