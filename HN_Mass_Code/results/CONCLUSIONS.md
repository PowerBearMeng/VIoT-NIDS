# HN群体强度小实验：结论

单种子、四份预声明捕获，单向1秒Flow；两种群体预测器各正常训练20轮，原HN权重和原评分保持不变。攻击数据仅用于冻结后的评估。

## 如何利用

每秒按匿名发送节点和16个正常学习的行为通道，完整累计Flow数、包数、字节数。端口只用于Flow/匿名关系，数量跨端口累计；数值IP/端口、业务标签和攻击标签不进网络。

16秒历史48维观测由GRU编码，与当前行为通道及方向选择8个正常活动状态。状态内容只由正常数据学习，测试冻结，component均值限制在正常变换空间的观测范围。当前群体数量只作为评分目标，不输入预测参数。

目标是某类行为出现后的量是否合理，未训练该类是否出现的概率。这样不单独用稀有性惩罚控制通信，但本轮也不能证明已解决低量、跨秒出现频率异常。

## 同一批Flow的结果

| scenario | HN_AUROC | HN_EER_pct | Mean_AUROC | Mean_EER_pct | Mass_AUROC | Mass_EER_pct | HN_Mass_AUROC | HN_Mass_AP | HN_Mass_EER_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mirai Single Bot / single-bot | 0.8701 | 17.8231 | 0.7501 | 29.9263 | 0.9917 | 2.3634 | 0.9840 | 0.9999 | 4.1743 |
| SYN DoS / camera-udp | 0.9515 | 13.4208 | 0.8838 | 18.1804 | 0.9879 | 2.3334 | 0.9878 | 1.0000 | 2.8205 |
| LRDoS Stealth / camera-udp | 0.9169 | 11.5945 | 0.6450 | 35.8307 | 0.9960 | 0.7747 | 0.9919 | 0.9993 | 2.9697 |
| OS Scan / camera-udp | 0.9943 | 0.8539 | 0.6998 | 32.2385 | 0.9920 | 1.4555 | 0.9974 | 0.9999 | 0.5240 |

四场景宏平均：

| branch | AUROC | AUPRC | EER_percent | captures |
| --- | --- | --- | --- | --- |
| HN | 0.9332 | 0.9978 | 10.9231 | 4 |
| HN_Mass | 0.9903 | 0.9998 | 2.6221 | 4 |
| HN_Mean | 0.8693 | 0.9877 | 18.0113 | 4 |
| Mass | 0.9919 | 0.9997 | 1.7317 | 4 |
| Mean | 0.7447 | 0.9718 | 29.0440 | 4 |

主比较HN_Mass为预声明的正常Q50/Q95不截断对齐后等权相加，仅作模块诊断；不是新训练的端到端融合网络。Mass单支更好的场景保留为诊断，没有据攻击结果修改主评分。

## 群体量证据

同一真实正常活动单元保持历史及标记不变，当前全部Flow重复，Mean分数不变；Mass分数随量变化。

| variant | multiplier | normal_row_id | behavior | original_group_flows | selected_behavior_flows | score |
| --- | --- | --- | --- | --- | --- | --- |
| mean | 1 | 266 | 0 | 4 | 2 | -0.7108 |
| mean | 2 | 266 | 0 | 4 | 2 | -0.7108 |
| mean | 10 | 266 | 0 | 4 | 2 | -0.7108 |
| mean | 100 | 266 | 0 | 4 | 2 | -0.7108 |
| mean | 1000 | 266 | 0 | 4 | 2 | -0.7108 |
| mass | 1 | 266 | 0 | 4 | 2 | 1.8135 |
| mass | 2 | 266 | 0 | 4 | 2 | 4.0472 |
| mass | 10 | 266 | 0 | 4 | 2 | 13.2243 |
| mass | 100 | 266 | 0 | 4 | 2 | 47.7346 |
| mass | 1000 | 266 | 0 | 4 | 2 | 108.0462 |

该对照是实际正常观测的数量核对，不是用于报告检测指标的合成攻击。

## 归因和正常误报

| branch | subset | normal | attack | FP | FN | FPR | FNR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HN | RTCP | 1959 | 0 | 63 | 0 | 0.0322 | — |
| HN | normal_in_mixed_sender_behavior | 286 | 0 | 116 | 0 | 0.4056 | — |
| HN | normal_in_pure_sender_behavior | 10844 | 0 | 846 | 0 | 0.0780 | — |
| HN | single60 | 114 | 716888 | 47 | 183539 | 0.4123 | 0.2560 |
| HN_Mass | RTCP | 1959 | 0 | 26 | 0 | 0.0133 | — |
| HN_Mass | normal_in_mixed_sender_behavior | 286 | 0 | 131 | 0 | 0.4580 | — |
| HN_Mass | normal_in_pure_sender_behavior | 10844 | 0 | 111 | 0 | 0.0102 | — |
| HN_Mass | single60 | 114 | 716888 | 54 | 17 | 0.4737 | 0.0000 |
| Mass | RTCP | 1959 | 0 | 26 | 0 | 0.0133 | — |
| Mass | normal_in_mixed_sender_behavior | 286 | 0 | 191 | 0 | 0.6678 | — |
| Mass | normal_in_pure_sender_behavior | 10844 | 0 | 0 | 0 | 0.0000 | — |
| Mass | single60 | 114 | 716888 | 54 | 20 | 0.4737 | 0.0000 |

各分支/各捕获采用自身最近经验EER点统计FP/FN；不是统一部署阈值。主EER为插值，群体强度单支存在大量并列分数，实际阈值的FPR/FNR可与交点有差异。

四场景正常RTCP误报：HN 63/1959（3.22%）→HN_Mass 26/1959（1.33%）。Mirai为54/534→3/534。

同一发送节点、行为、秒内混入攻击的正常Flow：HN 116/286（40.56%）→HN_Mass 131/286（45.80%）。

这证明检测群体量的收益可以很大，但群体内部归因尚未解决。当前同组强度分数相同；不同类型/节点避免了部分污染，同一粗行为组的正常成员仍可被一同打高分。更细匿名通信关系和私有历史需要独立验证，不能承诺完全相同观测的Flow可唯一归因。

## 历史贡献的诊断

| scenario | branch | AUROC | AUPRC | EER |
| --- | --- | --- | --- | --- |
| Mirai Single Bot / single-bot | Mass_actual_history | 0.9917 | 1.0000 | 0.0236 |
| Mirai Single Bot / single-bot | Mass_history_removed_diagnostic | 0.9917 | 1.0000 | 0.0236 |
| Mirai Single Bot / single-bot | Mass_receiver_diagnostic | 0.9890 | 0.9999 | 0.0236 |
| SYN DoS / camera-udp | Mass_actual_history | 0.9879 | 1.0000 | 0.0233 |
| SYN DoS / camera-udp | Mass_history_removed_diagnostic | 0.9879 | 1.0000 | 0.0233 |
| SYN DoS / camera-udp | Mass_receiver_diagnostic | 0.9896 | 1.0000 | 0.0341 |
| LRDoS Stealth / camera-udp | Mass_actual_history | 0.9960 | 0.9994 | 0.0077 |
| LRDoS Stealth / camera-udp | Mass_history_removed_diagnostic | 0.9959 | 0.9994 | 0.0103 |
| LRDoS Stealth / camera-udp | Mass_receiver_diagnostic | 0.9969 | 0.9997 | 0.0107 |
| OS Scan / camera-udp | Mass_actual_history | 0.9920 | 0.9994 | 0.0146 |
| OS Scan / camera-udp | Mass_history_removed_diagnostic | 0.9909 | 0.9993 | 0.0146 |
| OS Scan / camera-udp | Mass_receiver_diagnostic | 0.9835 | 0.9982 | 0.0200 |

置零历史是冻结同一模型后的诊断，没有训练新的无历史对照。Mirai AUROC 0.991738→0.991731；SYN也几乎不变。当前证据主要支持恢复群体强度，尚不能把GRU或视频状态协同称为主要收益来源。

## 稀疏阶段与群体指标

Mirai发送端活动-行为群体AUROC：Mean 0.9224，Mass 0.9176；Flow指标则为Mean 0.7501、Mass 0.9917。两种单位不同，密集DoS群体在Flow指标中权重大；高Flow AUROC不能解读成Mirai所有生命周期都已准确识别。

群体标签为含任一攻击Flow即攻击相关；归因分析仍按独立Flow标签检查正常成员。

## TFusion背景参照

| scenario | AUROC | AUPRC | EER_percent |
| --- | --- | --- | --- |
| SYN DoS / camera-udp | 0.9832 | 0.9999 | 2.5641 |
| LRDoS Stealth / camera-udp | 0.9348 | 0.9886 | 10.8808 |
| OS Scan / camera-udp | 0.9442 | 0.9947 | 9.1279 |
| Mirai Single Bot / single-bot | 0.9810 | 0.9999 | 3.1662 |

使用既有1秒TFusion，不新增训练。SYN camera-udp多一条攻击Flow，沿用原对齐说明。只比较本次四份，不替代全22场景结论。

## 后续设计依据

保留有归属的完整量作为独立正常性证据；维持测试冻结的正常参考。进一步重点是混合群体的成员归属和学习融合，避免原HN噪声削弱强度支路。

本轮全量正常训练、无独立正常验证、种子42。没有测试正常合法高并发的新模式，不保证凡数量高都恶意。未依据攻击结果调整模型、轮数、权重或阈值。

设计：DesignHNGroupIntensity.md；配置：configs/hn_group_intensity_pilot.yaml；代码：viot_nids_graph_pilot/group_intensity.py。完整曲线、连续分数及全部错误记录在benchmark各capture目录。
