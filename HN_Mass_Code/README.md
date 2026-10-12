# 当前模型：HN门控＋对比＋群体强度

整理日期：2026-10-12。对应本轮**单向1秒Flow、四场景强度小实验**。

## 先看这些文件

- `HN_Mass_Pseudocode.py`：当前完整流程的中文伪代码，从输入到HN评分、群体强度及最终输出。
- `HN_GatedContrast_Pseudocode.py`：原HN门控＋对比的详细张量与训练伪代码。
- `real_code/viot_nids_graph_pilot/group_intensity.py`：本次实际运行的强度提取、正常训练、评分和融合实现。
- `real_code/viot_nids_graph_pilot/fusion.py`：HN表示融合、形态及到达预测。
- `real_code/viot_nids_graph_pilot/network.py`：共享encoder、图关系注意力、Local及密度损失。
- `real_code/viot_nids_graph_pilot/data.py`、`graph.py`：Flow提取、匿名结构及邻居查询。
- `SOURCE_MAP.csv`：每份真实源码的原始绝对路径。真实代码均为原文件的原样副本。

这是便于集中阅读的源码目录。运行环境、公共数据读取/训练工具、数据、checkpoint仍复用原工程；该目录不是可脱离原工程运行的独立软件包。

## “HN强度”到底是什么？

本轮名称为 **HN＋Mass**：已有HN门控＋对比的预测分数，加上新增的群体强度预测分数。

原HN checkpoint保持冻结。新增Mean、Mass两个预测器各在全部12,518条正常Flow形成的19,109个正常活动-行为对上训练20轮。当前组合没有重新进行端到端联合训练，也没有重新学习融合门控。

| 部分 | 输入/来源 | 输出 |
| --- | --- | --- |
| 原HN | 25维Flow统计、匿名结构查询到的历史/邻居统计、关系类型、时间差、新连接标记和capture已运行时间 | 三组mask下的形态与到达节奏复合NLL |
| 新增Mass | 同一匿名节点、同一方向的过去16秒活动，每秒16类行为各自的Flow数/包数/字节数，共48维；当前行为类型和方向 | 当前行为的正常三维强度分布 |
| 强度评分 | 当前实际Flow数、包数、字节数，按正常拟合的log1p＋Robust尺度变换 | 当前行为三维平均NLL |
| 最终融合 | 原HN分数与发送端Mass分数，用正常Flow的Q50/Q95对齐 | `Z(HN) + Z(Mass)`，不截断 |

当前实际数量作为需要解释的**评分目标**；预测头只读取过去数量、行为类型和方向。16个行为类型由正常测量encoder与锚点产生，不是端口类别、视频/控制业务标签或攻击标签。

每条Flow的主强度项查询其 **发送匿名节点 × 行为类型 × 当前秒**。当前计数包含该Flow自身；这是对完整群体量的评分。接收端另存诊断。原HN的图邻居查询排除自身。

## 端口、IP在哪里用？

| 信息 | Flow/索引构造 | 进入神经网络？ |
| --- | --- | --- |
| 原始IP数值、端口数值、协议编号 | 构造单向五元组Flow及匿名索引 | 不输入这些数值 |
| 匿名IP节点 | 查找同一节点及其发送/接收活动历史 | 索引编号不输入，无身份embedding |
| 匿名`(IP,port)`端点与连接 | 原HN据此查邻居、同向/反向连接历史 | 编号不输入；查询所得关系及历史仍保留连接结构信息 |
| 端口类别、服务名、payload、TCP flags | 本轮没有这些模型特征 | 不输入 |
| 攻击标签、RTCP等业务标签 | 冻结评分后的评估与诊断 | 不输入 |

**整个流程仍使用端口相等关系和五元组聚合定义。** 端口变化可能改变Flow划分、连接新旧和图关联，进而影响检测；因此不能把它描述成“完全不使用任何端口有关的信息”。新增强度分组只用匿名IP节点、方向、行为和秒，跨端口完整累计。

源码中的`flags`是`[new_connection, capture_elapsed / 64]`，不是TCP flags。关系类型embedding仅编码发送/接收/同向/反向等关系，不编码端点身份。

## 原HN结构

```text
25维统计 → log1p＋正常Robust尺度 → 共享25→64→32 encoder
    ├─ 遮蔽目标＋mask → Local隐藏表示 l（64）
    ├─ 遮蔽目标＋mask＋辅助标记＋私有TCG历史 → c（64）
    └─ 残差门控 u = c + g * D([l,c])
            ├─ 两分量Gaussian形态预测
            │    相关历史/当前邻居只修正形态均值
            └─ 完整目标表示＋私有历史＋标记＋u → 到达间隔预测
三组固定mask的复合NLL平均 → 原HN分数
```

原HN训练有关系来源对比辅助任务，测试时critic不作为异常分数。新增群体强度没有重新做对比学习。

## 强度支路结构

```text
过去16秒 × 48维实际活动
    → GRU(48,64)
    → 拼接16维行为one-hot、2维方向one-hot
    → MLP(82,64,8)
    → 8个正常活动状态的混合权重
    → 当前行为三维Gaussian混合分布
当前实际[Flow数,包数,字节数] → 正常尺度变换 → 平均NLL
```

状态内容只在正常训练中学习，测试冻结。均值限制在正常变换空间的观测范围。这里是变换空间的连续密度近似，不是精确的离散计数概率。

只对当前出现的行为通道评分；真实零活动进入历史，capture开始前的padding不更新GRU。模型学习的是“这类出现后，量是否合理”，没有单独学习“这类是否应该出现”的概率。

## 当前输出与边界

- 每条Flow保存原HN分数、Mean/Mass分数、两种诊断融合分数、发送端群体三项数量和历史秒数。
- 同一发送节点、行为和秒内的Flow共享强度项；成员归因尚未解决。
- 四场景宏平均原HN为AUROC 0.9332／插值EER 10.92%，HN＋Mass为0.9903／2.62%。
- 四场景混合行为群体中的正常误报由116/286升至131/286（各捕获各模型自身最近经验EER点）；不能声称污染已解决。
- 冻结Mass模型后去掉历史，Mirai AUROC几乎不变；当前主要收益来自完整数量。
- 这是单种子、四场景模块实验；没有扩大到22场景，也没有据攻击结果改模型或融合权重。

`results/`保存这次的小结果表和完整结论，不包含大缓存或逐Flow大文件。

## 运行入口及checkpoint

在原工程环境运行以下命令。已有完成记录会复用训练和评分结果；它是本次完整小实验入口，不会重新训练原HN。

```bash
cd /home/mfh/Desktop/test/myModel
MPLCONFIGDIR=/tmp/hn_group_matplotlib .venv_v7/bin/python -u -m viot_nids_graph_pilot.group_intensity --config configs/hn_group_intensity_pilot.yaml
```

实际模型：

- 原HN：`/home/mfh/Desktop/test/myModel/outputs/graph_fusion_pilot/HN/joint/gated-contrast/model.pt`
- Mass：`/home/mfh/Desktop/test/myModel/outputs/hn_group_intensity_pilot_20261012/mass/model.pt`
- Mean对照：`/home/mfh/Desktop/test/myModel/outputs/hn_group_intensity_pilot_20261012/mean/model.pt`
- 正常尺度：`/home/mfh/Desktop/test/myModel/outputs/hn_group_intensity_pilot_20261012/normal_score_scales.json`
- 完整输出：`/home/mfh/Desktop/test/myModel/outputs/hn_group_intensity_pilot_20261012`

本目录不复制模型权重、PCAP和大缓存，不运行训练、smoke test或批量SHA256。仅核对源码副本一致、路径存在及伪代码语法。
