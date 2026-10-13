# BehaviorMassNIDS

正常行为原型距离＋条件行为强度建模（CBIM）的独立代码整理。CBIM对应192秒 BASE_GRU Behavior-Mass，用于阅读、复现和审阅当前方法。

## 分支名称

论文统一采用**条件行为强度建模分支（Conditional Behavioral Intensity Modeling，CBIM）**。正常行为原型确定群体归属，完整FPB刻画活动强度，过去192秒活动作为正常分布预测的条件。匿名节点仅用于归属和历史检索，各节点共享预测网络参数。

FPB是群体活动观测；CBIM是学习其历史条件分布的模型。实现中的`mass`字段、类名及旧结果表保留，均与CBIM对应。

## 当前模型与结果

主分数为 `Z(prototype_distance) + Z(Mass_NLL)`。K是行为原型数；MDN固定8个状态。Flow测量encoder固定，GRU和MDN联合训练；当前没有启用HN图门控、CL、容量分配或未知行为删桶。

同192秒历史、五捕获的Flow宏平均：

| 模型 | AUROC | Average precision | 插值EER |
| --- | ---: | ---: | ---: |
| K8原型距离单支 | 0.8668 | 0.9923 | 17.45% |
| K8 BASE_GRU Mass | 0.9903 | 0.9997 | 2.23% |
| K8 Mass＋距离 | 0.9933 | 0.9998 | 1.76% |
| K16 Mass＋距离 | 0.9934 | 0.9998 | 1.59% |

K16融合的宏AUROC/EER更好；K8的OS Scan EER、RTCP正常FP和检测后端耗时有优势。各自最近经验EER点的RTCP正常FP为K8的16条、K16的26条。不能称K8原型距离单支整体最好。主结果仅覆盖Mirai、SYN DoS、UDP Flood、LRDoS Stealth、OS Scan五份捕获，不是22场景结论。

## 阅读顺序

1. [中文方法设计](docs/METHOD.zh-CN.md)：按Overall、Snapshots、两个设计、Training、Deployment六节组织。
2. [伪代码](Pseudocode.py)：训练与单秒检测逻辑。
3. `src/behavior_mass_nids/network.py`、`activity.py`、`model.py`：网络、真实数量和评分。
4. [代码与结论审阅说明](docs/REVIEW.md)：已核对事项、尚存问题、结果口径。
5. `results/reference/`：既有五场景小结果、RTCP诊断和搬移一致性证据。

`docs/SOURCE_MAP.csv`记录从原工程抽取的实际核心函数，保留原公式和模块结构。新接口负责路径、输入、模型转换和独立评估。`LongMassModel`保留旧checkpoint的两个冻结CL投影器以兼容参数文件；本版本不调用对比损失或用critic评分。

## 环境与输入

使用Python3.10；原实验为PyTorch2.5.1 CUDA12.1、GTX1650，CUDA不可用时可用CPU。具体依赖在`pyproject.toml`；搬移核对的环境记录在结果目录。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[export]'
```

缓存只含 `row_id、timestamp_second、features[N,25]、source_node、destination_node`。原始IP/端口仅在瞬时五元组聚合中使用；匿名节点索引只用于活动检索，不进入神经网络。攻击/业务标签独立保存；所有训练与预测模块都不读取标签。

## 独立运行

从原始PCAP准备单向1秒缓存，攻击标签仅用于独立评价文件：

```bash
python -m behavior_mass_nids prepare --pcap datasets/normal.pcap --out datasets/normal
python -m behavior_mass_nids prepare --pcap datasets/attack.pcap --packet-labels datasets/packet_labels.csv --out datasets/attack
```

从全部正常数据训练，保存最后完整epoch及正常参考尺度：

```bash
python -m behavior_mass_nids train --cache datasets/normal --config configs/k8.yaml --out artifacts/new_K8
```

如需与当前实验使用同一测量encoder/原型和未训练的GRU初始参数，先转换原工程模型，再执行：

```bash
python scripts/export_existing.py --workspace /path/to/myModel --k 8 --out artifacts/K8
python -m behavior_mass_nids train --cache datasets/normal --config configs/k8.yaml --measurement-bundle artifacts/K8 --reuse-initial-mass --out artifacts/retrain_K8
```

该训练仅读取测量器、原型和`initial_mass.pt`，不会续训已完成的`mass_model.pt`。独立从头训练命令是一轮新实验，不能预先保证复现本表的全部数值；精确对照应使用共享测量器和原始未训练初始化。

冻结评分、独立标签评价：

```bash
python -m behavior_mass_nids score --model artifacts/K8 --cache datasets/attack --out outputs/attack_scores.csv.gz
python -m behavior_mass_nids evaluate --scores outputs/attack_scores.csv.gz --labels datasets/attack/labels.npz --out outputs/evaluation
```

`score`也支持原工程`observations.npz＋structure.npz`缓存，以及输出NPZ。每份capture重新构建独立历史。输出形态距离、Mass NLL、正常尺度分数、最终分数、完整FPB、历史长度及训练Q99参考标记。该标记不表示已验证部署误报率。

## 上传GitHub

以本目录作为仓库根目录。源码、配置、方法说明和小结果可以提交；`.gitignore`排除PCAP、数据缓存、模型权重、虚拟环境与新输出。本地`artifacts/`中已经转换的模型仅供核对，默认不进入Git。

无需上传原工程的全部V5/V6/V7/HN实验。这里没有自动上传、远程仓库操作或创建PR。授权许可尚未由作者指定，本目录保留来源说明并未替作者选择开源许可证。
