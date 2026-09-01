我希望你在 /home/mfh/Desktop/test/myModel 下实现一个新的无监督 Flow-level NIDS 原型。请先阅读 /home/mfh/Desktop/test/baselines 中现有 baseline 的 PCAP 读取、数据集路径组织、训练/测试方式和标签处理逻辑，尽量复用已有代码或接口，但不要修改 baseline 本身。

系统目标：

输入为 PCAP，假设应用层 payload 已加密，不允许依赖 payload、DNS/NTP/RTSP/ONVIF 等应用层语义；
检测粒度为 Flow，而不是 packet；
训练阶段只使用正常流量，不能使用攻击标签训练模型；
希望尽可能轻量，后续需要测试实时吞吐性能；
IP 和端口可以用于五元组 Flow 聚合以及建立实体关系，但不要直接作为神经网络数值特征。

第一阶段先实现 Flow representation 和无监督 Flow anomaly detection，不要一开始实现复杂 GNN。

Flow 构建
从 PCAP 中建立双向逻辑五元组 Flow；
将长 Flow 按固定 3 秒窗口切成 Flow Segment；
每个 3 秒 Segment 再划分为 100ms micro-bin，因此每个 Flow Segment 有 30 个 bin；
每个 bin 首先使用以下轻量特征：
A→B packet count
B→A packet count
A→B byte count
B→A byte count
A→B mean packet length
B→A mean packet length
不分析 payload。

最终每条 Flow Segment 表示为大约 30 x 6 的时间序列。

Flow Encoder

第一版使用轻量 TCN 或 1D-CNN encoder，不使用大 Transformer。

Encoder 将 30 x 6 Flow Segment 编码成一个 32 或 64 维 embedding：

X_flow -> encoder -> z_flow

使用正常数据进行 self-supervised 训练。第一版可以采用 masked reconstruction：随机 mask 一部分 micro-bin，让模型重建被 mask 的内容。

训练完成后，对正常训练数据生成所有 z_flow，使用 MiniBatchKMeans 建立 K 个 normal prototypes，K 做成配置项。

Flow local anomaly score 第一版定义为：

prototype_distance + reconstruction_error

两部分分别归一化后再组合。

代码结构

请尽量模块化，例如：

myModel/
  configs/
  data/
    pcap_reader.py
    flow_builder.py
    microbin_features.py
  models/
    flow_encoder.py
    flow_decoder.py
  train_flow.py
  extract_embeddings.py
  fit_prototypes.py
  evaluate_flow.py
  utils/

所有参数如：

flow window = 3s
micro-bin = 100ms
embedding dim
number of prototypes
batch size
learning rate

都放入配置文件，不要写死。

评估

训练必须只使用正常数据。

测试阶段可以读取已有标签，仅用于计算：

AUROC
AUPRC
FPR
TPR
TPR@FPR<=1%
TPR@FPR<=0.1%

阈值另外提供一个只依赖正常 validation/calibration 数据的 quantile threshold，例如目标 FPR=1% 时取正常 anomaly score 的 99% quantile。不要使用攻击样本选择部署阈值。

重要限制
不允许 DPI；
不允许使用应用层协议类型作为模型输入；
不允许使用攻击流量参与训练；
不允许把 IP/Port 数值直接送入网络；
暂时不要加入 GNN、Transformer 或复杂多模态融合；
优先保证整个 PCAP→Flow→模型→Score pipeline 能正确运行。

完成第一阶段以后，请先给我：

阅读 baseline 后找到的可复用组件；
最终目录结构；
Flow 构建规则；
特征张量 shape；
训练命令；
测试命令；
一个小规模 PCAP 的端到端运行结果。

第一阶段跑通以后，再继续实现 temporal entity state 和 reliability-aware graph context。

二十二、Phase 2 的 Prompt 也可以提前留着

第一阶段跑通以后，你再给 Codex：

在现有 Flow Encoder 基础上，实现 entity-level temporal normality modeling。

每个 3 秒窗口，以 IP entity 为单位聚合当前正常行为状态。不要加入应用层协议语义。

Entity state 至少包含：

active flow segment count
total packet count
total byte count
normal prototype histogram
可选：mean flow embedding

对同一 entity 构成时间序列：

H_(t-k), ..., H_(t-1) -> predict H_t

第一版使用轻量 GRU predictor，训练仍然只使用正常 PCAP。

定义：

entity_anomaly = distance(H_t, predicted_H_t)

注意：该 entity anomaly score 暂时不要直接加到每条 Flow 的 anomaly score 上。单独输出 flow anomaly 和 entity anomaly，以验证攻击期间是否出现“entity abnormal but benign flow normal”的现象。

二十三、Phase 3 再实现你的核心空间思想

第三阶段再给：

在已有 flow embedding 和 local anomaly score 基础上实现一个轻量 reliability-aware flow graph。

每个 3 秒窗口建立 entity graph：

node = IP entity
edge = active flow segment
edge feature = z_flow

对每条 edge 根据 local anomaly score 计算 reliability，例如：

r_e = exp(-alpha * normalized_local_score)

对 node 的邻接 edge embedding 做 reliability-weighted aggregation，而不是普通平均或无条件 GNN message passing：

C_v = sum(r_e * z_e) / sum(r_e)

使用正常训练图学习：

MLP(C_src, C_dst) -> predicted z_flow

spatial anomaly 定义为：

distance(z_flow, predicted_z_flow)

这样高异常 edge 的信息不会大规模污染其他正常 edge 使用的 context。

最终同时输出：

local flow anomaly
spatial flow anomaly
entity temporal anomaly
final flow anomaly

第一版 final flow anomaly 先采用正常 calibration 后的简单加权组合，不训练监督分类器。