请先完整阅读当前仓库，理解已有的 PCAP 解析、Flow 聚合、特征提取、模型训练、异常检测、数据集标签和评估代码。优先复用能够兼容本设计的实现，不要无意义重写；同时把本方法实现成一个结构清楚的新版本，不要继续给旧实验分支打补丁。不要只给设计或伪代码，请直接实现代码、配置、CLI 和必要的单元测试。

==================== 1. 方法目标与设计思想 ====================

实现一个面向 Video-IoT 的无监督 Flow-level NIDS：

- 输入：PCAP / PCAPNG
- 时间窗口：固定 1s
- 检测对象：1s 单向 Flow Slice
- 训练：仅使用正常流量
- 输出：每个 Flow 的连续 anomaly score
- 测试阶段使用攻击/正常标签仅计算 AUROC、AUPRC、EER，不参与训练、模型选择或异常分数计算

我们关注两个核心问题。

第一，Video-IoT 的 benign traffic 具有明显的 temporal-behavioral heterogeneity：同一种视频通信会因为分辨率、码率等正常运行状态变化产生统计漂移；与此同时，长期持续的视频通信和稀疏的 DNS/NTP/Control 等正常通信在统计特征、持续性和出现频率上差异巨大。若所有正常 Flow 共用一个全局正常分布，正常状态变化和稀疏正常业务都容易获得较高异常分数。

第二，Context 可能发生 contamination：当某个 Camera 发起 Scan/DoS 等攻击时，设备整体的 flow count、peer count、destination diversity 等统计会发生明显变化，但这些异常 Context 不应该无条件传播给同时存在的正常 Video/Control Flow。

因此，本方法采用三层条件正常性：

1. Behavioral Role：决定“这条 Flow 应该和哪类正常通信比较”；
2. Normal Operating Regime：决定“该 Role 当前允许处于哪些正常运行状态”；
3. Responsibility-scoped Context：Context 只有在当前 Flow 对异常 Context 确实具有责任时，才作为该 Flow 的异常证据。

整体目标可写为：

    Normality(flow | behavioral role, operating regime, relevant context)

最终仍然是 Flow-level anomaly detection。

==================== 2. 数据表示 ====================

将 PCAP 按整数秒划分为 [t,t+1) 的固定窗口，在每个窗口内按照单向五元组聚合：

    (src_ip, dst_ip, src_port, dst_port, protocol)

A→B 和 B→A 必须是两个不同 Flow。没有 packet 的时间不生成虚假全 0 Flow。

IP、Port 等身份信息只允许用于：
- Flow 聚合
- Temporal Channel 关联
- Host / Peer / Context 统计

不得作为 Local Model 的数值输入特征。

每个 1s Flow 提取严格固定的 25 维纯统计特征，不使用 TCP flags，不解析 payload：

packet_count, byte_count, active_duration,
packet_len_mean, packet_len_std, packet_len_min, packet_len_max, packet_len_p25, packet_len_p50, packet_len_p75,
iat_mean, iat_std, iat_min, iat_max, iat_p25, iat_p50, iat_p75, iat_cv,
active_bin_count, bin_bytes_mean, bin_bytes_std, bin_bytes_max, bin_packets_mean, bin_packets_std, bin_packets_max。

其中：
- active_duration = last_ts - first_ts；
- packet_count < 2 时所有 IAT 特征为 0；
- std 统一使用 ddof=0；
- 1s 内固定划分为 10 个绝对对齐的 100ms micro-bin，用于最后 7 个 burst statistics；
- 定义固定 FEATURE_NAMES，保证维度和顺序始终为 25。

PCAP 提取必须是 streaming processing，不能一次加载完整大型 PCAP。

==================== 3. 预处理与 Local Flow Encoder ====================

所有 preprocessing 只能在 benign training data 上 fit。

对非负长尾特征先使用 log1p，再使用 RobustScaler（median/IQR）。Scaler 必须支持保存和加载。

使用轻量 PyTorch Flow Encoder：

    x_t ∈ R^25
        ↓
    Linear(25,64)
    LayerNorm
    GELU
    Linear(64,32)
        ↓
    e_t ∈ R^32

这里的 e_t 只表示“这一秒当前 Flow 自己的统计行为”。

==================== 4. 统一 Temporal Memory ====================

检测样本使用完整单向五元组，但时间历史不能完全依赖 source port，因为 DNS/NTP/Control 的临时源端口可能变化。

默认 Temporal Channel Key：

    (src_ip, dst_ip, protocol, dst_port)

并写入配置，方便以后替换。

每个 Channel 保存：
- last_seen
- hidden_state

当新的 Flow Slice 出现时：

    Δt = current_time - last_seen

不存在通信时不产生样本，也不更新 hidden state。

对 log1p(Δt) 做一个轻量 8-D embedding，然后统一使用 GRUCell：

    input_t = concat(e_t, time_embedding(Δt))    # 40-D
    h_t = GRUCell(input_t, h_previous)           # hidden=32

最终：

    z_t = concat(e_t, h_t)                       # 64-D

持续 Video 通常 Δt≈1s；稀疏 NTP 可能 Δt≈60s，但两者必须使用完全相同的 Temporal Model，不为不同业务设计独立网络。

==================== 5. 自监督表示学习 ====================

只使用 benign training traffic。

主要 Neural Objective 只有 Masked Feature Reconstruction。

训练时随机 mask 20% 的 normalized 25-D features，经过 Flow Encoder + Temporal Memory 得到 z_t，然后：

    Decoder:
    Linear(64,64)
    GELU
    Linear(64,25)

重建原始 normalized feature vector。

使用 Huber Loss，并且只对被 mask 的维度计算 reconstruction loss。

V5 不增加 classification loss、attack loss、contrastive loss、role loss 或其他辅助 neural losses。

由于 Video Channel 会产生远多于稀疏 Channel 的 1s 样本，训练 DataLoader 必须实现 Channel-balanced Sampling：

    先近似均匀采样 Temporal Channel
    再从选中的 Channel 内采样 Flow Slice

避免长视频仅因为样本多而支配表示学习。

==================== 6. Behavioral Role Discovery ====================

Encoder 训练完成后，对 benign training flows 生成 z_t∈R^64。

不要直接用全部 Flow Slice 做 Role clustering，因为长视频仍然会贡献大量样本。首先为每个 Temporal Channel 构造一个 Channel Prototype：

    mean(z)      64
    std(z)       64
    median(z)    64
    q25(z)       64
    q75(z)       64
    log1p(slice_count)                         1
    median(log1p(inter-arrival between events)) 1
    std(log1p(inter-arrival between events))    1

总维度 323。

在 Channel Prototype 上训练 BayesianGaussianMixture：

    max_components = 8
    covariance_type = diag

参数写入 YAML。

活跃 mixture components 视为 Latent Behavioral Roles。不要在算法中写死 Video、DNS、NTP、Control 等标签；这些名称只能在后续分析阶段解释。

Role 的出现概率不能成为异常证据，即禁止直接使用：

    -log P(z,role)

因为其中的 -log P(role) 会天然惩罚少数 Role。

==================== 7. Role-conditioned Normal Operating Regimes ====================

Behavioral Role 解决不同正常业务之间的异质性，但同一 Role 内仍可能存在多个正常运行状态。

对于每个 Role r，收集对应 benign Flow 的 z_t，然后拟合 GaussianMixture：

    K ∈ {1,2,3,4}

使用 BIC 自动选择最合适的 K，默认：

    covariance_type = diag
    reg_covar > 0

每个 component 表示该 Role 的一个 latent Normal Operating Regime。

例如 Video-like Role 可能自然形成低码率和高码率两个 Regime，但训练过程中不使用 720p/1080p 标签。

Inference 时，对新的 z 分别计算所有 Role 的条件 likelihood：

    local_raw(r) = -log P(z | role=r)

禁止加入 P(role)。

最终：

    assigned_role = argmin_r local_raw(r)
    local_raw = min_r local_raw(r)

在 assigned_role 内，选择 posterior/likelihood 最大的 mixture component 作为 assigned_regime。

为了让不同 score 便于融合，可使用独立的 benign validation data 对 local_raw 做 empirical-CDF/percentile calibration：

    local_score ∈ [0,1]

注意这里的 calibration 只是把连续分数归一化，不产生部署 threshold。

==================== 8. Role-scoped Context ====================

Context 与 Local Flow Representation 必须完全分离。不要使用 GNN，也不要把其他 Flow embedding 直接聚合进 z_t。

每个 1s 窗口首先为每条 Flow 得到 assigned_role，然后按照：

    (src_host, assigned_role)

分组。

每个 Host×Role Group 提取 10 维 Context：

    flow_count
    total_bytes
    total_packets
    bytes_per_flow_mean
    bytes_per_flow_std
    packets_per_flow_mean
    packets_per_flow_std
    unique_peer_count
    unique_dst_port_count
    new_peer_count

其中 peer=dst_ip；new_peer 表示此前在该 src_host+role 历史中尚未出现的目标。IP/Port identity 本身不能作为 Context Model 的数值特征，只能形成 count statistics。

对于每个 Role 单独训练 Context Model：

    RobustScaler
        +
    GaussianMixture, K∈{1,2,3,4}, BIC selection

Inference：

    context_group_raw =
        -log P(context | assigned_role)

再用 benign validation empirical CDF 得到：

    context_group_score ∈ [0,1]

==================== 9. Responsibility-scoped Context ====================

一个 Group Context 异常不能直接赋给所有 Flow，否则仍会产生 Context Contamination。

因此对每条 Flow i 计算解析式 Responsibility，不增加 Attribution Neural Network，也不做 Leave-One-Out。

只使用容易解释贡献的 6 个 Context 维度：

    flow_count
    total_bytes
    total_packets
    unique_peer_count
    unique_dst_port_count
    new_peer_count

Flow i 的 contribution vector q_i 定义为：
- flow_count：1
- total_bytes：该 Flow byte_count
- total_packets：该 Flow packet_count
- unique_peer_count：若同一 peer 有 n 条 Flow，则每条贡献 1/n
- unique_dst_port_count：同理按 1/n 分摊
- new_peer_count：若 peer 为 new peer，则按同 peer Flow 数量分摊 1/n，否则为 0

根据 benign Role Context 的统计，计算当前 Group 在相同 6 个维度上的 standardized residual R，并只保留异常增加部分：

    R+ = max(R,0)

计算：

    responsibility_i =
        cosine_similarity(q_i, R+)

若任一向量 norm=0，则 responsibility=0，最终 clamp 到 [0,1]。

然后：

    context_flow_score_i =
        context_group_score × responsibility_i

这样 Context 只有在“Group 确实异常”且“当前 Flow 的行为方向与该异常一致”时才增强该 Flow 的异常证据。

==================== 10. 最终连续异常分数 ====================

不实现固定部署 Threshold，也不输出最终 hard-coded Normal/Anomaly 判定。

每条 Flow 输出连续分数：

    final_score =
        max(local_score, context_flow_score)

要求 final_score∈[0,1]。

保留所有中间结果用于分析：

    timestamp_second
    src_ip
    dst_ip
    src_port
    dst_port
    protocol
    channel_id
    assigned_role
    assigned_regime
    local_raw
    local_score
    context_group_raw
    context_group_score
    responsibility
    context_flow_score
    final_score

==================== 11. 攻击数据集评估 ====================

训练阶段只能使用 benign training set。

测试阶段允许加载带有 Normal/Attack Ground Truth 的测试集，但 label 只能进入 evaluation module，绝不能进入 Encoder、Role Discovery、Regime Model、Context Model、Calibration 或任何训练过程。

请优先复用当前仓库已有的 label mapping / attack interval / flow label 机制。如果当前仓库已有标签生成方式，不要自己创建新的规则。

至少实现以下基于连续 anomaly score 的评价指标：

    AUROC
    AUPRC
    EER

EER 定义为 False Positive Rate 与 False Negative Rate 最接近时对应的 error rate，只作为 evaluation metric，不作为部署 threshold。

建议同时保存：
    FPR
    TPR
    Precision
    Recall
    ROC curve data
    PR curve data

但不要因为这些指标在测试集上寻找“最佳 F1 threshold”并用于最终模型。

如果测试集包含多种攻击类型，在有标签支持的情况下，除了 Overall 结果，还输出 per-attack AUROC/AUPRC；不要为不同攻击训练不同 detector。

提供类似：

    python -m viot_nids evaluate \
        --scores scores.parquet \
        --labels labels.xxx \
        --out results/

输出例如：

    metrics.json
    roc.csv
    pr.csv

==================== 12. CLI 与代码结构 ====================

至少支持：

    extract   PCAP -> 1s Flow Features
    train     benign Flow -> model
    score     PCAP/Flow -> continuous anomaly scores
    evaluate  score + ground truth -> AUROC/AUPRC/EER
    inspect   查看模型结构和 learned roles/regimes

如果当前仓库已有 CLI，请集成现有结构。

inspect 至少能够显示：
- 25-D FEATURE_NAMES
- Encoder / GRU 参数
- 实际 Behavioral Role 数量
- 每个 Role 的 Channel 数量
- 每个 Role 选择出的 Regime 数量
- Context Models
- Calibration statistics

==================== 13. 工程实现要求 ====================

推荐：
    Python 3
    PyTorch
    NumPy
    pandas
    scikit-learn
    PyYAML
    dpkt

要求：
- CPU 必须可运行，CUDA 可选
- PCAP streaming extraction
- type hints
- logging
- reproducible random seed
- YAML config
- Encoder / Scaler / BGMM / Regime GMM / Context GMM / Calibration 均能 save/load
- Parquet 优先，CSV 可作为 fallback

关键 Unit Tests 至少覆盖：
- 单向 A→B / B→A 分离
- 1s window boundary
- 固定 25-D feature order
- 100ms micro-bin
- 单 packet Flow 的 IAT=0
- source port 改变后 Temporal Channel 连续
- sparse communication 的 delta_t 正确
- Channel-balanced sampler
- Role prior 不进入 anomaly score
- Host+Role Context grouping
- Responsibility 计算
- model save/load
- AUROC/AUPRC/EER 计算正确

不要创建 synthetic attack dataset，也不要实现 synthetic smoke test。

==================== 14. 不要自行扩展 V5 ====================

当前版本明确不要加入：

    GNN / Dynamic Graph / Line Graph
    Transformer
    Attention
    MoE
    TCP Flags
    Payload / DPI
    IP Embedding
    Port Embedding
    Supervised Attack Classifier
    Attack-specific Detector
    Multiple Auxiliary Neural Losses

如果发现当前设计存在潜在问题，先按要求实现，在最终总结中说明，不要自行换成更复杂的方法。

完成后请报告：
1. 原仓库复用了哪些模块；
2. 新增和修改了哪些文件；
3. 实际实现的模型结构；
4. benign training pipeline；
5. inference pipeline；
6. attack-dataset evaluation pipeline；
7. CLI 命令；
8. Unit Test 结果；
9. 与本设计存在的任何偏差及原因。

请现在开始检查仓库并实际实现代码，不要只返回实现计划。