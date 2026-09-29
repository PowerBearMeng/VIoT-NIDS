请先完整阅读当前仓库 https://github.com/PowerBearMeng/VIoT-NIDS.git，重点理解现有 V5 的 DesignV5.md、viot_nids/、configs/v5.yaml，以及当前 PCAP streaming 解析、1s directional Flow 聚合、25 维特征、FlowMemory、Role/Regime、Context、Calibration、Flow label 和 evaluation 的实现。

请在不破坏 V1-V5 的前提下，新建一个独立 V6 实现。优先复用 V5 中正确且兼容的代码，不要无意义重写，也不要继续给 V5 打补丁。V6 建议放在独立的 viot_nids_v6/ 包中，并增加 configs/v6.yaml 和 DesignV6.md。

==================== 1. V6 核心目标 ====================

V6 仍然是 Video-IoT 场景下的 benign-only、unsupervised、Flow-level NIDS：

- 输入：PCAP / PCAPNG
- 检测对象：1 秒单向五元组 Flow Slice
- 训练：只能使用正常流量
- 输出：每条 Flow 的连续 anomaly score
- 攻击标签只能用于最终 AUROC / AUPRC / EER evaluation
- 不使用 payload
- 不使用 TCP flags
- IP、Port 不作为 Local Neural Model 的数值输入

V6 的统一正常性定义为：

    Normality_i^t = Normality(x_i^t | R_i, H_i^t, C_i^t)

其中：

R_i：Behavioral Role，表示当前 Flow 属于哪类正常通信行为；
H_i^t：Temporal State，表示该行为自身历史和当前正常运行状态；
C_i^t：Context，表示与当前 Flow 相关的共享环境和关系/群体行为。

V6 不再把 Operating Regime 作为顶层独立概念。Regime 可以继续作为 H_i^t 内部的 latent normal state 使用。

V6 的核心不是“偏离一个全局正常分布有多远”，而是：

    在 Role、Temporal State 和相关 Context 都被考虑以后，
    当前 Flow 还有多少行为无法由正常因素解释。

不要实现任何针对 SYN DoS、OS Scan 或具体攻击类型的规则。
不要硬编码 Video、DNS、Control、8080、8554 等业务标签或端口先验。

==================== 2. 保留 V5 的数据表示 ====================

继续使用 V5 的 1 秒 directional Flow Slice：

    (src_ip, dst_ip, src_port, dst_port, protocol)

A→B 和 B→A 分开。

继续使用现有 V5 的固定 25 维纯统计特征、log1p + RobustScaler 和 streaming PCAP processing。

IP、Port 只允许用于：

- Flow 聚合
- Temporal Channel 关联
- peer/service relation history
- Context grouping 和 count/diversity statistics

禁止将 IP address 或具体 port number embedding 后输入神经模型。

==================== 3. Shared Flow Encoder 与 Temporal State ====================

优先复用 V5：

    x_t[25]
       ↓
    Flow Encoder
       ↓
    e_t[32]
       ↓
    time-gap embedding + GRUCell
       ↓
    h_t[32]
       ↓
    z_t = concat(e_t, h_t)[64]

Temporal Channel 默认仍使用：

    (src_ip, dst_ip, protocol, dst_port)

保持可配置。

Encoder 仍然只使用 benign data，通过 Masked Feature Reconstruction 训练。
继续保留 Channel-balanced Sampling，避免持续 Video Flow 因样本数量巨大而支配训练。

H_i^t 在 V6 中表示整个 temporal state，包括：

- GRU hidden history
- time gap
- 当前 channel 的历史行为
- role 内部 latent normal state / regime

因此不要在顶层算法接口中额外增加 O_i。

==================== 4. Behavioral Role R ====================

继续保留 V5 的 latent Behavioral Role 思想。

Role Discovery 必须：

- 只使用 benign training data；
- 不使用业务标签；
- 不使用 Role frequency 作为 anomaly evidence；
- 避免大量 Video slices 支配 role discovery。

可以优先复用 V5 的 Channel Prototype + BayesianGaussianMixture。

Role 的作用只是：

    “当前 Flow 应该和哪一类正常行为进行比较？”

禁止使用：

    -log P(role)

作为异常分数的一部分。

==================== 5. Role-conditioned Temporal Normal State ====================

V5 的 Role-conditioned GMM / latent regime 可以继续存在，但在 V6 中把它视为 H_i^t 内部 normal state。

对于每个 Role r，在 benign training embedding z 上学习多个 latent normal states：

    P(z | R=r, state=k)

仍然可以使用 diagonal GMM + BIC 选择 component 数量。

Inference 时选择最匹配的 normal component，并取得该 component：

    mean μ
    diagonal std σ

计算标准化 local residual：

    d_i = (z_i - μ_i) / σ_i

不要直接把 GMM 的 joint likelihood 当作 V6 最终 Local Score。

==================== 6. Shared Benign Variation ====================

V6 新增一个关键机制：

同一个 Role 中，多个长期存在的正常 Channel 可能由于分辨率、码率、网络状态等因素同时发生相似偏移。

因此在每个 second、每个 role 内，从“established channels”估计 shared benign shift。

established channel 至少满足：

- 不是当前第一次出现；
- 有足够 benign/history observations；
- 参数 min_history_count 写入 YAML；
- 不允许刚刚大量产生的新 Flow/channel 直接主导 shared shift。

对于 established channels 的 standardized residual d_j：

    delta_r_t = robust_aggregate({d_j})

第一版使用逐维 median 或 configurable trimmed mean，不要使用普通 mean。

为了避免目标自身影响自己的 reference，样本数允许时使用 leave-one-out；
样本过少时可以使用 role-level aggregate，并输出 insufficient_context 标记。

得到 corrected residual：

    u_i = d_i - delta_r_t

Local raw anomaly 应基于 u_i，而不是原始 d_i。

例如可以使用：

    local_raw_i = mean(Huber(u_i))

或等价稳定的 robust squared norm。

目标是：

- 单个 Flow 独自偏离 → Local anomaly 高；
- 同一 Role 的多个 established benign flows 同时产生相似 shift → shared component 被解释掉。

注意：

shared shift 只能解释“共同变化”，不能让大量新出现的 DoS/Scan Flow 把攻击解释成正常 drift。

==================== 7. Relational Context C ====================

删除 V5 当前的 per-flow contribution cosine responsibility：

    responsibility_i = cosine(q_i, R+)

以及：

    context_flow_score = group_score * responsibility

V6 不允许继续使用这套公式。

原因：
- q_i 是 raw contribution，而 R+ 是 standardized residual，尺度不一致；
- collective anomaly 不能要求单条 Flow 独自解释整体异常；
- 这种 responsibility 对大量小 Flow 形成的 DoS/Scan 不合理。

V6 改为 relation-aware cohort context。

首先为每条 Flow 从历史中构造 relational features，不使用具体 IP/Port identity 作为数值：

建议至少实现：

    peer_seen_count
    service_seen_count
    time_since_peer_seen
    time_since_service_seen
    is_new_peer
    is_new_service
    peer_recurrence_rate
    service_recurrence_rate

其中 service 可以由：

    (dst_ip, protocol, dst_port)

或可配置的 relation key 定义，但具体 dst_port 数值不能进入模型。

所有 history 更新必须 causal：
当前 Flow 的特征只能由它之前的历史计算，然后再更新 history。

==================== 8. Latent Relation State ====================

不要硬编码“stable service / transient service”。

在 benign training data 中，根据 relational feature 自动学习 latent relation state Q。

可以按 Role 单独训练小型 GaussianMixture：

    Q_i = relation_state(relational_features_i | R_i)

component 数量使用 BIC 自动选择，例如 K∈{1,2,3,4}。

Relation State 只是 C_i 内部变量，不需要成为顶层 Normality 公式中的新符号。

==================== 9. Behavior-Relation Cohort ====================

每个 second，将 Flow 按：

    (src_host, assigned_role, relation_state)

组成 Context Cohort。

注意：

不要按具体 dst_port 分组，否则 OS Scan 会被拆成大量 singleton；
不要按具体 dst_ip 直接拆组；
具体 peer/service identity 只用于计算 novelty / recurrence / diversity。

对每个 Cohort 提取 collective statistics，例如：

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
    new_service_count
    peer_diversity_ratio
    port_diversity_ratio

ratio 分母注意加 epsilon。

Context model 必须 benign-only，并至少条件于：

    (role, relation_state)

即学习：

    P(context | role, relation_state)

不要再仅仅学习：

    P(context | role)

因为 V5 diagnostic 已经表明同一个 Role 内不同通信关系可能有明显不同的正常 Context。

使用 RobustScaler + diagonal GMM/BIC 即可，不需要新建复杂 GNN。

==================== 10. 可选的 Causal Context Memory ====================

为了避免只能检测一秒内非常密集的 collective attack，为每个：

    (src_host, role, relation_state)

维护一个 causal EWMA context state。

例如：

    m_t = alpha * current_context + (1-alpha) * m_{t-1}

alpha 写入 YAML。

Context Model 输入可以是：

    concat(current_context, ewma_context)

或者分别建模后融合。

不要创建未来信息泄漏。

这样 1 秒 Flow 仍然是最终检测对象，但低速 Scan 可以通过跨时间累积的 relation/context behavior 被观察到。

==================== 11. Context Anomaly ====================

对每个 Cohort 得到：

    context_raw

并将该 Context evidence 赋给属于此 Cohort 的 Flow：

    context_raw_i = context_raw(cohort(i))

这里不再额外乘 heuristic responsibility。

Attribution 是通过：

    role + relation_state + cohort construction

在形成 Context 时完成，而不是检测后再把一个 Host-level anomaly score 分摊给所有 Flow。

保留所有 context 中间字段，方便检查 contamination。

==================== 12. Local / Context Evidence Calibration ====================

不要使用：

    final = max(local, context)

也不要使用：

    context * responsibility

分别利用独立 benign validation data 对 local_raw 和 context_raw 做 empirical calibration。

定义：

    F_local(local_raw)
    F_context(context_raw)

得到 upper-tail probability：

    p_local   = max(1 - F_local(local_raw), eps)
    p_context = max(1 - F_context(context_raw), eps)

异常 evidence：

    E_local   = -log(p_local)
    E_context = -log(p_context)

融合：

    E_joint = E_local + E_context

由于 Local 与 Context 不独立，不使用理论 Fisher chi-square 分布。

再使用 benign validation 上 E_joint 的 empirical CDF：

    final_score = F_joint_benign(E_joint)

最终：

    final_score ∈ [0,1]

如果某条 Flow 缺少有效 Context，则：

    E_context = 0

并输出 context_available=False。

所有 calibration 只能使用 benign validation data。

==================== 13. 训练数据划分 ====================

避免 validation history 泄漏。

benign data 按时间划分：

    train
    calibration/validation

所有以下组件只能在 train 上 fit：

- feature scaler
- Flow Encoder / GRU
- Behavioral Role
- latent temporal state GMM
- relation state model
- context model

validation 只允许：

- empirical local calibration
- empirical context calibration
- joint evidence calibration

攻击数据完全不能进入上述过程。

==================== 14. 输出 ====================

每条 Flow 至少输出：

    row_id
    timestamp_second
    src_ip
    dst_ip
    src_port
    dst_port
    protocol
    channel_id

    assigned_role
    temporal_state_id

    local_raw
    local_score

    shared_shift_norm
    corrected_residual_norm
    established_channel
    shared_context_available

    relation_state
    peer_seen_count
    service_seen_count
    is_new_peer
    is_new_service

    cohort_id
    context_raw
    context_score
    context_available

    local_evidence
    context_evidence
    joint_evidence
    final_score

尽量保留 debugging 所需的中间字段，但不要输出攻击标签到 scoring pipeline。

==================== 15. Evaluation ====================

复用现有 repository 的 packet-to-Flow label 机制。

支持：

    AUROC
    AUPRC
    EER

同时输出 ROC / PR curve 数据。

如果存在 attack type label，可以输出 per-attack metrics，但不得为不同攻击训练不同模型。

另外增加 diagnostics：

1. local-only AUROC/AUPRC
2. context-only AUROC/AUPRC
3. fused AUROC/AUPRC
4. normal Flow 在 attack period 内的 score distribution
5. normal Flow 与 attack Flow 共处 Cohort 时的 false-positive statistics

这些 diagnostics 只用于 evaluation，不进入训练。

==================== 16. CLI ====================

实现与 V5 类似的接口，例如：

python -m viot_nids_v6 extract ...
python -m viot_nids_v6 train ...
python -m viot_nids_v6 inspect ...
python -m viot_nids_v6 score ...
python -m viot_nids_v6 evaluate ...
python -m viot_nids_v6 benchmark ...

优先复用 V5 已有 parser / label / evaluation 代码。

==================== 17. 配置 ====================

新增 configs/v6.yaml，至少包含：

seed
validation_fraction
channel_fields

neural:
    epochs
    batch_size
    sequence_length
    mask_ratio
    learning_rate

roles:
    max_components

temporal_state:
    max_components
    reg_covar

shared_shift:
    min_history_count
    estimator
    trim_ratio

relation:
    max_components
    reg_covar
    history_timeout

context:
    max_components
    reg_covar
    ewma_alpha

calibration:
    epsilon

所有参数必须有合理默认值，不允许为了某一个攻击 capture 手工调参。

==================== 18. 实现原则 ====================

1. V1-V5 现有行为不得被修改。
2. 优先复用 V5 中已经正确实现的数据解析、25D feature、FlowMemory 和 evaluation。
3. 不使用 attack label、attack interval、attack type 参与训练、聚类、参数选择或 calibration。
4. 不写攻击特定规则。
5. 不根据当前已知 SYN DoS / OS Scan 结果硬编码阈值、端口、IP 或行为。
6. 所有 history/context 必须 causal，禁止未来信息。
7. 所有 preprocessing 只能 fit benign train。
8. 保持代码模块化，至少拆分：
   features / neural / roles / temporal_normality / relation / context / calibration / pipeline / evaluation。
9. 增加必要的 unit tests，重点检查：
   - history 无未来泄漏；
   - new peer/service 判断正确；
   - shared shift 不使用新出现 channel；
   - relation cohort grouping 正确；
   - empirical calibration 范围正确；
   - attack labels 不进入 train/score。
10. 完成实现后更新 DesignV6.md 和 viot_nids_v6/README.md，明确说明 V6 与 V5 的差异。

==================== 19. 最终设计必须保持的一条主线 ====================

V6 不是三个独立 detector 的拼接，而是 Factorized Conditional Normality：
    R：谁应该和谁比较；
    H：目标 Flow 的正常行为如何随时间变化；
    C：当前外部关系和 collective behavior 如何解释观测变化。
Local branch 判断：
    在 R 和 H 条件下，这条 Flow 自身是否存在无法解释的偏离？
Context branch判断：
    在 R 和 relation-aware C 条件下，这条 Flow 所属的行为群体是否出现异常 collective structure？
最后通过 benign-calibrated evidence accumulation 得到统一 Flow-level anomaly score。