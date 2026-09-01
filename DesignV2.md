# 无监督 Flow-level NIDS：Design V2

## 1. V2 的目标与不变约束

V2 针对 V1 在 Gotham 视频物联网数据上的两类失败进行修改：

1. 稀疏、短小的正常控制流容易形成稳定误报；
2. 扫描流量会污染摄像头和 NVR 的共享实体状态，连带抬高同一实体上的正常 RTP、RTCP、RTSP 和 ONVIF 流量。

以下约束保持不变：

- 输入仍然是每条 Flow Segment 的 `30 × 6` 原始微时间序列；
- 六个通道仍然只有双向 packet count、byte count 和 mean packet length；
- 不解析 payload，不使用 DNS、NTP、RTSP、ONVIF 等应用层语义；
- IP 和端口只用于双向五元组 Flow 聚合，IP 还可用于实体关系；
- IP、端口值不作为神经网络数值输入；
- **端口是否出现过、端口是否罕见、端口是否变化，都不构成异常规则**；
- Flow Encoder、normal prototypes、entity predictor 和 spatial predictor 仍只用正常训练流量；
- calibration 仍只使用原来的正常 calibration 数据；
- 不改变现有 train/calibration/test 划分，以便与 Kitsune、TFusion 和 V1 公平对比；
- 攻击标签只在最终评估时计算 AUROC、AUPRC、FPR、TPR 等指标。

## 2. 对 `30 × 6` 表示的判断

`30 × 6` 本身不是当前效果差的充分原因。它保留了三秒内双向包数、字节数、长度和时间位置，能够表达扫描突发、方向不平衡、速率变化和短控制交换。V1 的主要问题是模型使用方式，而不是输入维度本身：

- 对 30 个 hidden state 做普通时间均值，会把只占 1～2 个 bin 的事件稀释约 15～30 倍；
- 随机 mask 不保证遮到有包的 bin，稀疏流的大多数训练步骤只在学习空 bin；
- 所有被 mask bin 等权，空 bin 数量远大于非空 bin，优化目标容易被“正确重建零”主导；
- 推理时只抽取一次随机 mask，分数方差较大，也可能完全错过关键活动 bin；
- V1 的逐样本 GroupNorm 会削弱绝对流量强度，而绝对强度对扫描检测有价值。

因此 V2 先保留 `30 × 6`，通过自监督目标和 learned pooling 修复信息利用方式。这样仍满足“不人工提取协议特征”的初衷，也能明确判断问题来自表示容量还是训练机制。只有在完成严格消融后仍明显不足，才考虑增加同样协议无关的原始统计通道。

## 3. 第一阶段 V2：事件感知 Flow Encoder

### 3.1 输入与标准化

对 Flow Segment：

\[
X \in \mathbb{R}^{30\times6}
\]

六个通道不变。标准化参数只在正常训练集拟合，继续采用逐通道 `log1p + standardization`。

是否非空仅由两个 packet-count 通道计算：

\[
o_t=\mathbb{1}[n^{A\rightarrow B}_t+n^{B\rightarrow A}_t>0]
\]

`o_t` 只服务于自监督 mask 和 loss 权重，不是新增的网络流量特征，也不会引入协议知识。

### 3.2 V2 TCN

V2 仍采用轻量 dilated TCN，不引入 Transformer。输入端保留 corruption indicator，用于告诉 decoder 哪些位置由训练过程人为遮挡。该 indicator 不是来自 PCAP 的特征。

V2 删除 TCN block 内的逐样本 GroupNorm，保留训练集标准化后的绝对幅度。TCN 输出：

\[
H=[h_1,\ldots,h_{30}],\quad h_t\in\mathbb{R}^{d_h}
\]

### 3.3 Learned temporal pooling

V1 的固定 mean pooling 改为 attention pooling 与 max pooling 的组合：

\[
a_t=\operatorname{softmax}(w^Th_t)
\]

\[
h_{att}=\sum_t a_th_t,\qquad h_{max}=\max_t h_t
\]

\[
z_{flow}=W[h_{att};h_{max}]+b
\]

这是从六通道原始序列中端到端学习聚合方式，不是人工构造协议特征。attention 表达整体时序贡献，max 分支保留短时强事件，避免稀疏流被均值淹没。

### 3.4 事件感知 masked reconstruction

训练时仍随机 mask micro-bin，但每条含包 Flow Segment 必须至少 mask 一个非空 bin。被 mask 位置的重建损失为：

\[
L_{rec}=\frac{\sum_{t\in M} w_t\lVert \hat X_t-X_t\rVert_2^2}
{\sum_{t\in M} w_t}
\]

其中：

\[
w_t=\begin{cases}
\lambda_{active}, & o_t=1\\
1, & o_t=0
\end{cases}
\]

默认 `λ_active = 8`。它只提高真实事件在自监督目标中的权重，不指定什么流量是攻击，也不使用攻击标签。

### 3.5 确定性的全 bin 重建分数

推理时不再只使用一个随机 mask。对每个 Segment，根据 `segment_id` 生成确定性排列，将 30 个 bin 分成 `R=5` 个互补 mask 集合。五轮结束后，每个 bin 恰好被遮挡并评分一次。

分别计算非空和空 bin 的平均重建误差：

\[
E_{active}=\operatorname{mean}_{t:o_t=1} e_t,
\qquad
E_{empty}=\operatorname{mean}_{t:o_t=0} e_t
\]

\[
E_{rec}=\rho E_{active}+(1-\rho)E_{empty}
\]

默认 `ρ=0.8`。若某一类 bin 不存在，则只使用实际存在的另一类。该分数确定、覆盖完整，并重点观察真实数据事件。

### 3.6 Local anomaly 不变部分

Flow embedding 仍只用正常训练数据拟合 MiniBatchKMeans normal prototypes：

\[
E_{proto}=\min_k\lVert z_{flow}-c_k\rVert_2
\]

normal prototype 的中心仍只由正常训练数据拟合；prototype distance 和 V2 reconstruction error 的分数组合尺度继续只由正常 calibration 数据拟合。local score 仍为二者的无监督加权组合，V2 没有引入攻击监督分类器。

## 4. 第二阶段：Entity 分支的处理

V1 结果说明 entity temporal score 能识别攻击期整体状态变化，但也会把共享实体上的正常业务一起抬高。因此 V2 暂不把 entity score 加入最终 Flow score：

\[
w_{entity}=0
\]

该分支继续独立输出，用于分析攻击实体状态，而不是作为当前 Flow 判定的直接证据。后续若重新设计为“慢更新正常基线 + 快状态偏移”的双时间尺度模型，再单独做消融决定是否进入最终融合。

## 5. 第三阶段 V2：历史可信空间上下文

### 5.1 V1 当前窗口上下文的问题

V1 在同一个三秒窗口内用所有邻接 edge 计算上下文。即使使用

\[
r_e=\exp(-\alpha s_{local,e})
\]

只要大量扫描 edge 的 local score 不够高，它们仍能依靠数量占据聚合结果。正常视频 edge 随后使用已经被攻击污染的 camera/NVR context，于是出现连带误报。

V2 的原则是：**当前窗口的 edge 只能读取此前历史，不能互相构造判定自己的上下文。**

### 5.2 历史状态的键

历史库只维护：

- IP entity 的 embedding EMA；
- 无序 IP pair 的 embedding EMA；
- 无序 IP pair 每窗口 edge 数的历史 EMA；
- 正常训练 embedding 的全局均值，作为冷启动参考。

无序 IP pair 为：

\[
p=\{IP_a,IP_b\}
\]

**pair key 中没有端口。** 端口未见过、端口变化或端口稀有不会产生 anomaly score，也不会降低历史可信度。

### 5.3 只读历史上下文

在窗口 `t` 的读阶段，对 edge `e=(u,v)`，只从窗口 `t-1` 及更早的状态读取：

\[
C_u^{(t)}=\operatorname{blend}(H_u^{(<t)},H_{\{u,v\}}^{(<t)})
\]

\[
C_v^{(t)}=\operatorname{blend}(H_v^{(<t)},H_{\{u,v\}}^{(<t)})
\]

当前实现对已知 pair 使用 entity/pair 历史各 0.5；未知 pair 使用 entity 历史。若 entity 也未见过，则回退到正常训练 embedding 的全局均值。

**未知 entity 或未知 pair 的历史一致性默认是 1，即中性可信。** 因此系统不会把“第一次出现”直接判成攻击。

Spatial MLP 仍学习：

\[
\hat z_e=MLP(C_u^{(t)},C_v^{(t)})
\]

\[
s_{spatial,e}=\lVert z_e-\hat z_e\rVert_2^2
\]

### 5.4 历史可信度

状态更新可信度由两部分组成：

\[
r_{local,e}=\exp(-\alpha s_{local,e})
\]

若 IP pair 已存在历史 centroid，则：

\[
r_{hist,e}=\exp(-\beta\operatorname{MSE}(z_e,H_p))
\]

若 pair 未见过，则 `r_hist = 1`，不因新颖性惩罚。最终：

\[
r_e=\max(r_{min},r_{local,e}r_{hist,e})
\]

这里的可靠性只控制样本写入历史的程度，不把端口或新 pair 直接转换成异常分数。

### 5.5 先按 IP pair 聚合，再更新一次

窗口评分结束后才进入 commit 阶段。同一无序 IP pair 在当前窗口内的所有 edge 先聚合为一个加权平均 embedding：

\[
\bar z_p^{(t)}=\frac{\sum_{e\in p}r_ez_e}{\sum_{e\in p}r_e}
\]

随后 pair 和两个 endpoint entity 各更新一次，而不是每条五元组 edge 更新一次。这样同一 Camera/NVR IP 对之间即使出现 1,000 个不同端口 Flow，也只产生一次状态写入。

更新率还使用历史 pair 每窗口 edge 数进行 multiplicity gate：

\[
g_p=\exp\left[-\gamma\max\left(0,\frac{m_t}{\bar m_{<t}}-1\right)\right]
\]

\[
\eta_p=\eta_0\cdot \operatorname{mean}(r_e)\cdot g_p
\]

该 gate 只回答“当前突发是否应该快速改写历史”，不回答“它是不是攻击”，不会直接加入 anomaly score。统计粒度是 IP pair 的 edge 数，不使用端口身份或端口白名单。

### 5.6 训练、calibration 与测试隔离

1. 只使用正常 train embedding 构造历史 spatial training samples；
2. train 结束后的 normal history bank 随 spatial checkpoint 保存；
3. calibration 从这份 frozen normal reference 初始化；
4. 每个 calibration/test capture 都从同一份 normal reference 独立初始化，避免前一个 PCAP 污染后一个 PCAP；
5. 当前窗口先全部读取旧状态和评分，再统一 commit，杜绝同窗 target leakage；
6. 原 train/calibration/test 数据和切分完全不变。

## 6. V2 最终分数

所有 component scaler 与 threshold 仍只拟合正常 calibration：

\[
s_{final}=0.7s_{local}+0.3s_{spatial}+0.0s_{entity}
\]

部署阈值继续取正常 calibration final score 的 `1 - target_FPR` quantile。当前权重是攻击评估前固定的设计选择，不使用攻击标签调参。

## 7. 需要验证的假设与消融

V2 不能只报告 final score，应至少保留下列消融：

1. V1 Flow Encoder + V1 current-window spatial；
2. 仅替换 V2 event-aware Flow Encoder；
3. V2 Flow Encoder + historical spatial；
4. local only；
5. reconstruction only 与 prototype only；
6. historical spatial 去掉 multiplicity gate；
7. entity 独立报告但不融合。

重点分流量与阶段核查：

- 扫描攻击的 local AUROC 是否不再反向；
- NTP 等稀疏控制流的稳定 FP 是否下降；
- OS Scan 攻击期正常 RTP、RTCP、RTSP、ONVIF 的 FP 是否下降；
- 攻击前、攻击中、攻击后的 FPR；
- Camera-01/NVR 正常流在攻击窗口内的 score 分布；
- AUROC、AUPRC、FPR、TPR、TPR@FPR≤1%、TPR@FPR≤0.1%；
- flow model 与完整 pipeline 的吞吐和峰值内存。

由于测试集中攻击占比极高，AUPRC 不能作为主要结论。优先看 AUROC、正常流 FPR、严格低 FPR 下的 TPR，以及按协议/阶段拆解的误报机制。

## 8. 实现参数与兼容性

默认 V2 参数：

```yaml
flow_model:
  architecture: v2
  nonempty_loss_weight: 8.0
  score_mask_rounds: 5
  active_error_weight: 0.8

spatial_model:
  context_mode: historical
  history_beta: 1.0
  state_update_rate: 0.10
  multiplicity_gamma: 0.05

scoring:
  final_weights: {local: 0.70, spatial: 0.30, entity: 0.00}
```

旧 checkpoint 若没有 `architecture` 或 `context_mode` 字段，分别自动按 `v1` 和 `current_window` 加载。因此已有 V1 结果仍可复现；V2 训练会生成带显式版本信息的新 checkpoint。

## 9. 当前结论边界

V2 的实现解决的是已由反事实分析支持的机制性缺陷：稀疏事件在 Flow objective 中权重不足，以及当前窗口共享状态被大量同 IP pair edge 污染。它不承诺在尚未重新训练前必然提高 Gotham 指标。最终是否有效，必须在完全相同的数据划分、标签映射和评估脚本下重新训练，并与 V1、Kitsune、TFusion 做上述分机制对比。
