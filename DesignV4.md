# Design V4.0：连续神经行为强度 Flow NIDS

## 1. 目标

V4 解决 V3 暴露出的三个结构性问题：

1. directional Flow 下原双向 `30×6` 有三个恒零通道，且没有 IAT；
2. `KMeans → hard mode_id → mode count z-score` 丢失连续 embedding 信息；
3. empirical upper-tail 与 `max` fusion 在测试分数超出有限 calibration 支持时产生大规模并列和饱和。

V4 不是“发现某个 IP 数量超过固定阈值就判异常”。IP 只用于确定当前 3 秒窗口中哪些 Flow 属于同一 pair/entity scope；模型既不学习 IP 数值，也不保存逐 IP 异常阈值。正常强度的期望与不确定性由目标 Flow 的连续 embedding 条件化预测。

## 2. 不变约束

- 3 秒 epoch-aligned directional 五元组与 TFusion 对齐；
- 不解析 payload，不使用应用层协议；
- 不增加 NetFlow feature；
- IP/port 不进入神经网络数值输入；
- port 不用于 scope key、novelty、rarity 或 anomaly rule；
- Flow model 与 Context model 只训练正常 traffic；
- calibration 只使用正常 calibration traffic；
- 攻击标签只在最终 evaluation 使用；
- Gotham train/calibration/test 划分保持不变；
- V1/V2/V3 代码、checkpoint 与配置路径继续保留。

## 3. Directional IAT `30×6`

每条 directional Flow 仍按 3 秒划分为 30 个 100ms micro-bin。每个 bin 使用：

```text
packet_count
byte_count
mean_packet_length
std_packet_length
mean_iat_ms
std_iat_ms
```

对同一 directional 五元组的连续包：

```text
IAT_i = timestamp_i - timestamp_(i-1)
```

IAT 跨 micro-bin 和 3 秒窗口连续计算，并归入第 `i` 个包所在的 bin。一个五元组在 capture 中首次出现时不伪造 IAT。训练侧 `FeatureStandardizer` 对非负输入统一使用 train-only `log1p` 后标准化。

这使六个通道都具有明确含义，不再保留无效的反向恒零通道。

## 4. Flow 表示与 Local anomaly

V4 保留 V2 Flow Autoencoder：

```text
30×6 directional IAT input
  → event-aware deterministic masking
  → TCN
  → attention pooling + max pooling
  → z_flow ∈ R^32
  → masked reconstruction error
```

Local raw score 仍为：

```text
s_local = reconstruction_error
```

V4 最终路径不拟合 KMeans，不使用 prototype distance，也不产生 `mode_id`。

## 5. 连续神经行为表示

Context model 使用可学习 soft behavior gate：

```text
q_i = softmax(g_theta(z_i) / temperature)
```

`q_i` 是连续概率向量，不执行 `argmax`，不转成离散行为编号。训练使用 assignment balance 与 entropy regularization，避免全部 Flow 塌缩到同一 latent channel，同时鼓励可区分的连续行为权重。

这些 latent channel 只是用于线性时间的连续相似度特征映射，不导出或使用 hard mode。

## 6. Target-specific neural intensity

对一个 scope `G`（当前 capture、当前 3 秒窗口中的无序 IP pair 或 IP entity），目标 Flow `i` 的 leave-one-out soft behavior mass 为：

```text
m_i(G) = q_i^T (Σ_{j∈G} q_j - q_i)
y_i(G) = log1p(max(0, m_i(G)))
```

这个量具有三个性质：

1. 使用目标 Flow 自身的连续行为权重查询 scope；
2. 与目标行为不相似的 entity 流量权重较低，不会把整个 entity burst 无差别复制给所有 Flow；
3. 相似 Flow 大量出现时 mass 增大，保留 Scan/DoS 所必需的 multiplicity sensitivity。

计算通过每个 scope 的 soft assignment sum 实现，复杂度为 `O(NK)`，不构造 Flow 两两 attention 矩阵。

## 7. Normal-only 条件强度学习

神经模型从目标 Flow embedding 预测该行为在正常 pair/entity scope 中应具有的 log mass 分布：

```text
(μ_pair(z_i), log σ_pair(z_i)) = f_pair(z_i)
(μ_entity(z_i), log σ_entity(z_i)) = f_entity(z_i)
```

只在正常训练 Flow 上最小化 Gaussian NLL：

```text
NLL(y, μ, σ) = 0.5 ((y-μ)/σ)^2 + log σ
```

推理时只把超过正常条件期望的部分作为强度异常，正常业务暂时减少不视为 NIDS 异常：

```text
E(y, μ, σ) = 0.5 ReLU((y-μ)/σ)^2
```

Pair、源 entity、目的 entity 的 energy 使用 normalized log-sum-exp 连续聚合。这里没有“IP 出现 N 次”的固定规则；同样的 mass 对不同目标 embedding 可以有不同的正常期望和不确定性。

任何部署检测器最终都需要阈值，但 V4 的阈值只作用于最终连续 anomaly score，并由正常 calibration 确定；它不等价于手写 IP count threshold。AUROC 和 EER 评测本身不依赖固定部署阈值。

## 8. 不饱和连续融合

正常 calibration 分别对 raw reconstruction 与 neural context energy 拟合 Q05/Q95 连续尺度：

```text
a = max(0, (score - Q05) / (Q95 - Q05))
```

尺度变换不做 upper clipping，超出 calibration 支持的异常仍保留相对大小。最终使用 smooth OR：

```text
final = T · logmeanexp([a_local/T, a_context/T])
```

它在两个分量均为 0 时等于 0，接近较大分量但保持连续，不会像 empirical rank tail 一样把所有超界攻击映射成同一个最大证据。

## 9. 实际训练路径

```text
prepare directional IAT Flow
  → train V2 Flow Autoencoder on normal train
  → extract embeddings/reconstruction
  → train NeuralIntensityContext on normal train
  → early stop on normal calibration
  → fit continuous calibration scales and deployment threshold
```

V4 不执行：

```text
fit_prototypes
train_entity GRU
train_spatial MLP
fit behavior-composition reference
```

主要 artifact：

```text
flow_model.pt
train_embeddings.npz
neural_context_model.pt
calibration.json
```

## 10. 配置与运行

完整训练：

```bash
cd /home/mfh/Desktop/test/myModel
/home/mfh/miniconda3/envs/wxy/bin/python run_pipeline.py \
  --config configs/gotham_v4_train.yaml \
  --mode prepare
/home/mfh/miniconda3/envs/wxy/bin/python run_pipeline.py \
  --config configs/gotham_v4_train.yaml \
  --mode train \
  --device cuda
```

九数据集评测：

```bash
/home/mfh/miniconda3/envs/wxy/bin/python evaluate_gotham_manifest.py \
  --config configs/gotham_v4_train.yaml \
  --device cuda
```

CPU smoke：

```bash
/home/mfh/miniconda3/envs/wxy/bin/python tests/generate_smoke_pcaps.py
/home/mfh/miniconda3/envs/wxy/bin/python run_pipeline.py \
  --config configs/smoke_v4.yaml \
  --mode all \
  --device cpu
```

V4 score CSV 不包含 `mode_id`、prototype distance 或逐 mode count；保留 IP/port 仅用于评分完成后的机制审计与业务名称映射。

