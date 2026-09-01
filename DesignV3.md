# Design V3.0：基于正常行为组成的 Flow NIDS

## 1. 本轮目标

V3.0 不修改 Flow 输入和 V2 编码器，而是修复三个问题：prototype 的职责混淆、spatial embedding prediction 的实体连带误报，以及不同分数组件直接线性相加时量纲不一致。

V3.0 的默认检测路径为：

```text
30×6 micro-bin Flow
  └─ V2 event-aware TCN + attention/max pooling
       ├─ deterministic masked reconstruction → local score
       └─ normal-train prototype assignment → mode_id

同一 3 秒窗口内的 mode_id
  ├─ 无序 IP pair × mode count
  └─ IP entity × mode count
       └─ frozen normal-train reference → context score

normal calibration empirical upper tail
  ├─ reconstruction → A_local
  └─ context deviation → A_context
       └─ final = max(A_local, A_context)
```

## 2. 不变约束

- Flow 仍为 `30 × 6` micro-bin representation；每个 micro-bin 仍只有方向性 packet count、byte count 和 mean packet size。
- 当前默认与 TFusion 基线对齐：使用绝对时间对齐的 3 秒半开窗口和有向五元组；反向流量形成独立 Flow。为兼容旧 V1/V2 数据，`bidirectional + capture` 仍可通过配置复现。
- 保留 V2 event-aware masking、TCN、attention + max pooling 和 deterministic reconstruction scoring。
- 不加入 NetFlow feature，不解析 payload，不使用应用层协议。
- IP 和 port 不作为神经网络数值输入。
- port 不参与 prototype、reference key、novelty、rarity、deviation 或 anomaly rule。
- Flow model、prototype、entity predictor、behavior reference 均只使用正常 train traffic。
- calibration 只使用正常 calibration traffic；攻击标签只在最终 evaluation 使用。
- train/calibration/test 划分和 Gotham manifest 均保持不变。

评测 CSV 保留 IP/port，是为了在模型完成评分后核查 Camera-01/NVR 以及 RTP、RTCP、RTSP、ONVIF 的误报机制。它们不回流到模型。

## 3. Local anomaly 与 prototype

V3 默认：

```text
local_score = reconstruction_score
```

`MiniBatchKMeans` 仅在 normal train embedding 上拟合。每条 Flow embedding 只通过最近中心获得正常行为模式：

```text
mode_id = argmin_k ||z_flow - prototype_k||
```

prototype distance 不进入 V3 默认 local score。为复现实验，artifact 同时保留以下三个数组：

- `reconstruction_score`；
- `prototype_score`；
- `combined_local_score`，即 V2 reconstruction + prototype 分数。

旧 prototype artifact 若缺少新数组，推理代码会从旧的 distance/reconstruction scale 和 local weights 恢复它们，因此 V1/V2 checkpoint 与 evaluation 仍可使用。

## 4. Historical Behavior Composition

### 4.1 当前窗口 profile

对每个 capture 内的 3 秒 window，以 `mode_id` 计数。

无序 IP pair profile：

```text
pair_mode_count[{u,v}][k]
  = 当前窗口 pair {u,v} 中 mode k 的 flow segment 数
```

Entity profile：

```text
entity_mode_count[v][k]
  = 当前窗口与 entity v 相连且属于 mode k 的 flow segment 数
```

同一 Flow 的两个端点相同时只对该 entity 计数一次。所有 key 均不包含端口。

### 4.2 Frozen normal reference

对完整 normal train window 集中每个已知 scope 的 `log1p(mode_count)` 拟合逐 mode mean/std。一个 scope 或某个 mode 在当前正常训练窗口未出现时，其 observation 为 0；从未在 normal train 出现的 scope 不建立 reference，并在推理时保持中性。

calibration 和 test 均加载同一份冻结 reference，不做 EMA 更新，避免攻击污染历史状态。

### 4.3 Positive deviation

```text
d(scope,k) = max(0,
  (log1p(current_count(scope,k)) - normal_mean(scope,k))
  / (normal_std(scope,k) + eps))
```

未在正常训练出现的 pair 默认 deviation 为 0，不因首次出现而异常。V3.0 也将未知 entity 设为中性，避免把身份 novelty 偷渡成异常规则。

对当前 Flow `e=(u,v)`，仅查询其自己的 `mode_id=k`：

```text
pair_context(e)   = d({u,v}, k)
entity_context(e) = max(d(u,k), d(v,k))
context_score(e)  = max(pair_context(e), entity_context(e))
```

因此，某个 entity 的扫描 mode 数量异常不会自动抬高同一实体正常视频 mode 的分数。这是 V3 相对 V2 spatial/entity 状态传播的关键隔离机制。

## 5. Score calibration 与 fusion

分别在正常 calibration score 上构建 empirical upper-tail reference：

```text
p(s) = P(normal_score >= s)
A(s) = -log(p(s) + eps)
```

实现使用 plus-one smoothing，有限 calibration sample 之外的分数也不会产生零概率。默认最终分数：

```text
A_local   = tail_evidence(reconstruction_score)
A_context = tail_evidence(context_score)
final_flow_score = max(A_local, A_context)
```

最终 deployment threshold 同样只在正常 calibration 上按目标 FPR 拟合。Entity temporal predictor 继续独立训练和输出，但权重固定为 0，不进入 flow final score。

V2 的 `0.7 × combined_local + 0.3 × legacy_spatial` 仍保留为消融，不是 V3 默认值。

## 6. 统一消融

`evaluate_flow.py` 在同一次 inference、同一 test label vector 和同一 evaluation function 上输出：

1. `v2_reconstruction_only`；
2. `v2_reconstruction_prototype`；
3. `v3_pair_mode_context_only`；
4. `v3_entity_mode_context_only`；
5. `v3_pair_entity_context`；
6. `local_only`；
7. `context_only`；
8. `old_weighted_fusion`；
9. `v3_normal_tail_max`。

前五项保留组件原始/既有标度，用正常 calibration 阈值评测；第六、七、九项使用 empirical upper-tail evidence；第八项加载保留的旧 spatial predictor。九项结果写入 `metrics.json -> ablations`。

## 7. OS Scan 机制审计

`analyze_v3_os_scan.py` 从 V3 的 per-flow score CSV 和 OS Scan `label_summary.json` 生成 JSON 与中文 Markdown，包含：

- attack Flow reconstruction error/score distribution；
- attack Flow mode distribution；
- 主要 attack mode 在攻击前、攻击期、攻击后的 pair/Camera-01/NVR entity count；
- 对应 pair-mode 和 entity-mode deviation；
- 攻击期正常 RTP、RTCP、RTSP、ONVIF 的 local/context/final 分布与 FP；
- Camera-01/NVR 正常业务三阶段 FP/FPR；
- 两个设计假设的定量检查。

该分析中的端口只负责最终业务名称映射，不构成检测特征或规则。

## 8. 配置与运行

V3 Gotham 训练：

```bash
python run_pipeline.py --config configs/gotham_v3_train.yaml --mode prepare
python run_pipeline.py --config configs/gotham_v3_train.yaml --mode train --device cuda
```

用冻结训练 artifact 逐个评测 manifest 中九个数据集：

```bash
python evaluate_gotham_manifest.py --config configs/gotham_v3_train.yaml --device cuda
```

最小 V3 配置为：

```yaml
flow_model:
  architecture: v2
  anomaly_score: reconstruction_only

context_model:
  mode: behavior_composition
  pair_enabled: true
  entity_enabled: true
  history: frozen_train_reference
  use_log_count: true
  positive_deviation_only: true

scoring:
  fusion: normal_tail_max
  entity_weight: 0.0
```

V2 配置未删除；没有 `context_model.mode=behavior_composition` 的配置仍走旧 calibration/evaluation 分支。
