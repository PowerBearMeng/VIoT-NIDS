# Design V4.5：Target-Conditioned Sparse Attention Context

## 1. 版本定位

V4.5 保留 V4 的 Local branch，不修改 Packet、Flow、`30×6`、TCN、
32 维 Flow embedding 或 deterministic reconstruction score。Context branch
完全替换为带 NULL 的单头 target-conditioned Sparsemax attention。

V4.5 最终路径不包含 V4 的 `behavior_gate`、soft assignment `q_i`、
latent channel、soft-sum mass、assignment balance 或 entropy regularization，
也不兼容旧 V4 Context checkpoint。

## 2. 保持不变的 Local branch

```text
Packet
  → 3 秒 epoch-aligned directional 五元组 Flow
  → 30 个 100 ms micro-bin
  → [packet count, byte count, packet-length mean/std, IAT mean/std]
  → train-only log1p standardization
  → event-aware masked TCN autoencoder
  → reconstruction error + z_i ∈ R^32
```

Local raw anomaly 仍为 deterministic masked reconstruction error。训练仅使用
normal train，early stopping 仅使用 normal calibration。

## 3. Port-free scope

每个 capture 的每个 3 秒窗口临时建立：

- Pair scope：相同无序 IP pair `{u,v}` 的所有 directional Flow segment；
- Entity scope：与同一 endpoint IP 相连的所有 directional Flow segment。

端口不参与 scope key。IP 只用于当前窗口分组，不进入神经网络，不保存逐
IP 正常参考，也不存在 unknown-IP neutral 分支。

## 4. Target-conditioned single-head attention

Pair 和 Entity 使用各自的一组共享参数。对目标 Flow `i` 和 scope 内邻居
`j`：

```text
Q_i = W_q(z_i)
K_j = W_k(z_j)
s_ij = Q_i K_j^T / sqrt(d)
```

目标自身从候选集中排除。每个 scope 另由目标 embedding 预测一个 NULL
logit：

```text
s_null_pair(i)   = f_null_pair(z_i)
s_null_entity(i) = f_null_entity(z_i)
```

在 `[s_null, s_i1, ..., s_in]` 上执行 Sparsemax：

```text
[alpha_null, alpha_i1, ..., alpha_in] = sparsemax(logits)
```

Sparsemax 将 logits 投影到概率单纯形，允许低相关邻居获得严格为零的权重。
单 Flow/no-neighbor 时唯一有效候选是 NULL，因此 `alpha_null=1`。

## 5. Attention diagnostics 与 observed intensity

对 Pair、Entity-A、Entity-B 分别计算：

```text
null_weight         = alpha_null
support_size        = count(alpha_neighbor > 0)
relevance           = 1 - alpha_null
effective_neighbors = relevance^2 / (sum(alpha_neighbor^2) + eps)
observed_intensity  = log1p(relevance * effective_neighbors)
```

`effective_neighbors` 是对注意力集中度的连续等效邻居数；大量低相关邻居若
被 Sparsemax 清零，就不会仅凭数量抬高目标 Flow 的 intensity。

实现按 scope 精确计算 Sparsemax，并按 target chunk 限制临时矩阵内存。
时间复杂度仍由 scope 内两两 QK 决定，最坏为 `O(sum_g |g|^2 d)`。

## 6. Normal intensity distribution

轻量共享 head 从目标 embedding 预测正常 intensity 的条件分布：

```text
(mu_pair(z_i), log_sigma_pair(z_i))     = f_pair(z_i)
(mu_entity(z_i), log_sigma_entity(z_i)) = f_entity(z_i)
```

只用 normal train 最小化 Pair/Entity Gaussian NLL：

```text
NLL(y, mu, sigma) = 0.5 ((y-mu)/sigma)^2 + log(sigma)
```

测试只惩罚超过正常条件期望的强度：

```text
E(y, mu, sigma) = 0.5 ReLU((y-mu)/sigma)^2
```

Entity-A 与 Entity-B、以及 Pair 与 Entity 使用现有 normalized log-sum-exp
平滑聚合。

## 7. Calibration 与 Final score

normal calibration 分别为 Local、Pair Context、Entity Context 和总 Context
拟合 Q05/Q95 连续尺度：

```text
a(s) = max(0, (s-Q05)/(Q95-Q05))
```

不进行 upper clipping。Final 保持 V4 的 smooth OR：

```text
final = T * logmeanexp([a_local/T, a_context/T])
```

deployment threshold 仅由 normal calibration 的目标 FPR 分位数确定；攻击
标签只用于最终 evaluation。

## 8. 模型与输出

Context checkpoint：

```text
sparse_context_model.pt
format_version = "4.5"
context_mode = "sparse_context"
```

每条 Flow CSV 至少输出：

- Pair/Entity-A/Entity-B observed intensity；
- expected intensity mean/scale；
- NULL weight、relevance、support size、effective neighbors；
- Pair/Entity/Context raw anomaly；
- calibration-scaled Local/Context/Final anomaly；
- deployment prediction。

`os_scan` manifest evaluation 还会生成
`analysis/os_scan_sparse_attention.json`，按攻击前/攻击期/攻击后汇总攻击源、
目标实体及其正常 source-target 业务的 NULL weight、support size、effective
neighbors、Context/Final score 和 FP，用于直接核查 benign Flow 是否仍被
低相关攻击邻居连带抬高。

## 9. 运行命令

完整训练：

```bash
cd /home/mfh/Desktop/test/myModel
/home/mfh/miniconda3/envs/wxy/bin/python run_pipeline.py \
  --config configs/gotham_v45_train.yaml --mode prepare
/home/mfh/miniconda3/envs/wxy/bin/python run_pipeline.py \
  --config configs/gotham_v45_train.yaml --mode train --device cuda
```

九数据集评测：

```bash
/home/mfh/miniconda3/envs/wxy/bin/python evaluate_gotham_manifest.py \
  --config configs/gotham_v45_train.yaml --device cuda --keep-intermediates
```

单元测试和 CPU smoke：

```bash
/home/mfh/miniconda3/envs/wxy/bin/python -m unittest discover -s tests -v
/home/mfh/miniconda3/envs/wxy/bin/python tests/generate_smoke_pcaps.py
/home/mfh/miniconda3/envs/wxy/bin/python run_pipeline.py \
  --config configs/smoke_v45.yaml --mode all --device cpu
```
