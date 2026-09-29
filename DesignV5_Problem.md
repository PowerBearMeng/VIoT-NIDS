所以你现在这个结果，其实把一个更深层问题暴露出来了
你现在的 V5 把正常性主要建模成：
\[
P(z\mid role,regime)
\]
但真实正常性可能应该是：
\[
\boxed{
P(
\text{flow behavior},
\text{temporal state},
\text{relation structure}
)
}
\]
也就是说：
Flow-level normality
“这条 Flow 自己长什么样？”
Temporal normality
“这个行为现在处于什么运行状态？”
Relational normality
“这种 Flow 出现在什么通信关系中？”
你现在 OS Scan/SYN DoS 的 diagnostic 说明第三个维度不能靠简单的 group count + cosine responsibility代替。
这也意味着我之前提的 counterfactual cohort removal 不应该成为你唯一的设计核心
我现在会收回一点之前的强调。
\[
A(G)-A(G\setminus C)
\]
对于验证某个 cohort 是否解释 aggregate anomaly 很有用。
但它更适合作为：
attribution operator

而不是：
定义 cohort 本身的方法。

你的关键问题实际上先是：
\[
\boxed{\text{哪些 Flow 应该被视为同一种 collective behavior？}}
\]
如果 cohort 定错了，再漂亮的 counterfactual 都没用。
这次结果就说明：
\[
(role,regime)
\]
对于 Scan 可能已经够细；
但对于 SYN DoS：
\[
(role,regime)
\]
仍然把 recurring normal service 和 transient attack response 混在一起。
所以需要：
\[
(role,regime,\ relational\ pattern)
\]
如果现在让我给 V5 一个“最小修改版”，我会这么做
不是立刻上新神经网络，而是保留：
\[
Flow Encoder
\rightarrow Temporal Memory
\rightarrow Role
\rightarrow Regime
\]
然后删除现在这一段：
\[
q_i
\rightarrow
\cos(q_i,R^+)
\rightarrow
group\_score\times responsibility
\]
换成：
\[
\boxed{
\text{Regime-scoped relational evidence}
}
\]
也就是说，对：
\[
C=(host,role,regime,t)
\]
计算两类东西：
\[
\text{volume evidence}
\]
例如：
\[
flow\_count,\ bytes,\ packets
\]
和：
\[
\text{relation evidence}
\]
例如：
\[
unique\_peer,\ unique\_port,\ new\_peer,\ new\_service,
port\ diversity,
service\ familiarity
\]
然后 benign-only 学：
\[
P(Context\mid role,regime)
\]
而不是现在：
\[
P(Context\mid role)
\]
这一步其实很关键。
你现在 Context Model 是 Role-conditioned，而你的 diagnostic 已经证明：
同一个 Role 内，不同 Regime 的 Context 行为差异很大。

所以最小的 V5.1，甚至不需要 counterfactual，先改成：
\[
\boxed{
P(Context\mid Role,Regime)
}
\]
我觉得应该首先做。
然后一个 C 里的 Flow 怎么拿 context score？
这里暂时不要再做 cosine。
如果：
\[
C=(host,role,regime,t)
\]
本身异常，那么 C 中的所有 Flow 都获得同一个 cohort context evidence：
\[
S_{c,i}=S_C,\quad i\in C
\]
你可能会问：
SYN DoS 那 58 条正常 Flow 还是会被污染啊。

没错。
所以第二步再引入 relation familiarity。
例如：
\[
r_i=
g(
service\_familiarity_i,
peer\_familiarity_i,
relation\_persistence_i
)
\]
但它不是：
\[
S_C\times r_i
\]
而更建议分别形成：
\[
E_C
\]
和：
\[
E_{relation,i}
\]
最后一起融合。
这就回到你之前问的融合问题
现在 diagnostic 更让我确信：
\[
\max(local,context)
\]
不应该继续用了。
因为你实际上已经有三种证据：
\[
E_L
=
\text{local behavioral evidence}
\]
\[
E_C
=
\text{collective context evidence}
\]
\[
E_R
=
\text{relational evidence}
\]
一个比较自然的框架就是：
\[
E_i
=
w_L E_{L,i}
+
w_C E_{C,i}
+
w_R E_{R,i}
\]
但我甚至不建议人工设：
\[
w_L=0.5,\quad w_C=0.3,\quad w_R=0.2
\]
而是仍然利用你的 benign validation。
把每种 raw score 校准成 benign tail evidence：
\[
E_L=-\log p_L
\]
\[
E_C=-\log p_C
\]
\[
E_R=-\log p_R
\]
然后：
\[
E_{\text{joint}}
=
E_L+E_C+E_R
\]
再使用 benign validation：
\[
S_{\text{final}}
=
F_{\text{benign}}
(E_{\text{joint}})
\]
这样就没有：
\[
group\times responsibility
\]
也没有：
\[
\max()
\]
而是一个统一概念：
\[
\boxed{\text{multi-source anomaly evidence}}
\]
我觉得你这次 diagnostic 最应该写下来的 Finding 其实是这个
不是：
“regime 可以解决 OS Scan，但解决不了 SYN DoS。”

这还太实验化。
应该提升成：
Flow-level behavioral similarity is insufficient for contextual attribution. Flows exhibiting similar local behavior can participate in fundamentally different communication relationships, causing benign and malicious flows to share the same behavioral regime.

中文就是：
仅依赖 Flow 自身的行为相似性不足以完成 Context 归因。局部统计形态相似的 Flow 可能处于完全不同的通信关系中，因此正常 Flow 与攻击 Flow 即使属于同一 Behavioral Regime，也不应共享相同的 Context 异常证据。

这个 Finding 很重要。
它正好解释：
- 为什么 OS Scan 上 regime 已经很好；
- 为什么 SYN DoS 仍然失败；
- 为什么 V5 的 responsibility cosine 不只是数学尺度错；
- 为什么下一步不是“再聚类细一点”，而是引入 relational context。
所以我现在对 V5 的判断会变成：
\[
\boxed{
\text{V5 的 Role/Regime 值得保留；
真正应该重构的是 Context attribution。}
}
\]
并且我建议你下一步优先做的实验不是直接改模型，而是再做一个 read-only diagnostic：
对那 58 条 SYN DoS 中 same-C 的正常 Flow 和同 C 的攻击 response Flow，比较下面这些完全不使用攻击标签训练出来的关系指标分布：
- service_seen_count
- time_since_service_last_seen
- is_new_service
- unique_dst_port / flow_count
- peer-service recurrence
- dst_port frequency percentile within that host→peer history
如果这几组分布明显分开，我们就有很强的证据说明：
\[
\boxed{\text{缺失的是 relational normality，而不是更多的 local clustering。}}
\]
那下一版设计的方向就会比现在清楚很多。

---

## 实证核对与问题归纳（2026-09-29）

下面是对上述判断的核对。这里的攻击标签只用于事后分组和评价；关系统计只由原来的 10,017 条正常训练 Flow 建立。没有训练或修改新模型。

### 已确认的问题

1. **Group 有检测信号，但最终融合几乎丢失了信号。** Camera-TCP SYN DoS 的 `group_score` AUROC 为 0.9626，最终 AUROC 为 0.8429；同单位 TFusion 为 0.9876。攻击 Flow 的 group 分数中位数 0.9944、responsibility 中位数 0.0667、Local 中位数 0.9888；全部攻击 Flow 的最终分数由 Local 决定。因此不能只说“Local 不好”，也不能说“Group 已经达到基线”。
2. **Responsibility 既有尺度 bug，也有归因对象错误。** 当前 `q_i=[1,bytes,packets,peer_share,port_share,new_peer_share]` 保留原始单位，`R⁺` 却是按正常 group 均值/标准差标准化的残差，二者直接计算余弦。这不是概率或可信的责任份额。一次 DoS 由数万条小 Flow 共同使 Flow 数激增，要求每条单包 Flow 的向量与整个 group 残差对齐，天然会压低它们的分数。即使把 `q` 标准化，也不能自动解决这种集体归因问题。
3. **Role/Regime 是行为参照，不是关系类别。** 737,839/737,840 条 SYN DoS 攻击 Flow 被强制分到 role 2；Local 异常分数中位数仍为 0.9888，并非被判正常。Role 2 同时容纳多种正常短通信，且条件密度较宽，可能成为新流量的默认参照；目前没有可解释的未知 role 选项或可靠的归属概率。
4. **只加 Regime 粒度能帮助 Scan，但不足以解决 SYN DoS。** 三份 OS Scan 几乎全部攻击 Flow 均落在 `(role 2, regime 3)`；SYN DoS 的攻击 Flow 则约 77.7% 落在最多的一个 regime，几乎全部落在 regime 2/3。然而 SYN DoS 中 85 条 `group_score≥0.99` 的正常 Flow，有 58 条与攻击 Flow 同属 `C=(src_host,role,regime,second)`，13 条只同属原 group，另外 14 条位于没有攻击的高分 group。故 `P(Context|role,regime)` 值得试验，却不能单靠广播 `S_C` 解决污染。

### 文档建议的关系诊断：已完成

针对上述 58 条与攻击同 C 的正常 Flow，使用正常训练期的 `(src_host,dst_peer,protocol,port)` 出现次数；源端口和目的端口分别统计，再看任一端点是否有熟悉的服务端口。这符合单向 Flow 的实际情况：正常摄像头→NVR 可能是**源**端口稳定，正常 NVR→摄像头则可能是**目的**端口稳定。

| 同 C 的方向 | 正常 Flow | 训练期至少一端端口见过 | 同 C 攻击 Flow | 两端端口都未见过 |
|---|---:|---:|---:|---:|
| NVR→摄像头 | 53 | 53/53 | 168,589 | 168,584/168,589 |
| 摄像头→NVR | 5 | 5/5 | 133,698 | 133,695/133,698 |

53 条 NVR→摄像头正常 Flow 的目的端口主要是 8080/8554，训练期相同通信对出现次数的中位数约 48；对应攻击响应 Flow 的目的端口几乎全是训练期未见过的临时端口。5 条摄像头→NVR 正常 Flow 的目的端口也是临时端口，但其**源**端口在训练期出现过；只用 `new_dst_port` 会误伤它们。这些比较都发生在训练期已知的 host→peer 关系内，不只是识别新 IP。测试捕获内部按秒因果统计也显示，53 条 NVR→摄像头正常 Flow 的相同服务此前出现次数中位数为 5，攻击响应为 1；相同端口在该 host→peer 历史中的 Flow 份额中位数分别为 0.5 和约 0.000012。正常 Flow 的上次服务出现间隔中位数为 1 秒；此前见过的攻击响应为 2 秒，而且约 29.7% 的攻击响应此前未见过同一服务。这些因果历史指标有区分度，但攻击端口会逐秒重复，因此弱于训练期熟悉度。`unique_dst_port/flow_count` 是整个 group 共享的值，不能区分同 C 的 58 条正常 Flow 与攻击 Flow。

这给出的是**当前测试床中的可分性证据**，不是新检测器的性能。确切的主机/端口记忆可能利用了固定拓扑或攻击生成方式。作为最小压力检查：正常训练集的独立验证段 2,501 条 Flow 中，双端端口未见过为 0；两份独立正常环境变化捕获共 4,219 条 Flow 中为 1；但 Mirai 捕获的 3,258 条标注正常 Flow 中为 **233 条（7.15%）**。各攻击捕获中双端端口未见过的比例几乎总是 99.9% 以上，这也提示它可能是数据集 shortcut，不能直接做硬规则或宣称跨设备泛化。逐捕获结果见 `outputs/v5/attack_scaled/RELATION_NOVELTY_DIAGNOSTIC.csv`。

### 对 V5.1 方案的必要修正

- **先定义关系证据，再重拟合 Context。** 保留 Encoder、Temporal Memory、Role、Regime。以正常训练流量建立方向敏感的 host→peer 关系历史；同时观察源/目的端口的熟悉度、服务重复次数和最近出现时间。应有新设备、新服务及端口轮换的冷启动处理，不能把精确 IP/端口是否出现过当成唯一异常条件。
- **Context 可以按 `(role,regime)` 条件建模，但需要样本量回退。** 目前正常训练/验证中的 `(role 2,regime 3)` 分别只有 275/93 个 C，`(role 2,regime 2)` 为 291/94；一些稀疏 role-regime 训练 C 只有 1–8 个，验证 C 甚至为 0。对少样本组合应退回 role 级或共享模型，不能机械地为每个组合拟合 10 维 GMM。93 个验证 C 的经验 CDF 也会快速饱和，应记录并处理分数并列问题。
- **取消当前余弦乘法；把关系证据保留到 Flow 级。** `S_C` 可作为候选集体证据，但同 C 的正常短 TCP 与攻击响应仍需 `E_R` 区分。首先单独评价 `E_L`、`E_C`、`E_R` 及正常 Flow 误报，再决定融合形式。`E_L+E_C+E_R` 是一个可试的候选，不是无参数的自然真理：它隐含三个系数均为 1，而且 Context 与关系证据可能重复计数。各边际尾概率和最终总分应使用互不泄漏的正常验证数据校准，并给概率设置有限下界；攻击标签不能用于权重或方案选择。
- **验证目标不能只看攻击 AUROC。** 在原有 22 份攻击捕获之外，检查独立正常网络/媒体变化、新服务和端口变化场景；报告每个家族的 AUROC/EER、正常 Flow 的高分率，以及 SYN DoS 那 58 条同 C 正常 Flow 的分数变化。继续和同 Flow 单位的 TFusion 对比。只有确认关系证据不是固定端口 shortcut、且不会提高正常误报，再落地 V5.1 的训练与融合。

因此下一步的最小可验证改动是：**正常数据驱动的关系特征 + 常见 `(role,regime)` 的 Context 候选模型 + 稀疏组合回退**，先分别输出分数，不直接部署加和融合。当前证据支持重构 Context attribution，但尚不足以宣称任何特定融合公式已经优于 V5 或 TFusion。
