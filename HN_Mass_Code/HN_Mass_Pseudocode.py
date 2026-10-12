"""当前HN门控＋对比＋完整群体强度：中文阅读用伪代码。

对应2026-10-12四场景小实验，单向1秒Flow。
实际代码见real_code/viot_nids_graph_pilot/group_intensity.py等源码副本。
这里省略PCAP reader、索引器和batch工具，不作为独立训练程序运行。

当前两路：
    Flow＋私有历史＋图邻居 → 冻结HN门控/对比模型 → S_HN
    过去完整活动＋行为/方向 → 正常强度预测 → 当前量的NLL → S_Mass
    正常Q50/Q95尺度对齐 → Z(S_HN)＋Z(S_Mass) → 每条Flow最终分数

本轮原HN没有更新encoder；新增强度预测器用全部正常数据训练20轮。
端口没有数值输入；原Flow划分和图关系查询仍使用端口相等关系。
"""

import math
from collections import defaultdict


def prepare_one_second_flows(packets, normal_artifacts):
    """省略真实提取器；算法步骤如下。

    1. 按(floor(packet_time), src_ip, dst_ip, src_port, dst_port, protocol)
       聚合。方向不同为不同Flow；长连接每秒产生独立Slice。
    2. 计算每条Flow的25维统计，含包数、字节数、包长、IAT及bin活动。
       不向encoder输入IP、port、protocol、payload、业务或攻击标签。
    3. 用匿名索引记录source/destination node、(IP,port) endpoint、
       connection和reverse_connection；仅用于分组与查询。
    4. 正常拟合的log1p＋Robust尺度变换25维统计。
    5. 冻结的测量encoder产生表示，查询16个正常行为锚点的RBF响应，
       取argmax，得到behavior=0..15。该类别由统计形态学习得到。
    6. 同一row_id保留秒、首包时间、原始包量/字节量与匿名结构索引。
       标签另存，只在冻结后的评估中按row_id对齐。
    """
    raise NotImplementedError("参见真实data.py及Scorer.prepare_data")


def score_original_hn(flow, capture, frozen_hn, normal_artifacts):
    """原HN评分，详细张量伪代码见HN_GatedContrast_Pseudocode.py。

    x = normal_log1p_robust(flow.raw_25_features)
    private = 64秒内同连接最近8条＋反向最近4条，完整Slice仅截至上一秒
    historical = 最近16秒线图端点关联，四方向关系各最多8条
    current = 当前整秒端点关联，四方向关系各最多8条，排除自身
    flags = [是否新连接, min(64,capture已运行秒数)/64]  # 不是TCP flags

    scores = []
    for mask in 三组固定mask:  # 包长11维、时间16维、数量18维
        z = encoder(x * (1-mask))
        l = Local隐藏层(concat(z, mask))
        q = Query(concat(z, mask, flags))
        p = Private关系注意力(q, encoder(private.features))
        c = ContextMLP(concat(q, p))
        a = 是否存在私有历史
        g = sigmoid(G(concat(l,c,mask,flags,a)))
        u = c + g * D(concat(l,c))

        shape_weights, mu, sigma = ShapeMDN(u)
        dh = HistoryMeanCorrection(c, encoder(historical.features))
        dn = CurrentMeanCorrection(c, encoder(current.features))
        mu = mu + sigma * tanh(dh + dn)
        # 当前邻居只修正mu，不改变权重/sigma/u/到达预测。

        time_distribution = TimingMDN(concat(encoder(x), p, flags, u))
        # 实际到达间隔只作为下方NLL的答案，不能输入预测头。
        shape_joint_nll = -log p(x[mask] | shape_distribution)
        time_nll = -log p(gap | time_distribution) if gap_available else 0
        scores.append((shape_joint_nll + time_nll) / (mask_size + gap_available))
    return mean(scores)

    测试冻结所有参数。关系对比critic不直接输出检测分数。
    """
    raise NotImplementedError("参见真实fusion.py、fusion_scoring.py")


def full_group_observations(flows, behavior_of):
    """完整真实量：跨端口累计，不做邻居采样、不按异常阈值筛选。

    activity[(匿名节点, 发送/接收方向, 秒)][behavior] = [Flow数,包数,字节数]
    每个节点/方向每秒得到16×3=48维。群体包含目标自身。
    本段是可读算法表达；实际实现用bincount批量计算。
    """
    activity = defaultdict(lambda: [[0.0, 0.0, 0.0] for _ in range(16)])
    for row in flows:
        behavior = behavior_of[row.row_id]
        for node, direction in ((row.source_node, "send"),
                                (row.destination_node, "receive")):
            total = activity[(node, direction, row.second)][behavior]
            total[0] += 1
            total[1] += row.packet_count
            total[2] += row.byte_count
    return activity


def choose_quantity_view(channel_totals, variant):
    """Mean对照丢掉绝对数量；Mass保留绝对数量。"""
    if variant == "mass":
        return channel_totals
    total_flows = max(1.0, sum(channel[0] for channel in channel_totals))
    return [[value / total_flows for value in channel] for channel in channel_totals]


def predict_normal_intensity(past_16_seconds, valid_length, behavior,
                             direction, normal_model, artifacts):
    """输入与输出；节点编号和端口编号不交给normal_model。

    history = normal_input_scale(log1p(过去16秒16×3的活动))
    h = GRU(48,64)(history[:valid_length]) if valid_length>0 else zeros(64)
    query = concat(h, one_hot(behavior,16), one_hot(direction,2))  # 82维
    log_weights = log_softmax(MLP(82,64,8)(query))

    # 8状态×16行为×3资源；参数只在正常训练时学习。
    state_means = normal_lower + (normal_upper-normal_lower)*sigmoid(mean_logits)
    state_scales = softplus(scale_logits) + 0.15
    return log_weights, state_means[:,behavior,:], state_scales[:,behavior,:]

    历史只读[t-16,t-1]。节点出现后的空白秒是零量，出现前padding不更新GRU。
    当前数量不能输入GRU或混合权重网络。测试冻结状态与均值/尺度。
    """
    raise NotImplementedError("参见真实Predictor.forward")


def intensity_nll(current_quantity, behavior, distribution, normal_target_scales):
    """三维变换空间的平均NLL；允许负值，不截断。

    y = normal_target_scales[behavior].transform(current_quantity)
    logw, mu, sigma = distribution
    logp_k = logw[k] + sum_j(
        -0.5*((y[j]-mu[k,j])/sigma[k,j])**2
        -log(sigma[k,j]) -0.5*log(2*pi)
    )
    return -logsumexp(logp_k) / 3

    这是连续Gaussian密度近似，不是离散Flow计数的精确概率质量函数。
    """
    raise NotImplementedError("参见真实Predictor.nll")


def normal_affine(score, q50, q95):
    return (score - q50) / max(q95 - q50, 1e-6)


def score_flow_with_mass(flow, behavior, activity, hn_score, mass_model, artifacts):
    """最终主评分：sender强度；receiver另存诊断。

    history = 同发送节点、发送方向的[t-16,t-1]完整48维活动
    dist = predict_normal_intensity(history, available_length, behavior,
                                    send_direction, mass_model, artifacts)
    observed = activity[(flow.source_node,"send",flow.second)][behavior]
    s_mass = intensity_nll(observed, behavior, dist, artifacts.target_scales)
    z_hn = normal_affine(hn_score, artifacts.hn_q50, artifacts.hn_q95)
    z_mass = normal_affine(s_mass, artifacts.mass_q50, artifacts.mass_q95)
    return z_hn + z_mass

    此处是固定相加，没有新门控、attention或容量求解。
    同发送节点、行为和秒的Flow共享s_mass；最终HN部分仍逐Flow不同。
    """
    raise NotImplementedError("参见真实group_intensity.main的评分部分")


def train_and_evaluate_current_pilot():
    """本次实际实验顺序，不按攻击效果挑epoch或融合权重。

    1. 加载并冻结已训练HN/joint/gated-contrast checkpoint。
    2. 全部12,518条正常Flow按上述规则形成19,109个占用活动-行为对。
       正常数据拟合输入尺度、各行为目标尺度和8个活动状态初始化。
    3. 分别训练Mean和Mass各20epochs，seed42，batch256，Adam1e-3。
       每epoch完整覆盖正常对，行为平衡补采样至至少100步，clip_grad=5。
       补采样改变训练次数，实际Flow数量不增加。
       仅出现的行为通道作为强度目标；真实零活动仍进入历史。
    4. 冻结新预测器，全部正常Flow重评分，拟合各支路Q50/Q95。
    5. 冻结后评估Mirai、SYN、LRDoS Stealth、OS Scan四份捕获。
       后三者固定camera-udp配置。每份capture历史独立重置。
    6. 逐Flow对齐并保存HN、Mean、Mass、HN+Mean、HN+Mass。
       标签仅用于AUROC、average precision、插值EER与离线归因分析。

    输出至少包含row_id、timestamp_second、HN、mass、HN_mass、
    sender_group_flows/packets/bytes、intensity_history_seconds。
    同组强度相同，群体内部归因尚未解决。本轮没有重新联合训练主encoder。
    """
    raise NotImplementedError("实际入口：python -m viot_nids_graph_pilot.group_intensity")
