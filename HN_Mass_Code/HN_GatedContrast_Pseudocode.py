"""HN 门控＋对比：当前已评估模型的阅读用伪代码。

对应 run：outputs/graph_fusion_pilot/HN/joint/gated-contrast。
这是架构说明，不是另一个训练实现；省略 PCAP reader、索引器和 batch 调度。
张量部分用 PyTorch 表达；标有“省略”的函数需要实际实现才能训练。
实际代码：viot_nids_graph_pilot/{network,fusion,fusion_training,fusion_scoring}.py。

输入                           模型                               输出
单向 1 秒 Flow 的 25 维统计 ──→ 共享 encoder ──→ Local 隐藏表示 l ─┐
自身/反向连接的过去 Slice ──→ 共享 encoder ──→ 私有历史表示 c ───┤
                                                               ↓
                                                      残差门控融合 u
                                                               ↓
相关历史线图邻居 ───────────→ 关系消息 ─┐                形态 MDN
当前秒线图邻居 ─────────────→ 关系消息 ─┴─→ 仅修正形态均值 μ
完整目标 Flow + 私有历史 + u ─────────────→ 到达间隔 MDN
                                                               ↓
实际被遮蔽字段 + 实际可用间隔 ─────────────→ 条件 NLL → 单 Flow 分数

训练额外有：原四项辅助损失 + 融合来源对比 critic。
测试没有在线对比学习；critic、门控值、旧分支分数不直接加入最终分数。
当前邻居不进入融合门控、分布权重、标准差或到达间隔预测。

重要限制：这里预测形态和间隔，没有直接预测邻域 Flow 数量/到达强度。
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


# 1. 数据输入；这里的包长是现有 reader 的 wire_length。
FEATURE_NAMES = (
    "packet_count", "byte_count", "active_duration",
    "packet_len_mean", "packet_len_std", "packet_len_min", "packet_len_max",
    "packet_len_p25", "packet_len_p50", "packet_len_p75",
    "iat_mean", "iat_std", "iat_min", "iat_max", "iat_p25", "iat_p50",
    "iat_p75", "iat_cv", "active_bin_count", "bin_bytes_mean",
    "bin_bytes_std", "bin_bytes_max", "bin_packets_mean",
    "bin_packets_std", "bin_packets_max",
)

# m=1 表示遮蔽。三个固定分组，不是测试时随机抽 mask。
MASK_GROUPS = (
    (1, 3, 4, 5, 6, 7, 8, 9, 19, 20, 21),  # 包长：11 个坐标
    (2, *range(10, 25)),                       # 时间：16 个坐标
    (0, 1, 2, *range(10, 25)),                 # 数量：18 个坐标
)


def masks_like(x):
    for columns in MASK_GROUPS:
        mask = torch.zeros_like(x)
        mask[:, list(columns)] = 1
        yield mask


def prepare_capture(packets, normal_fitted_scalers, frozen_measurement_encoder):
    """数据部分伪代码；每份 capture 独立建立历史。

    1. 按 (floor(timestamp), src_ip, dst_ip, src_port, dst_port, protocol)
       聚合，得到单向、1 秒 Flow Slice；反向通信是另一条 Flow。
    2. 每条 Slice 算 FEATURE_NAMES 的 25 个统计，记 raw_x。
       x = (log1p(raw_x) - 正常中位数) / max(正常 IQR, 尺度下限)。
       所有 scaler 只用全部正常数据拟合，不在攻击捕获上重新拟合。
    3. IP/port/protocol 仅用于生成匿名节点、端点、连接及反向连接索引。
       网络不读取这些数值、索引 embedding、服务类别或攻击标签。
       本轮也没有 TCP flags、payload、逐包序列输入。
    4. 16 个固定行为锚点由测量 encoder 的正常表示产生。
       锚点只用于新连接的事件分组；不是主 encoder 的冻结缓存。
    5. 间隔标签 gap：
       已有连接：当前 Slice 首包 - 此连接上一 Slice 首包。
       新连接：当前首包 - 同方向匿名 IP 对、同锚点的上条新连接首包。
       无前序则 gap_valid=False；不能把缺失间隔当作正常低分。
       gap_scaled 使用独立的正常 log1p + RobustScaler。
    6. meta_flags = [new_connection,
                     min(64, 当前秒 - capture 起始秒) / 64]。
       这两个是结构/运行时间标记，不是 TCP flags。
       后者实际为 capture 已运行时间，不是私有历史记录数。
    7. 标签通过 row_id 独立保存，只用于最后的离线评估。
    """
    raise NotImplementedError("省略：复用实际数据提取及预处理代码")


def query_graph(target, index, training):
    """为目标分别查询三个稀疏通道，保留原始 Flow 数量。

    private: 64 秒内，同向连接最近 8 条 + 反向连接最近 4 条。
    historical: 最近 16 秒、截至 t-1 的相关端点邻居，4 关系各最多 8 条。
                同向/反向连接链已走 private，不在此重复。
    current: 当前整秒的相关端点邻居，4 关系各最多 8 条，排除自身。

    四种端点关系：发送-发送、接收-接收、发送-接收、接收-发送。
    按匿名 (IP,port) 相等查询；不构造完整两两矩阵，候选重复时合并。
    训练均匀采样；评分按稳定 event_id 和固定种子采样。
    当前整秒可包含目标首包之后的邻居，检测需等待这一秒收集完成。

    private_dt = log1p(最后一个已观测历史首包时间 - 各历史首包时间)/log(65)。
    禁止使用“目标首包 - 私有历史首包”作为 private_dt：这会泄露待预测间隔。
    historical/current dt 采用目标与邻居的首包时间差，经 log1p/log(65)。

    每个通道返回 rows、relation_type、dt、valid。
    完整邻居数量另存 diagnostics，不输入后面的密度预测网络。
    """
    raise NotImplementedError("省略：复用 GraphIndex 的查询/采样")


@dataclass
class NeighborChannel:
    z: torch.Tensor       # [B,K,32]：当前主 encoder 编码的邻居完整 25 维输入
    relation: torch.Tensor  # [B,K]：只有关系类型 0..5，不是匿名身份编号
    dt: torch.Tensor      # [B,K]：上面定义的变换后时间差
    valid: torch.Tensor   # [B,K]：padding 为 False


class RelationAttention(nn.Module):
    """四头关系注意力；复现实际消息平均与门控路径。"""

    def __init__(self):
        super().__init__()
        self.relation = nn.Embedding(6, 8)
        self.time = nn.Sequential(nn.Linear(1, 16), nn.GELU())
        self.content = nn.Sequential(nn.Linear(56, 64), nn.GELU())
        self.q, self.k, self.v = (nn.Linear(64, 64) for _ in range(3))
        self.edge_gate = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.out = nn.Linear(64, 64, bias=False)

    def forward(self, query, channel, average_by_relation=False):
        z, r, dt, valid = channel.z, channel.relation, channel.dt, channel.valid
        content = self.content(torch.cat([z, self.relation(r), self.time(dt[..., None])], -1))
        q = self.q(query).reshape(-1, 4, 16)
        k = self.k(content).reshape(len(query), -1, 4, 16)
        v = self.v(content).reshape(len(query), -1, 4, 16)
        similarity = (q[:, None] * k).sum(-1) / math.sqrt(16)
        repeated_query = query[:, None].expand(-1, z.shape[1], -1)
        edge_logits = self.edge_gate(torch.cat([repeated_query, content], -1)).squeeze(-1)
        gate = edge_logits.sigmoid() * valid

        if average_by_relation:
            # historical/current：每种关系均值，再对四关系平均。
            # 把相同消息重复很多次，不会按数量线性增大输出。
            head_weights = similarity.sigmoid() * gate[..., None]
            messages = []
            for rel in range(4):
                selected = valid & (r == rel)
                denominator = selected.sum(1).clamp_min(1)[:, None, None]
                messages.append((v * (head_weights * selected[..., None])[..., None]).sum(1)
                                / denominator)
            message = sum(messages) / 4
        else:
            # private：邻居维度 softmax；没有历史时输出零。
            weights = similarity.masked_fill(~valid[..., None], -1e9).softmax(1)
            weights = weights * valid[..., None]
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
            message = (v * (weights * gate[..., None])[..., None]).sum(1)

        result = self.out(message.reshape(-1, 64)) * valid.any(1)[:, None]
        mean_gate = gate.sum(1) / valid.sum(1).clamp_min(1)
        return result, mean_gate, edge_logits


def unpack_shape(parameters):
    # 两个 component，共享每个 component 的对角 Gaussian。
    log_weights = parameters[:, :2].log_softmax(-1)
    mu, rho = parameters[:, 2:].reshape(-1, 2, 50).split(25, -1)
    return log_weights, mu, F.softplus(rho) + 0.05


def unpack_timing(parameters):
    return (parameters[:, :2].log_softmax(-1), parameters[:, 2:4],
            F.softplus(parameters[:, 4:]) + 0.05)


class HNGatedContrast(nn.Module):
    def __init__(self):
        super().__init__()
        # 主 encoder：目标完整/遮蔽输入、所有历史/邻居共享；联合训练中可更新。
        self.encoder = nn.Sequential(nn.Linear(25, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32))
        self.local_hidden = nn.Sequential(nn.Linear(57, 64), nn.GELU())
        self.local_output = nn.Linear(64, 25)
        self.query = nn.Sequential(nn.Linear(59, 64), nn.GELU())
        self.private = RelationAttention()
        self.historical = RelationAttention()
        self.current = RelationAttention()
        self.context = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 64))
        self.gate = nn.Sequential(nn.Linear(156, 64), nn.GELU(), nn.Linear(64, 64))
        self.delta = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 64))
        self.history_delta = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 50))
        self.current_delta = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 50))
        self.fused_shape = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 102))
        self.fused_timing = nn.Sequential(nn.Linear(162, 64), nn.GELU(), nn.Linear(64, 6))
        self.critic = nn.Sequential(nn.Linear(121, 64), nn.GELU(), nn.Linear(64, 1))
        # 保留原 Context 预测头，仅作为辅助训练和分支诊断。
        self.base_shape = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 102))
        self.base_timing = nn.Sequential(nn.Linear(98, 64), nn.GELU(), nn.Linear(64, 6))
        for layer in (self.gate[-1], self.delta[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        # 随后加载正常 backbone，并按 train_normal_only 中的说明复制密度/到达头。

    def represent(self, x, mask, private, meta_flags):
        z_masked = self.encoder(x * (1 - mask))              # [B,32]
        local = self.local_hidden(torch.cat([z_masked, mask], -1))  # l: [B,64]
        query = self.query(torch.cat([z_masked, mask, meta_flags], -1))
        private_message, pg, _ = self.private(query, private)
        context = self.context(torch.cat([query, private_message], -1))  # c: [B,64]
        available = private.valid.any(1).to(x.dtype)[:, None]
        # 64+64+25+2+1=156；不含异常分数或当前邻居。
        gate = self.gate(torch.cat([local, context, mask, meta_flags, available], -1)).sigmoid()
        correction = gate * self.delta(torch.cat([local, context], -1))
        fused = context + correction                       # u: [B,64]
        return fused, context, private_message, local, gate

    def forward(self, x, mask, private, historical, current, meta_flags):
        u, c, p, local, gate = self.represent(x, mask, private, meta_flags)
        h, hg, _ = self.historical(c, historical, average_by_relation=True)
        n, ng, _ = self.current(c, current, average_by_relation=True)
        hd = self.history_delta(h).reshape(-1, 2, 25) * hg[:, None, None]
        nd = self.current_delta(n).reshape(-1, 2, 25) * ng[:, None, None]

        log_w, mu_private, sigma = unpack_shape(self.fused_shape(u))
        mu = mu_private + sigma * torch.tanh(hd + nd)
        # 当前/相关历史只修正形态均值，修正不超过一个当前预测标准差。
        # 当前邻居不改变 log_w、sigma，也不改变 u。

        full_z = self.encoder(x)
        timing = unpack_timing(self.fused_timing(torch.cat([full_z, p, meta_flags, u], -1)))
        # 完整 x 可以预测到达节奏；实际 gap 及其副本不能是输入。
        # 测试时可计算以下旧分支，但它们不叠加到最终分数。
        bw, bm, bs = unpack_shape(self.base_shape(c))
        base_shape = (bw, bm + bs * torch.tanh(hd + nd), bs)
        base_timing = unpack_timing(self.base_timing(torch.cat([full_z, p, meta_flags], -1)))
        return dict(shape=(log_w, mu, sigma), timing=timing, fused=u, gate=gate,
                    local_prediction=self.local_output(local),
                    base_shape=base_shape, base_timing=base_timing)


# 3. 概率评分：被预测值只在这里作为“答案”计算误差，不输入预测头。
def gaussian_log_density(value, mean, sigma):
    return -0.5 * ((value - mean) / sigma).square() - sigma.log() - 0.5 * math.log(2 * math.pi)


def shape_joint_nll(x, mask, distribution):
    log_w, mu, sigma = distribution
    coordinate_logp = gaussian_log_density(x[:, None], mu, sigma)
    # 先在每个 component 内对遮蔽坐标求和，再混合。
    return -torch.logsumexp(log_w + (coordinate_logp * mask[:, None]).sum(-1), -1)


def timing_nll(gap_scaled, distribution):
    log_w, mu, sigma = distribution
    return -torch.logsumexp(log_w + gaussian_log_density(gap_scaled[:, None], mu, sigma), -1)


def score_one_group(x, mask, gap_scaled, gap_valid, prediction):
    shape_total = shape_joint_nll(x, mask, prediction['shape'])
    timing = torch.zeros_like(shape_total)
    if gap_valid.any():
        selected_distribution = tuple(v[gap_valid] for v in prediction['timing'])
        timing[gap_valid] = timing_nll(gap_scaled[gap_valid], selected_distribution)
    score = (shape_total + timing) / (mask.sum(-1) + gap_valid.to(x.dtype))
    return score, shape_total / mask.sum(-1), timing


def inference(model, x, private, historical, current, meta_flags, gap_scaled, gap_valid):
    model.eval().requires_grad_(False)
    outputs = []
    with torch.inference_mode():
        for mask in masks_like(x):                         # 测试固定算全部三组
            predicted = model(x, mask, private, historical, current, meta_flags)
            score, shape, timing = score_one_group(x, mask, gap_scaled, gap_valid, predicted)
            outputs.append(dict(score=score, shape=shape, timing=timing,
                                gate=predicted['gate'], local=predicted['local_prediction']))
    final_score = torch.stack([o['score'] for o in outputs]).mean(0)
    # 每条 Flow 一个连续分数，越大越异常；允许负值，无上下限截断。
    # gap 可用时，间隔项权重分别为 1/12、1/17、1/19，平均约 6.49%。
    # 旧 Context=(平均形态NLL+平均间隔NLL)/2，间隔权重是 50%。
    # 权重差异是实际评分定义，不意味着已经证明它单独造成退化。
    return dict(final_score=final_score, per_mask=outputs, gap_available=gap_valid)


# 4. 正常训练的伪代码说明。
def train_normal_only():
    """
    全部 12,518 条正常 Flow 参与训练，不使用攻击标签选参数/epoch。
    初始化正常 HN backbone；保留原 Local/Context 头和关系模块。
    新形态头复制原 Context 密度头；到达头复制原输入权重，新增 u 权重置零。
    gate/delta 最后一层置零，使初始 u=c。

    阶段一：冻结原 backbone，训练新模块 20 epochs。
    阶段二：从对应阶段一结果继续，主 encoder、历史及全部模块联合训练 40 epochs。
    每个 batch 都用当前 encoder 重新编码目标、历史、邻居、重采样历史，保留梯度。
    测试冻结参数后才允许主表示缓存；冻结测量副本始终仅用于行为锚点分组。

    原辅助任务随机选择三组之一：
        L_local = 按 mask 覆盖次数补偿的 Huber；
        L_shape = 原 Context 形态 NLL；
        L_timing = 原 Context 到达间隔 NLL（仅有前序的 Flow）；
        L_relation = 图关系门控的真实/重采样来源 BCE。
    新融合任务每 batch 算全部三组：
        L_fusion = mean(score_one_group(...))。

    融合对比：目标 x、mask、meta_flags 保持不变，只更换 private 历史。
        真实配对：原 private → c → u → critic([u, encoder(x), mask])。
        重采样：来自不同 10 秒块、同 new_connection 标记、有私有历史的正常目标。
        一个随机参照；另一个从最多 32 候选中按可见字段距离选最近参照。
        来源历史保留内部相对时间；重新编码并计算 u_negative。
        L_pair = 0.5 * [BCE(真实,1) + mean(BCE(重采样,0))]。
        这些 0/1 是来源标签，不是正常/攻击标签。

    联合目标：
        L = L_local + λ_shape L_shape + λ_timing L_timing + λ_relation L_relation
            + λ_fusion L_fusion + λ_pair L_pair。
    λ 通过正常 batch 梯度范数确定并固定，不按攻击结果修改。
    seed=42，batch=256；完整正常遍历后平衡补采样至每轮至少 100 步。
    Adam：encoder 3e-4，其余模块 1e-3；梯度裁剪 5。
    保存最后完整 epoch；没有攻击监督、验证早停、best.pt 或测试时再训练。

    输出逐 Flow：row_id、秒、final_score、形态/间隔NLL、gap_available，
                原 Local/Context/LegacyFusion 分数和门控/修正诊断。
    全量正常 final_score q99 仅作参考；AUROC/AP/EER 使用独立捕获连续分数。
    """
    raise NotImplementedError("阅读用伪代码；实际训练入口见 fusion_training.py")
