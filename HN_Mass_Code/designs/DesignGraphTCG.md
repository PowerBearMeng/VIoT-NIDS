# Flow线图与TCG：端到端H／HN污染修复实验

## 目标与固定范围

仅使用全部正常捕获的12,518条Flow进行正常预训练及联合训练；不划分验证集、
不读取攻击标签训练、不按攻击效果选checkpoint。先H后HN，评估OS Scan、LRDoS Flood、
LRDoS Stealth三份camera-udp Access-OVS捕获。MIL、payload、TCP flags和包序列不进入本轮。

检测单位是单向、1秒五元组Flow Slice。原25维统计、IPv4长度1500 clamp、乱序缓冲、
包含任何攻击包即标攻击的规则与V7一致。IP／端口／协议只参与提取和结构匹配；
持久缓存只保存匿名索引，没有原始IP、端口、服务类别或固定身份embedding。

## 图与时钟

原始通信图以匿名IP为节点，每条Slice是独立平行边。模型使用按匿名(IP,port)
端点展开的Flow线图；Flow变为节点，关系区分共享发送端、共享接收端、两种收发关联。
因此不只是共享IP就默认混合通信。匿名索引只用于等值匹配、查找，不作为神经网络数值输入。

历史线图查询最近16秒，四种关系各至多8个邻居。TCG同向最近8个、反向最近4个，
最大回溯64秒。历史线图排除已由同向／反向通道覆盖的连接，目标自行排除；重复采样邻居合并。
HN当前线图四种关系各至多8个。训练随机采样；评分种子按稳定事件编号生成，
不因其他目标的遍历顺序改变邻居选择。实际Flow不合并、不复制，
关系incidence数量单独记录而不当作每条Flow的异常分数。没有稠密Flow两两矩阵。

所有完整历史Slice截至t-1。当前邻居是单独的t秒通道，不提前进入历史。
空白秒保留真实时间差；每个capture独立初始化，padding不是实际流量。

到达间隔：已有五元组取上一Slice首包时间；首次五元组取相同方向匿名IP对、
相同正常行为锚点的上一条新连接时间。时钟按首包时间、稳定首帧编号排序，只看前序事件。
同时间事件允许零间隔；无前序事件跳过时间NLL，不把新连接直接视为攻击。
间隔log1p后使用全部正常可用间隔拟合RobustScaler，下限0.05。
16锚点来自新预训练f0的正常embedding，冻结独立测量副本用于稳定事件分组，
检测模型fθ一直参与端到端训练。

## 网络与目标

共享encoder：25→64、LayerNorm、GELU→32。Local decoder为32表示+25mask→64→25，
不读取历史或图。三组遮蔽、训练coverage修正及评分按坐标预测次数平均沿用V7。

关系注意力hidden64、4头、关系类型embedding8、log时间差投影16；没有身份embedding。
私有TCG产生目标自身参照；历史／当前其他连接通过关系门控消息平均形成受限均值修正。
Context不能读取完整目标表示：使用目标遮蔽表示+mask、私有历史预测2分量25维对角Gaussian。
只在遮蔽坐标上算联合混合NLL并按遮蔽维数平均；评分平均三个分组。
σ=softplus(rho)+0.05。共享历史和当前消息不进入混合权重或σ。

mu=mu_private+sigma_private*tanh(delta_history+delta_current)。H的current项恒为0；
HN的当前输出层初始化为0。门控作用于消息、不是攻击概率，不乘检测分数。
时间头读取当前目标形态、私有TCG及冷启动／新连接观测，输出2分量Gaussian，
不读取实际间隔、当前邻居、共享历史消息或邻居数量。

关系对比只用正常数据：真实邻接配对为正来源；一半随机、一半匹配正常窗口Flow数量四分位，
从其他正常窗口重采样配对作为负来源。标签表示配对来源，不是攻击。
推理不做在线训练，critic不作为检测分数。

## 训练与初始化

从头正常预训练20epoch、Adam1e-3。H／HN从同一新预训练和同一初始图模型分别训练40epoch，
不续训H完成参数。encoder学习率3e-4，新增模块1e-3，batch256，clip5，seed42。
每epoch完整覆盖全部正常Flow，再补充行为／10秒块平衡抽样至max(100,ceil(N/256))步。
上下文始终来自真实数据，抽样副本不进入图。

L=Local Huber+λs Shape NLL+λt Timing NLL+λr Relation来源分类。
16个正常batch校准共享encoder梯度比例1:1:1:0.2，分母下限1e-8，权重[0.01,10]后固定。
记录loss、尺度、门控、梯度、encoder参数变化及邻居embedding梯度。
训练每batch重算节点embedding并保留梯度；只缓存原始统计和图索引。
仅冻结checkpoint评分时允许编码缓存。保存最后完整epoch，非有限值停止修复。

## 评分、报告与边界

Context=(Shape NLL+可用Timing NLL)/(1+间隔是否可用)。各项原始NLL另行保留。
Local、Context分别用全量正常重评分Q50/Q95作不截断仿射变换，fusion为两者和。
正常q99仅参考；测试AUROC、average precision、最近离散ROC EER，不报告固定q99 TPR/FPR。

H与HN端到端更新后Local可能不同，必须分别报告Local／Context／Fusion及各分量幅度。
比较原V7 D，完整统计误报漏报阶段、新／持续连接、单包60字节、稀疏控制样本。
标签只用于最终指标及离线攻击消息／时钟移除诊断，所有反事实保持checkpoint不变。
私有攻击历史仍可能被预测器解释，相关新连接也可能影响正常新连接的时钟；
限定均值修正不能当作已经解决全部污染，必须由逐场景结果与通路诊断判断。

## 2026-10-08 首轮执行结果

已完成新预训练20轮、H／HN各40轮，以及三个完整捕获的评估和标签隔离诊断。
主encoder均更新，历史／邻居embedding梯度非零；匿名ID置换、自身排除、历史因果性、
事件级评分采样、不相关Flow不影响目标、连续分数有限等必要核对通过。
没有运行smoke test，也没有生成SHA256清单。

三个场景的融合宏平均（不以pooled替代）：

| 模型 | AUROC | average precision | EER |
| --- | --- | --- | --- |
| 原V7 D | 0.957232 | 0.996509 | 0.099509 |
| H | 0.949899 | 0.993114 | 0.086825 |
| HN | 0.947513 | 0.993027 | 0.077973 |

HN在两个LRDoS场景相对H改善，但OS Scan退化；EER宏平均下降不足以证明整体排序改善。
OS Scan/H形态NLL的EER仅0.008151，加入到达间隔再融合Local逐步退化。
两份LRDoS正常Flow所选图消息没有攻击Flow，删除攻击消息不能降低误报。
高分正常RTCP实例的Local误差主要集中在重复的IAT统计；相似实例已经存在于正常训练数据。
这是本轮观察，尚未用新的损失或评分方法进行修复，也未根据攻击结果调整当前模型。

完整报告：[REPORT.md](/home/mfh/Desktop/test/myModel/outputs/graph_tcg_pilot/REPORT.md)。
模块与实例分析：[MODULE_ANALYSIS.md](/home/mfh/Desktop/test/myModel/outputs/graph_tcg_pilot/MODULE_ANALYSIS.md)。
正常缓存、checkpoint、日志、连续分数、曲线与全部EER错误均保留；共享攻击提取临时缓存已清理。
