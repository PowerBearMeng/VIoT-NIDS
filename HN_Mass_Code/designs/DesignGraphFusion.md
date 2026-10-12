# 正常通信联合建模：表示融合、门控与联合预测

## 固定研究问题

融合目标Flow的可见形态与其私有历史，从全部12,518条正常Flow中学习条件预测分布。
攻击标签、业务类别、IP/端口数值、Local/Context异常分数均不是融合输入。
门控与对比学习组合属于待验证设计，不预设检测收益。

## 输入与图

复用单向1秒五元组Flow Slice及25维统计；五元组只负责聚合及匿名图索引。
输入在全量正常数据上拟合log1p和RobustScaler；间隔使用独立正常尺度。
固定预训练副本仅负责16个行为锚点与事件分组，端到端阶段的主encoder持续更新。
历史仅使用过去完整秒。H使用TCG私有历史与相关历史；HN额外读取当前相关邻居。
私有TCG的时间差以该目标已观测私有历史中的最新事件为原点，只描述历史节点彼此的间距。
不得使用“当前目标首包时间−上一Slice首包时间”作为私有边时间输入，它会复制待预测的到达间隔。
相关历史和当前邻居仍可读取原关系时间差，但仅进入形态均值修正，不进入融合表示或到达因子。
当前邻居只修正形态均值，不改变融合表示、混合权重、标准差或间隔预测。

## 表示融合

三组遮蔽沿用原实现。Local隐藏表示l=64维；遮蔽目标与私有历史state形成c=64维。
g=sigmoid(MLP156→64→64([l,c,mask25,flags2,private_available1]))。
D=MLP128→64→64([l,c])，u=c+g*D；中间均GELU。
gate和D最后一层均零初始化，初始u=c。Fixed版本固定g=0.5。
融合形态头64→64→102，复制原密度头，输出两分量对角Gaussian，sigma=softplus+0.05。
均值修正继续使用sigma*tanh(delta_history+delta_current)。
间隔头162→64→6，输入完整目标32、私有历史64、flags2、融合表示64；原98输入参数复制，新增64列零初始化。
完整目标只供间隔头及辅助critic使用，不能进入形态预测。

## 目标与评分

每组mask的条件因子为p(x_M|x_O,H,N)*p(gap|x,H,u)。
loss_m=(shape_joint_NLL+valid_gap*gap_NLL)/(masked_dimension+valid_gap)。
三个mask均值为融合损失及最终连续分数。跨mask是复合评分，不声称单一25维联合密度。
缺失间隔被边缘化，只评分形态，不填为正常低分。Gaussian常数保留，NLL允许负值。
最终不加入旧分支分数，不乘gate/critic，也不截断。旧分支仅供诊断，其Q50/Q95重新拟合全量正常分布。
最终正常q99仅作为训练内参考，不选择模型。

## 来源对比

真实正常目标和私有历史为正来源；其他正常10秒块、同新连接标记且有私有历史为重采样来源。
每目标一个随机参照及一个从32候选中按可见字段距离选取的参照，二者不同。
更换私有历史并保留来源相对时间，重算c和u；无候选跳过，无完整两两矩阵。
critic输入u64、完整目标32、mask25，MLP121→64→1。正/重采样等权BCE。
标签只代表来源，不代表攻击；不强制所有重采样不合理，无margin，critic不参与检测分数。

## 固定预算

阶段一：H/HN各Fixed、Gated、Gated+Contrast，冻结旧模型，每种20轮。
阶段二：H/HN各原模型继续训练、Gated、Gated+Contrast，分别40轮，共六个端到端模型。
H只使用相关历史；HN在H基础上增加当前邻居。Gated+Contrast是在门控融合基础上增加关系对比辅助目标，不是单独的对比模型。
原模型继续训练对照及融合模型均保留原四项loss及原权重。
联合encoder学习率3e-4、新模块及原head1e-3，Adam，clip5，seed42，batch256。
每轮全覆盖正常样本再按原行为/10秒块平衡采样补至至少100步，图不增加副本。
16个正常batch标定权重：阶段一融合:来源=1:0.2；阶段二Local:融合:来源=1:1:0.2。
分母下限1e-8，新增权重限制[0.01,10]，Gated与Gated+Contrast共用阶段二融合权重，记录实际梯度比。
最后完整轮model.pt是最终模型；recovery.pt只用于数值/中断恢复，不是best.pt。
无验证划分、早停或攻击调参。配置在首次攻击评分前保存protocol.json。

## 评估与文件

阶段一固定三份camera-udp捕获（OS Scan、LR Flood、LR Stealth）。
2026-10-09用户调整下一次阶段二测试为camera-udp的OS Scan（0924-1128）、LRDoS Stealth（0924-1312）、SYN DoS（0924-1233）。
训练配置与预算保持固定，仅改变冻结模型后的评估集合。CLI保留all22选项，本轮不宣称完成22场景。
不同集合的宏平均分别保存；模块比较仅在双方共同完成的捕获上计算。
随后用户要求将六个端到端模型扩大到全部22份捕获，并与已有TFusion结果比较。
已有三个场景的冻结评分直接复用，其余19份逐捕获共享匿名图缓存；不重新训练或根据攻击调整模型。
对比实现为fusion_tfusion_comparison.py，CLI为compare-fusion-tfusion --require-complete。
TFusion与图模型都使用同一正常捕获、单向1秒Flow及相同packet labels。20份捕获数量一致，SYN DoS/camera-udp和UDP Flood/camera-udp的TFusion各多一条攻击Flow，来自晚到包的时间桶处理差异；正常数量均一致。
主对比EER使用双方的插值口径，图模型原离散EER单独保留；AP使用average precision。
TFusion原连续分数已被旧实验清理，仅报告其22捕获宏平均，不从逐捕获指标伪造pooled结果。
按捕获及模型报告Local、Context、旧融合、形态、新融合形态、新融合的AUROC、AP、EER及曲线。
逐Flow保留连续分数、门控、修正、历史支持、间隔可用性及高分正常/低分攻击实例。
RTCP等业务信息只从独立离线元数据读取；其余捕获的TCP ACK精细子类无原元数据时不推断。
各捕获独立历史，统计缓存各模型共用；仅冻结模型允许缓存encoder表示。
实际正常数据核对遮蔽泄漏、当前邻居隔离、匿名置换、不相关Flow不变性、梯度与完整覆盖。
不运行smoke test或大量SHA256。正常PIT与NLL是训练内诊断，检测结果才是独立捕获证据。

日志中的encoder_delta_l2定义为本次训练末参数相对本次训练起点的整体L2距离：sqrt(sum((theta-theta_start)^2))。
它包含encoder权重、偏置及LayerNorm参数；不是单步梯度、百分比或检测指标。冻结阶段应为零，联合阶段非零只证明参数更新。
逐轮数值保存在train_log.csv，常规进度不再重复显示它。

## 执行

```bash
.venv_v7/bin/python -m viot_nids_graph_pilot --config configs/graph_fusion_pilot.yaml run-fusion
```

CLI另外提供train-fusion、score-fusion、benchmark-fusion、report-fusion。
原结果保留；所有新结果位于outputs/graph_fusion_pilot。

## 推理执行记录

训练批次始终为256。大捕获推理采用2048批次，事件种子控制的邻居选择不依赖批次。
实际12,518条正常Flow上核对256与2048推理，最终分数最大浮点差约2.1e-6，Local分数不变。
早期完成的捕获采用256，其余捕获在complete.json记录实际推理批次。
图候选使用稀疏位置查询，保留原采样结果与全部关系计数；固定索引使用紧凑整数存储及磁盘映射，避免整份图重复复制。
训练参数、预算和架构不根据攻击结果修改。

## 时间输入修正记录（协议v2）

旧实现中，11,474条已有正常连接的私有历史边包含待预测gap的等价副本。
旧实验完整移至outputs/graph_fusion_pilot_invalid_gap_input_20261009，标为失效，不据此宣称检测收益。
协议v2用past_relative_v1修正私有时间原点，包括重采样历史；所有12个版本重新按20/40轮预算训练。
初始化仍按计划加载已有H/HN参数，但训练、正常重评分及攻击评分一律使用修正后的私有时间输入。
基线相加、固定门控及学习门控均使用同一修正输入；旧backbone参数的初始化来源作为实验限制记录。
新增核对直接改变真实目标的首包时刻、保持过去观测及Flow字段不变，要求私有边时间及到达预测参数均不变。
