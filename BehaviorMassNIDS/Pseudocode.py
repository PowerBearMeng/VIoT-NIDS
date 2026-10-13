"""论文方法简化逻辑：原型偏离＋条件行为强度建模（CBIM）。

实际代码在src/behavior_mass_nids；实现中的mass字段对应CBIM。
"""

def normal_training(all_normal_packets):
    flows = directional_five_tuple_slices(all_normal_packets, seconds=1)
    x = fit_normal_log1p_robust_scale(flows.statistics25)
    f0 = train_masked_prediction_encoder(x, epochs=20)
    freeze(f0)
    prototypes, radial_widths = fit_K_normal_prototypes(f0(x))
    behavior = nearest_scaled_normal_prototype(f0(x))
    # 匿名节点仅作为分组/检索索引，不进入encoder或密度网络。
    snapshots = complete_FPB(flows, anonymous_node=True, direction=True,
                             behavior=behavior, preserve_all_counts=True)
    # 一个节点的历史包含全部K个行为通道；当前秒不是历史输入。
    predictor = train_GRU_and_bounded_state_MDN(
        snapshots.past(seconds=192), snapshots.current(), normal_only=True,
        epochs=20, gaussian_states=8)
    normal_distance, normal_mass = rescore_all_normal(flows, f0, prototypes, predictor)
    reference = fit_Q50_Q95_and_fused_Q99(normal_distance, normal_mass)
    return freeze(f0, prototypes, radial_widths, predictor, reference)


def detect_complete_second(flows, past_history, model):
    z = model.f0(model.flow_scale(flows.statistics25))
    behavior, distance = scaled_nearest_prototype(z, model.prototypes)
    current_FPB = complete_sender_behavior_FPB(flows, behavior)
    density = model.MDN(model.GRU(past_history), behavior, direction="send")
    mass = mean_dimension_NLL(model.mass_scale(current_FPB), density)
    score = normal_affine(distance) + normal_affine(mass)
    # 原型索引用于统计归属，不把“有最近原型”等同于正常。
    alarm = score > model.training_normal_fusion_Q99
    update_history_after_scoring(current_FPB)  # 实际空白秒更新为零；测试权重不更新
    return score, alarm
