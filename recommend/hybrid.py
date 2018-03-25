"""
属于融合模型（混合模型）分类下的推荐算法。
"""

import recommend.cf as cf
import recommend.graph as graph
import math
from collections import defaultdict
import numpy as np


def ucfq_reranking_itemdegree(target_user, args):
    # 引入物品流行度的重排序。Score(u,i) <- Score(u,i) / klog(i)^Norm(klog(u))
    #
    # // 特别注意，推荐结果数目是不固定的。

    items_score = cf.ucf_q(target_user, args)

    items_degree = args.get('items_degree', None)
    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p

    items_score_reranking = {item_new: (score / math.pow(math.log1p(items_degree[item_new]), norm_ku))
                             for (item_new, score) in items_score.items()}

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def ucfu_reranking_itemdegree(target_user, args):
    # 引入物品流行度的重排序。Score(u,i) <- Score(u,i) / klog(i)^Norm(klog(u))
    #
    # // 特别注意，推荐结果数目是不固定的。

    items_score = cf.ucf_userdegree(target_user, args)

    items_degree = args.get('items_degree', None)
    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p

    items_score_reranking = {item_new: (score / math.pow(math.log1p(items_degree[item_new]), norm_ku))
                             for (item_new, score) in items_score.items()}

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def ucfuv_reranking_itemdegree(target_user, args):
    # 引入物品流行度的重排序。Score(u,i) <- Score(u,i) / klog(i)^Norm(klog(u))
    #
    # // 特别注意，推荐结果数目是不固定的。

    items_score = cf.ucf_userdegree_neighbordegree(target_user, args)

    items_degree = args.get('items_degree', None)
    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p

    items_score_reranking = {item_new: (score / math.pow(math.log1p(items_degree[item_new]), norm_ku))
                             for (item_new, score) in items_score.items()}

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def icfknnnorm_reranking_itemdegree(target_user, args):
    # 引入物品流行度的重排序。Score(u,i) <- Score(u,i) / klog(i)^Norm(klog(u))
    #
    # // 特别注意，推荐结果数目是不固定的。

    items_score = cf.icf_knn_norm(target_user, args)

    items_degree = args.get('items_degree', None)
    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p

    items_score_reranking = {item_new: (score / math.pow(math.log1p(items_degree[item_new]), norm_ku))
                             for (item_new, score) in items_score.items()}

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def icfj_reranking_itemdegree(target_user, args):
    # 引入物品流行度的重排序。Score(u,i) <- Score(u,i) / klog(i)^Norm(klog(u))
    #
    # // 特别注意，推荐结果数目是不固定的。

    items_score = cf.icf_knn_norm_itemdegree(target_user, args)

    items_degree = args.get('items_degree', None)
    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p

    items_score_reranking = {item_new: (score / math.pow(math.log1p(items_degree[item_new]), norm_ku))
                             for (item_new, score) in items_score.items()}

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def probs_reranking_itemdegree(target_user, args):
    # 引入物品流行度的重排序。Score(u,i) <- Score(u,i) / klog(i)^Norm(klog(u))
    #
    # // 特别注意，推荐结果数目是不固定的。

    items_score = graph.probs(target_user, args)

    items_degree = args.get('items_degree', None)
    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p

    items_score_reranking = {item_new: (score / math.pow(math.log1p(items_degree[item_new]), norm_ku))
                             for (item_new, score) in items_score.items()}

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def ucf_icf_reranking(target_user, args):
    # 引入多样性推荐算法的重排序。波达计数法（Borda Count）
    #
    # // 特别注意，推荐结果数目是不固定的。

    items_score_diversity = cf.icf_knn_norm(target_user, args)  # 面向多样性
    items_score_accuracy = cf.ucf_q(target_user, args)  # 面向准确率

    # 定新物品候选集
    candidates = items_score_diversity.keys() | items_score_accuracy.keys()  # 尽最大可能扩大新物品候选集C(u)
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None
    items_score_diversity_valid = {}
    items_score_accuracy_valid = {}
    # 填充最小值0.0
    for candidate in candidates:
        items_score_diversity_valid[candidate] = items_score_diversity.get(candidate, 0.0)
        items_score_accuracy_valid[candidate] = items_score_accuracy.get(candidate, 0.0)

    # 重排序。波达计数法
    # 排名值Rank属于[1, candidates_length]，和评分值Score一致，越大越好
    items_score_diversity_sorted = sorted(items_score_diversity_valid.items(), key=lambda a: a[1], reverse=True)
    items_score_accuracy_sorted = sorted(items_score_accuracy_valid.items(), key=lambda a: a[1], reverse=True)
    items_rank_diversity = {items_score_diversity_sorted[i][0]: (candidates_length - i) for i in
                            range(candidates_length)}
    items_rank_accuracy = {items_score_accuracy_sorted[i][0]: (candidates_length - i) for i in range(candidates_length)}

    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p
    items_score_reranking = {
        candidate: (norm_ku * items_rank_diversity[candidate] + (1 - norm_ku) * items_rank_accuracy[candidate])
        for candidate in candidates}

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def ucfq_icf_reranking_topsis(target_user, args):
    # 引入多样性推荐算法的重排序。TOPSIS指标方法。
    #
    # // 特别注意，推荐结果数目是不固定的。

    # # Debug
    # target_user = 113061

    items_score_diversity = cf.icf_knn_norm(target_user, args)  # 面向多样性
    items_score_accuracy = cf.ucf_q(target_user, args)  # 面向准确率

    # 定新物品候选集
    candidates = items_score_diversity.keys() | items_score_accuracy.keys()  # 尽最大可能扩大新物品候选集C(u)
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None
    items_score_diversity_valid = {}
    items_score_accuracy_valid = {}
    # 填充最小值0.0
    for candidate in candidates:
        items_score_diversity_valid[candidate] = items_score_diversity.get(candidate, 0.0)
        items_score_accuracy_valid[candidate] = items_score_accuracy.get(candidate, 0.0)

    # 重排序。TOPSIS指标方法
    # 评价值（评分值Score标准化）
    max_score_diversity = max(items_score_diversity_valid.values())
    min_score_diversity = min(items_score_diversity_valid.values())
    max_score_accuracy = max(items_score_accuracy_valid.values())
    min_score_accuracy = min(items_score_accuracy_valid.values())

    # boundary condition
    if math.fabs(max_score_diversity - min_score_diversity) < 1E-6:
        items_score_diversity_valid = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_diversity_valid = {candidate: ((items_score_diversity_valid[candidate] - min_score_diversity)
                                                   / (max_score_diversity - min_score_diversity))
                                       for candidate in candidates}

    # boundary condition
    if math.fabs(max_score_accuracy - min_score_accuracy) < 1E-6:
        items_score_accuracy_valid = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_accuracy_valid = {candidate: ((items_score_accuracy_valid[candidate] - min_score_accuracy)
                                                  / (max_score_accuracy - min_score_accuracy))
                                      for candidate in candidates}

    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p
    # 定权
    w_diversity = norm_ku
    w_accuracy = 1 - norm_ku
    # 得到了加权之后的评价值
    for candidate in candidates:
        items_score_diversity_valid[candidate] *= w_diversity
        items_score_accuracy_valid[candidate] *= w_accuracy

    # 寻找最优最劣解（正、负理想解）
    # 正
    positive_diversity = max(items_score_diversity_valid.values())
    positive_accuracy = max(items_score_accuracy_valid.values())
    # 负
    negative_diversity = min(items_score_diversity_valid.values())
    negative_accuracy = min(items_score_accuracy_valid.values())
    # boundary condition. 当最优解和最劣解相同时
    if positive_diversity == negative_diversity and positive_accuracy == negative_accuracy:
        items_score_reranking = {candidate: 1.0 for candidate in candidates}
        return items_score_reranking

    # 计算新物品与最优最劣解的欧式距离，并根据此得出贴近度（结果）
    items_score_reranking = {}  # 贴近度。越大越好，注意与python版本定义正好相反
    for candidate in candidates:
        rank_diversity = items_score_diversity_valid[candidate]
        rank_accuracy = items_score_accuracy_valid[candidate]

        distance_positive = (
            (rank_diversity - positive_diversity) * (rank_diversity - positive_diversity)
            + (rank_accuracy - positive_accuracy) * (rank_accuracy - positive_accuracy)
        )

        distance_negative = (
            (rank_diversity - negative_diversity) * (rank_diversity - negative_diversity)
            + (rank_accuracy - negative_accuracy) * (rank_accuracy - negative_accuracy)
        )

        # 靠近1说明靠近positive，靠近0说明靠近negtive
        res = distance_negative / (distance_positive + distance_negative)
        items_score_reranking[candidate] = res

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def ucfuv_icf_reranking_topsis(target_user, args):
    # 引入多样性推荐算法的重排序。TOPSIS指标方法。
    #
    # // 特别注意，推荐结果数目是不固定的。

    # # Debug
    # target_user = 113061

    items_score_diversity = cf.icf_knn_norm(target_user, args)  # 面向多样性
    items_score_accuracy = cf.ucf_userdegree_neighbordegree(target_user, args)  # 面向准确率

    # 定新物品候选集
    candidates = items_score_diversity.keys() | items_score_accuracy.keys()  # 尽最大可能扩大新物品候选集C(u)
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None
    items_score_diversity_valid = {}
    items_score_accuracy_valid = {}
    # 填充最小值0.0
    for candidate in candidates:
        items_score_diversity_valid[candidate] = items_score_diversity.get(candidate, 0.0)
        items_score_accuracy_valid[candidate] = items_score_accuracy.get(candidate, 0.0)

    # 重排序。TOPSIS指标方法
    # 评价值（评分值Score标准化）
    max_score_diversity = max(items_score_diversity_valid.values())
    min_score_diversity = min(items_score_diversity_valid.values())
    max_score_accuracy = max(items_score_accuracy_valid.values())
    min_score_accuracy = min(items_score_accuracy_valid.values())

    # boundary condition
    if math.fabs(max_score_diversity - min_score_diversity) < 1E-6:
        items_score_diversity_valid = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_diversity_valid = {candidate: ((items_score_diversity_valid[candidate] - min_score_diversity)
                                                   / (max_score_diversity - min_score_diversity))
                                       for candidate in candidates}

    # boundary condition
    if math.fabs(max_score_accuracy - min_score_accuracy) < 1E-6:
        items_score_accuracy_valid = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_accuracy_valid = {candidate: ((items_score_accuracy_valid[candidate] - min_score_accuracy)
                                                  / (max_score_accuracy - min_score_accuracy))
                                      for candidate in candidates}

    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p
    # 定权
    w_diversity = norm_ku
    w_accuracy = 1 - norm_ku
    # 得到了加权之后的评价值
    for candidate in candidates:
        items_score_diversity_valid[candidate] *= w_diversity
        items_score_accuracy_valid[candidate] *= w_accuracy

    # 寻找最优最劣解（正、负理想解）
    # 正
    positive_diversity = max(items_score_diversity_valid.values())
    positive_accuracy = max(items_score_accuracy_valid.values())
    # 负
    negative_diversity = min(items_score_diversity_valid.values())
    negative_accuracy = min(items_score_accuracy_valid.values())
    # boundary condition. 当最优解和最劣解相同时
    if positive_diversity == negative_diversity and positive_accuracy == negative_accuracy:
        items_score_reranking = {candidate: 1.0 for candidate in candidates}
        return items_score_reranking

    # 计算新物品与最优最劣解的欧式距离，并根据此得出贴近度（结果）
    items_score_reranking = {}  # 贴近度。越大越好，注意与python版本定义正好相反
    for candidate in candidates:
        rank_diversity = items_score_diversity_valid[candidate]
        rank_accuracy = items_score_accuracy_valid[candidate]

        distance_positive = (
            (rank_diversity - positive_diversity) * (rank_diversity - positive_diversity)
            + (rank_accuracy - positive_accuracy) * (rank_accuracy - positive_accuracy)
        )

        distance_negative = (
            (rank_diversity - negative_diversity) * (rank_diversity - negative_diversity)
            + (rank_accuracy - negative_accuracy) * (rank_accuracy - negative_accuracy)
        )

        # 靠近1说明靠近positive，靠近0说明靠近negtive
        res = distance_negative / (distance_positive + distance_negative)
        items_score_reranking[candidate] = res

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def probs_heats_reranking(target_user, args):
    # 引入多样性推荐算法的重排序。波达计数法（Borda Count）
    #
    # // 特别注意，推荐结果数目是不固定的。

    items_score_diversity = graph.heats(target_user, args)  # 面向多样性
    items_score_accuracy = graph.probs(target_user, args)  # 面向准确率

    # 定新物品候选集
    candidates = items_score_diversity.keys() | items_score_accuracy.keys()  # 尽最大可能扩大新物品候选集C(u)
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None
    items_score_diversity_valid = {}
    items_score_accuracy_valid = {}
    # 填充最小值0.0
    for candidate in candidates:
        items_score_diversity_valid[candidate] = items_score_diversity.get(candidate, 0.0)
        items_score_accuracy_valid[candidate] = items_score_accuracy.get(candidate, 0.0)

    # 重排序。波达计数法
    # 排名值Rank属于[1, candidates_length]，和评分值Score一致，越大越好
    items_score_diversity_sorted = sorted(items_score_diversity_valid.items(), key=lambda a: a[1], reverse=True)
    items_score_accuracy_sorted = sorted(items_score_accuracy_valid.items(), key=lambda a: a[1], reverse=True)
    items_rank_diversity = {items_score_diversity_sorted[i][0]: (candidates_length - i) for i in
                            range(candidates_length)}
    items_rank_accuracy = {items_score_accuracy_sorted[i][0]: (candidates_length - i) for i in range(candidates_length)}

    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p
    items_score_reranking = {
    candidate: (norm_ku * items_rank_diversity[candidate] + (1 - norm_ku) * items_rank_accuracy[candidate])
    for candidate in candidates}

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def probs_heats_reranking_topsis(target_user, args):
    # 引入多样性推荐算法的重排序。TOPSIS指标方法。
    #
    # // 特别注意，推荐结果数目是不固定的。

    items_score_diversity = graph.heats(target_user, args)  # 面向多样性
    items_score_accuracy = graph.probs(target_user, args)  # 面向准确率

    # 定新物品候选集
    candidates = items_score_diversity.keys() | items_score_accuracy.keys()  # 尽最大可能扩大新物品候选集C(u)
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None
    items_score_diversity_valid = {}
    items_score_accuracy_valid = {}
    # 填充最小值0.0
    for candidate in candidates:
        items_score_diversity_valid[candidate] = items_score_diversity.get(candidate, 0.0)
        items_score_accuracy_valid[candidate] = items_score_accuracy.get(candidate, 0.0)

    # 重排序。TOPSIS指标方法
    # 评价值（评分值Score标准化）
    max_score_diversity = max(items_score_diversity_valid.values())
    min_score_diversity = min(items_score_diversity_valid.values())
    max_score_accuracy = max(items_score_accuracy_valid.values())
    min_score_accuracy = min(items_score_accuracy_valid.values())

    # boundary condition
    if math.fabs(max_score_diversity - min_score_diversity) < 1E-6:
        items_score_diversity_valid = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_diversity_valid = {candidate: ((items_score_diversity_valid[candidate] - min_score_diversity)
                                                   / (max_score_diversity - min_score_diversity))
                                       for candidate in candidates}

    # boundary condition
    if math.fabs(max_score_accuracy - min_score_accuracy) < 1E-6:
        items_score_accuracy_valid = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_accuracy_valid = {candidate: ((items_score_accuracy_valid[candidate] - min_score_accuracy)
                                                  / (max_score_accuracy - min_score_accuracy))
                                      for candidate in candidates}

    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p
    # 定权
    w_diversity = norm_ku
    w_accuracy = 1 - norm_ku
    # 得到了加权之后的评价值
    for candidate in candidates:
        items_score_diversity_valid[candidate] *= w_diversity
        items_score_accuracy_valid[candidate] *= w_accuracy

    # 寻找最优最劣解（正、负理想解）
    # 正
    positive_diversity = max(items_score_diversity_valid.values())
    positive_accuracy = max(items_score_accuracy_valid.values())
    # 负
    negative_diversity = min(items_score_diversity_valid.values())
    negative_accuracy = min(items_score_accuracy_valid.values())
    # boundary condition. 当最优解和最劣解相同时
    if positive_diversity == negative_diversity and positive_accuracy == negative_accuracy:
        items_score_reranking = {candidate: 1.0 for candidate in candidates}
        return items_score_reranking

    # 计算新物品与最优最劣解的欧式距离，并根据此得出贴近度（结果）
    items_score_reranking = {}  # 贴近度。越大越好，注意与python版本定义正好相反
    for candidate in candidates:
        rank_diversity = items_score_diversity_valid[candidate]
        rank_accuracy = items_score_accuracy_valid[candidate]

        distance_positive = (
            (rank_diversity - positive_diversity) * (rank_diversity - positive_diversity)
            + (rank_accuracy - positive_accuracy) * (rank_accuracy - positive_accuracy)
        )

        distance_negative = (
            (rank_diversity - negative_diversity) * (rank_diversity - negative_diversity)
            + (rank_accuracy - negative_accuracy) * (rank_accuracy - negative_accuracy)
        )

        # 靠近1说明靠近positive，靠近0说明靠近negtive
        res = distance_negative / (distance_positive + distance_negative)
        items_score_reranking[candidate] = res

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def probsb_heats_reranking_topsis(target_user, args):
    # 引入多样性推荐算法的重排序。TOPSIS指标方法。
    #
    # // 特别注意，推荐结果数目是不固定的。

    items_score_diversity = graph.heats(target_user, args)  # 面向多样性
    items_score_accuracy = graph.probs_randomwalk(target_user, args)  # 面向准确率

    # 定新物品候选集
    candidates = items_score_diversity.keys() | items_score_accuracy.keys()  # 尽最大可能扩大新物品候选集C(u)
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None
    items_score_diversity_valid = {}
    items_score_accuracy_valid = {}
    # 填充最小值0.0
    for candidate in candidates:
        items_score_diversity_valid[candidate] = items_score_diversity.get(candidate, 0.0)
        items_score_accuracy_valid[candidate] = items_score_accuracy.get(candidate, 0.0)

    # 重排序。TOPSIS指标方法
    # 评价值（评分值Score标准化）
    max_score_diversity = max(items_score_diversity_valid.values())
    min_score_diversity = min(items_score_diversity_valid.values())
    max_score_accuracy = max(items_score_accuracy_valid.values())
    min_score_accuracy = min(items_score_accuracy_valid.values())

    # boundary condition
    if math.fabs(max_score_diversity - min_score_diversity) < 1E-6:
        items_score_diversity_valid = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_diversity_valid = {candidate: ((items_score_diversity_valid[candidate] - min_score_diversity)
                                                   / (max_score_diversity - min_score_diversity))
                                       for candidate in candidates}

    # boundary condition
    if math.fabs(max_score_accuracy - min_score_accuracy) < 1E-6:
        items_score_accuracy_valid = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_accuracy_valid = {candidate: ((items_score_accuracy_valid[candidate] - min_score_accuracy)
                                                  / (max_score_accuracy - min_score_accuracy))
                                      for candidate in candidates}

    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p
    # 定权
    w_diversity = norm_ku
    w_accuracy = 1 - norm_ku
    # 得到了加权之后的评价值
    for candidate in candidates:
        items_score_diversity_valid[candidate] *= w_diversity
        items_score_accuracy_valid[candidate] *= w_accuracy

    # 寻找最优最劣解（正、负理想解）
    # 正
    positive_diversity = max(items_score_diversity_valid.values())
    positive_accuracy = max(items_score_accuracy_valid.values())
    # 负
    negative_diversity = min(items_score_diversity_valid.values())
    negative_accuracy = min(items_score_accuracy_valid.values())
    # boundary condition. 当最优解和最劣解相同时
    if positive_diversity == negative_diversity and positive_accuracy == negative_accuracy:
        items_score_reranking = {candidate: 1.0 for candidate in candidates}
        return items_score_reranking

    # 计算新物品与最优最劣解的欧式距离，并根据此得出贴近度（结果）
    items_score_reranking = {}  # 贴近度。越大越好，注意与python版本定义正好相反
    for candidate in candidates:
        rank_diversity = items_score_diversity_valid[candidate]
        rank_accuracy = items_score_accuracy_valid[candidate]

        distance_positive = (
            (rank_diversity - positive_diversity) * (rank_diversity - positive_diversity)
            + (rank_accuracy - positive_accuracy) * (rank_accuracy - positive_accuracy)
        )

        distance_negative = (
            (rank_diversity - negative_diversity) * (rank_diversity - negative_diversity)
            + (rank_accuracy - negative_accuracy) * (rank_accuracy - negative_accuracy)
        )

        # 靠近1说明靠近positive，靠近0说明靠近negtive
        res = distance_negative / (distance_positive + distance_negative)
        items_score_reranking[candidate] = res

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def ucfq_reranking_di_borda(target_user, args):
    # 引入反向推荐思想的重排序。波达计数法。
    #
    # // 特别注意，推荐结果数目是不固定的。

    # 正向推荐
    items_score_direct = cf.ucf_q(target_user, args)
    # 定新物品候选集
    candidates = items_score_direct.keys()
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None

    # 反向推荐
    items_score_inverse = {}
    # 种子用户
    items_users_score = args.get('items_users_score', None)
    # 生成结果
    for (item_new, score_old) in items_score_direct.items():
        # 为每一个item_new做更新
        # boundary condition
        if item_new not in items_users_score:
            items_score_inverse[item_new] = score_old  # 不变
            continue
        max_seed = max(items_users_score[item_new].values())
        max_scores = max([max_seed, score_old])

        items_score_inverse[item_new] = (score_old) / (max_scores)

    # 重排序。波达计数法
    # 排名值Rank属于[1, candidates_length]，和评分值Score一致，越大越好
    items_score_direct_sorted = sorted(items_score_direct.items(), key=lambda a: a[1], reverse=True)
    items_score_inverse_sorted = sorted(items_score_inverse.items(), key=lambda a: a[1], reverse=True)
    items_rank_direct = {items_score_direct_sorted[i][0]: (candidates_length - i) for i in range(candidates_length)}
    items_rank_inverse = {items_score_inverse_sorted[i][0]: (candidates_length - i) for i in range(candidates_length)}

    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p
    items_score_reranking = {
        candidate: ((1 - norm_ku) * items_rank_direct[candidate] + norm_ku * items_rank_inverse[candidate])
        for candidate in candidates}

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def ucfq_reranking_di_topsis(target_user, args):
    # 引入反向推荐思想的重排序。TOPSIS指标方法。
    #
    # // 特别注意，推荐结果数目是不固定的。

    # 正向推荐
    items_score_direct = cf.ucf_q(target_user, args)
    # 定新物品候选集
    candidates = items_score_direct.keys()
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None

    # 反向推荐
    items_score_inverse = {}
    # 种子用户
    items_users_score = args.get('items_users_score', None)
    # 生成结果
    for (item_new, score_old) in items_score_direct.items():
        # 为每一个item_new做更新
        # boundary condition
        if item_new not in items_users_score:
            items_score_inverse[item_new] = score_old  # 不变
            continue
        max_seed = max(items_users_score[item_new].values())
        max_scores = max([max_seed, score_old])
        items_score_inverse[item_new] = score_old / max_scores

    # 重排序。TOPSIS指标方法
    # 只需要对正向推荐的评分值Score标准化
    max_score_direct = max(items_score_direct.values())
    min_score_direct = min(items_score_direct.values())
    # boundary condition
    if math.fabs(max_score_direct - min_score_direct) < 1E-6:
        items_score_direct = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_direct = {candidate: ((items_score_direct[candidate] - min_score_direct)
                                          / (max_score_direct - min_score_direct))
                              for candidate in candidates}

    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p
    # 定权
    w_direct = 1 - norm_ku
    w_inverse = norm_ku
    # 得到了加权之后的评价值
    for candidate in candidates:
        items_score_direct[candidate] *= w_direct
        items_score_inverse[candidate] *= w_inverse

    # 寻找最优最劣解（正、负理想解）
    # 正
    positive_direct = max(items_score_direct.values())
    positive_inverse = max(items_score_inverse.values())
    # 负
    negative_direct = min(items_score_direct.values())
    negative_inverse = min(items_score_inverse.values())
    # boundary condition. 当最优解和最劣解相同时
    if positive_direct == negative_direct and positive_inverse == negative_inverse:
        items_score_reranking = {candidate: 1.0 for candidate in candidates}
        return items_score_reranking

    # 计算新物品与最优最劣解的欧式距离，并根据此得出贴近度（结果）
    items_score_reranking = {}  # 贴近度。越大越好，注意与python版本定义正好相反
    for candidate in candidates:
        rank_direct = items_score_direct[candidate]
        rank_inverse = items_score_inverse[candidate]

        distance_positive = (
            (rank_direct - positive_direct) * (rank_direct - positive_direct)
            + (rank_inverse - positive_inverse) * (rank_inverse - positive_inverse)
        )

        distance_negative = (
            (rank_direct - negative_direct) * (rank_direct - negative_direct)
            + (rank_inverse - negative_inverse) * (rank_inverse - negative_inverse)
        )

        # 靠近1说明靠近positive，靠近0说明靠近negtive
        res = distance_negative / (distance_positive + distance_negative)
        items_score_reranking[candidate] = res

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def ucfq_reranking_di_topsis_weight(target_user, args):
    # 引入反向推荐思想的重排序。TOPSIS指标方法。
    #
    # // 特别注意，推荐结果数目是不固定的。

    # 正向推荐
    items_score_direct = cf.ucf_q(target_user, args)
    # 定新物品候选集
    candidates = items_score_direct.keys()
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None

    # 反向推荐
    items_score_inverse = {}
    # 种子用户
    items_users_score = args.get('items_users_score', None)
    # 生成结果
    for (item_new, score_old) in items_score_direct.items():
        # 为每一个item_new做更新
        # boundary condition
        if item_new not in items_users_score:
            items_score_inverse[item_new] = score_old  # 不变
            continue
        max_seed = max(items_users_score[item_new].values())
        max_scores = max([max_seed, score_old])
        items_score_inverse[item_new] = score_old / max_scores

    # 重排序。TOPSIS指标方法
    # 只需要对正向推荐的评分值Score标准化
    max_score_direct = max(items_score_direct.values())
    min_score_direct = min(items_score_direct.values())
    # boundary condition
    if math.fabs(max_score_direct - min_score_direct) < 1E-6:
        items_score_direct = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_direct = {candidate: ((items_score_direct[candidate] - min_score_direct)
                                          / (max_score_direct - min_score_direct))
                              for candidate in candidates}

    # 定权
    w_inverse = args.get('weight', None)
    w_direct = 1 - w_inverse
    # 得到了加权之后的评价值
    for candidate in candidates:
        items_score_direct[candidate] *= w_direct
        items_score_inverse[candidate] *= w_inverse

    # 寻找最优最劣解（正、负理想解）
    # 正
    positive_direct = max(items_score_direct.values())
    positive_inverse = max(items_score_inverse.values())
    # 负
    negative_direct = min(items_score_direct.values())
    negative_inverse = min(items_score_inverse.values())
    # boundary condition. 当最优解和最劣解相同时
    if positive_direct == negative_direct and positive_inverse == negative_inverse:
        items_score_reranking = {candidate: 1.0 for candidate in candidates}
        return items_score_reranking

    # 计算新物品与最优最劣解的欧式距离，并根据此得出贴近度（结果）
    items_score_reranking = {}  # 贴近度。越大越好，注意与python版本定义正好相反
    for candidate in candidates:
        rank_direct = items_score_direct[candidate]
        rank_inverse = items_score_inverse[candidate]

        distance_positive = (
            (rank_direct - positive_direct) * (rank_direct - positive_direct)
            + (rank_inverse - positive_inverse) * (rank_inverse - positive_inverse)
        )

        distance_negative = (
            (rank_direct - negative_direct) * (rank_direct - negative_direct)
            + (rank_inverse - negative_inverse) * (rank_inverse - negative_inverse)
        )

        # 靠近1说明靠近positive，靠近0说明靠近negtive
        res = distance_negative / (distance_positive + distance_negative)
        items_score_reranking[candidate] = res

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def ucfu_reranking_di_topsis(target_user, args):
    # 引入反向推荐思想的重排序。TOPSIS指标方法。
    #
    # // 特别注意，推荐结果数目是不固定的。

    # 正向推荐
    items_score_direct = cf.ucf_userdegree(target_user, args)
    # 定新物品候选集
    candidates = items_score_direct.keys()
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None

    # 反向推荐
    items_score_inverse = {}
    # 种子用户
    items_users_score = args.get('items_users_score', None)
    # 生成结果
    for (item_new, score_old) in items_score_direct.items():
        # 为每一个item_new做更新
        # boundary condition
        if item_new not in items_users_score:
            items_score_inverse[item_new] = score_old  # 不变
            continue
        max_seed = max(items_users_score[item_new].values())
        max_scores = max([max_seed, score_old])

        items_score_inverse[item_new] = (score_old) / (max_scores)

    # 重排序。TOPSIS指标方法
    # 只需要对正向推荐的评分值Score标准化
    max_score_direct = max(items_score_direct.values())
    min_score_direct = min(items_score_direct.values())
    # boundary condition
    if math.fabs(max_score_direct - min_score_direct) < 1E-6:
        items_score_direct = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_direct = {candidate: ((items_score_direct[candidate] - min_score_direct)
                                          / (max_score_direct - min_score_direct))
                              for candidate in candidates}

    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p
    # 定权
    w_direct = 1 - norm_ku
    w_inverse = norm_ku
    # 得到了加权之后的评价值
    for candidate in candidates:
        items_score_direct[candidate] *= w_direct
        items_score_inverse[candidate] *= w_inverse

    # 寻找最优最劣解（正、负理想解）
    # 正
    positive_direct = max(items_score_direct.values())
    positive_inverse = max(items_score_inverse.values())
    # 负
    negative_direct = min(items_score_direct.values())
    negative_inverse = min(items_score_inverse.values())
    # boundary condition. 当最优解和最劣解相同时
    if positive_direct == negative_direct and positive_inverse == negative_inverse:
        items_score_reranking = {candidate: 1.0 for candidate in candidates}
        return items_score_reranking

    # 计算新物品与最优最劣解的欧式距离，并根据此得出贴近度（结果）
    items_score_reranking = {}  # 贴近度。越大越好，注意与python版本定义正好相反
    for candidate in candidates:
        rank_direct = items_score_direct[candidate]
        rank_inverse = items_score_inverse[candidate]

        distance_positive = (
            (rank_direct - positive_direct) * (rank_direct - positive_direct)
            + (rank_inverse - positive_inverse) * (rank_inverse - positive_inverse)
        )

        distance_negative = (
            (rank_direct - negative_direct) * (rank_direct - negative_direct)
            + (rank_inverse - negative_inverse) * (rank_inverse - negative_inverse)
        )

        # 靠近1说明靠近positive，靠近0说明靠近negtive
        res = distance_negative / (distance_positive + distance_negative)
        items_score_reranking[candidate] = res

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def ucfuv_reranking_di_topsis(target_user, args):
    # 引入反向推荐思想的重排序。TOPSIS指标方法。
    #
    # // 特别注意，推荐结果数目是不固定的。

    # 正向推荐
    items_score_direct = cf.ucf_userdegree_neighbordegree(target_user, args)
    # 定新物品候选集
    candidates = items_score_direct.keys()
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None

    # 反向推荐
    items_score_inverse = {}
    # 种子用户
    items_users_score = args.get('items_users_score', None)
    # 生成结果
    for (item_new, score_old) in items_score_direct.items():
        # 为每一个item_new做更新
        # boundary condition
        if item_new not in items_users_score:
            items_score_inverse[item_new] = score_old  # 不变
            continue
        max_seed = max(items_users_score[item_new].values())
        max_scores = max([max_seed, score_old])

        items_score_inverse[item_new] = (score_old) / (max_scores)

    # 重排序。TOPSIS指标方法
    # 只需要对正向推荐的评分值Score标准化
    max_score_direct = max(items_score_direct.values())
    min_score_direct = min(items_score_direct.values())
    # boundary condition
    if math.fabs(max_score_direct - min_score_direct) < 1E-6:
        items_score_direct = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_direct = {candidate: ((items_score_direct[candidate] - min_score_direct)
                                          / (max_score_direct - min_score_direct))
                              for candidate in candidates}

    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p
    # 定权
    w_direct = 1 - norm_ku
    w_inverse = norm_ku
    # 得到了加权之后的评价值
    for candidate in candidates:
        items_score_direct[candidate] *= w_direct
        items_score_inverse[candidate] *= w_inverse

    # 寻找最优最劣解（正、负理想解）
    # 正
    positive_direct = max(items_score_direct.values())
    positive_inverse = max(items_score_inverse.values())
    # 负
    negative_direct = min(items_score_direct.values())
    negative_inverse = min(items_score_inverse.values())
    # boundary condition. 当最优解和最劣解相同时
    if positive_direct == negative_direct and positive_inverse == negative_inverse:
        items_score_reranking = {candidate: 1.0 for candidate in candidates}
        return items_score_reranking

    # 计算新物品与最优最劣解的欧式距离，并根据此得出贴近度（结果）
    items_score_reranking = {}  # 贴近度。越大越好，注意与python版本定义正好相反
    for candidate in candidates:
        rank_direct = items_score_direct[candidate]
        rank_inverse = items_score_inverse[candidate]

        distance_positive = (
            (rank_direct - positive_direct) * (rank_direct - positive_direct)
            + (rank_inverse - positive_inverse) * (rank_inverse - positive_inverse)
        )

        distance_negative = (
            (rank_direct - negative_direct) * (rank_direct - negative_direct)
            + (rank_inverse - negative_inverse) * (rank_inverse - negative_inverse)
        )

        # 靠近1说明靠近positive，靠近0说明靠近negtive
        res = distance_negative / (distance_positive + distance_negative)
        items_score_reranking[candidate] = res

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def icfknnnorm_reranking_di_borda(target_user, args):
    # 引入反向推荐思想的重排序。波达计数法。
    #
    # // 特别注意，推荐结果数目是不固定的。

    # 正向推荐
    items_score_direct = cf.icf_knn_norm(target_user, args)
    # 定新物品候选集
    candidates = items_score_direct.keys()
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None

    # 反向推荐
    items_score_inverse = {}
    # 种子用户
    items_users_score = args.get('items_users_score', None)
    # 生成结果
    for (item_new, score_old) in items_score_direct.items():
        # 为每一个item_new做更新
        # boundary condition
        if item_new not in items_users_score:
            items_score_inverse[item_new] = score_old  # 不变
            continue
        max_seed = max(items_users_score[item_new].values())
        max_scores = max([max_seed, score_old])

        items_score_inverse[item_new] = (score_old) / (max_scores)

    # 重排序。波达计数法
    # 排名值Rank属于[1, candidates_length]，和评分值Score一致，越大越好
    items_score_direct_sorted = sorted(items_score_direct.items(), key=lambda a: a[1], reverse=True)
    items_score_inverse_sorted = sorted(items_score_inverse.items(), key=lambda a: a[1], reverse=True)
    items_rank_direct = {items_score_direct_sorted[i][0]: (candidates_length - i) for i in range(candidates_length)}
    items_rank_inverse = {items_score_inverse_sorted[i][0]: (candidates_length - i) for i in range(candidates_length)}

    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p
    items_score_reranking = {
        candidate: ((1 - norm_ku) * items_rank_direct[candidate] + norm_ku * items_rank_inverse[candidate])
        for candidate in candidates}

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def icfknnnorm_reranking_di_topsis(target_user, args):
    # 引入反向推荐思想的重排序。TOPSIS指标方法。
    #
    # // 特别注意，推荐结果数目是不固定的。

    # 正向推荐
    items_score_direct = cf.icf_knn_norm(target_user, args)
    # 定新物品候选集
    candidates = items_score_direct.keys()
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None

    # 反向推荐
    items_score_inverse = {}
    # 种子用户
    items_users_score = args.get('items_users_score', None)
    # 生成结果
    for (item_new, score_old) in items_score_direct.items():
        # 为每一个item_new做更新
        # boundary condition
        if item_new not in items_users_score:
            items_score_inverse[item_new] = score_old  # 不变
            continue
        max_seed = max(items_users_score[item_new].values())
        max_scores = max([max_seed, score_old])
        items_score_inverse[item_new] = score_old / max_scores

    # 重排序。TOPSIS指标方法
    # 只需要对正向推荐的评分值Score标准化
    max_score_direct = max(items_score_direct.values())
    min_score_direct = min(items_score_direct.values())
    # boundary condition
    if math.fabs(max_score_direct - min_score_direct) < 1E-6:
        items_score_direct = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_direct = {candidate: ((items_score_direct[candidate] - min_score_direct)
                                          / (max_score_direct - min_score_direct))
                              for candidate in candidates}

    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p
    # 定权
    w_direct = 1 - norm_ku
    w_inverse = norm_ku
    # 得到了加权之后的评价值
    for candidate in candidates:
        items_score_direct[candidate] *= w_direct
        items_score_inverse[candidate] *= w_inverse

    # 寻找最优最劣解（正、负理想解）
    # 正
    positive_direct = max(items_score_direct.values())
    positive_inverse = max(items_score_inverse.values())
    # 负
    negative_direct = min(items_score_direct.values())
    negative_inverse = min(items_score_inverse.values())

    # 计算新物品与最优最劣解的欧式距离，并根据此得出贴近度（结果）
    # boundary condition. 当最优解和最劣解相同时
    if positive_direct == negative_direct and positive_inverse == negative_inverse:
        items_score_reranking = {candidate: 1.0 for candidate in candidates}
        return items_score_reranking

    items_score_reranking = {}  # 贴近度。越大越好，注意与python版本定义正好相反
    for candidate in candidates:
        rank_direct = items_score_direct[candidate]
        rank_inverse = items_score_inverse[candidate]

        distance_positive = (
            (rank_direct - positive_direct) * (rank_direct - positive_direct)
            + (rank_inverse - positive_inverse) * (rank_inverse - positive_inverse)
        )

        distance_negative = (
            (rank_direct - negative_direct) * (rank_direct - negative_direct)
            + (rank_inverse - negative_inverse) * (rank_inverse - negative_inverse)
        )

        # 靠近1说明靠近positive，靠近0说明靠近negtive
        res = distance_negative / (distance_positive + distance_negative)
        items_score_reranking[candidate] = res

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def icfj_reranking_di_topsis(target_user, args):
    # 引入反向推荐思想的重排序。TOPSIS指标方法。
    #
    # // 特别注意，推荐结果数目是不固定的。

    # 正向推荐
    items_score_direct = cf.icf_knn_norm_itemdegree(target_user, args)
    # 定新物品候选集
    candidates = items_score_direct.keys()
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None

    # 反向推荐
    items_score_inverse = {}
    # 种子用户
    items_users_score = args.get('items_users_score', None)
    # 生成结果
    for (item_new, score_old) in items_score_direct.items():
        # 为每一个item_new做更新
        # boundary condition
        if item_new not in items_users_score:
            items_score_inverse[item_new] = score_old  # 不变
            continue
        max_seed = max(items_users_score[item_new].values())
        max_scores = max([max_seed, score_old])

        items_score_inverse[item_new] = (score_old) / (max_scores)

    # 重排序。TOPSIS指标方法
    # 只需要对正向推荐的评分值Score标准化
    max_score_direct = max(items_score_direct.values())
    min_score_direct = min(items_score_direct.values())
    # boundary condition
    if math.fabs(max_score_direct - min_score_direct) < 1E-6:
        items_score_direct = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_direct = {candidate: ((items_score_direct[candidate] - min_score_direct)
                                          / (max_score_direct - min_score_direct))
                              for candidate in candidates}

    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p
    # 定权
    w_direct = 1 - norm_ku
    w_inverse = norm_ku
    # 得到了加权之后的评价值
    for candidate in candidates:
        items_score_direct[candidate] *= w_direct
        items_score_inverse[candidate] *= w_inverse

    # 寻找最优最劣解（正、负理想解）
    # 正
    positive_direct = max(items_score_direct.values())
    positive_inverse = max(items_score_inverse.values())
    # 负
    negative_direct = min(items_score_direct.values())
    negative_inverse = min(items_score_inverse.values())

    # 计算新物品与最优最劣解的欧式距离，并根据此得出贴近度（结果）
    # boundary condition. 当最优解和最劣解相同时
    if positive_direct == negative_direct and positive_inverse == negative_inverse:
        items_score_reranking = {candidate: 1.0 for candidate in candidates}
        return items_score_reranking

    items_score_reranking = {}  # 贴近度。越大越好，注意与python版本定义正好相反
    for candidate in candidates:
        rank_direct = items_score_direct[candidate]
        rank_inverse = items_score_inverse[candidate]

        distance_positive = (
            (rank_direct - positive_direct) * (rank_direct - positive_direct)
            + (rank_inverse - positive_inverse) * (rank_inverse - positive_inverse)
        )

        distance_negative = (
            (rank_direct - negative_direct) * (rank_direct - negative_direct)
            + (rank_inverse - negative_inverse) * (rank_inverse - negative_inverse)
        )

        # 靠近1说明靠近positive，靠近0说明靠近negtive
        res = distance_negative / (distance_positive + distance_negative)
        items_score_reranking[candidate] = res

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def probs_reranking_di_borda(target_user, args):
    # 引入反向推荐思想的重排序。波达计数法。
    #
    # // 特别注意，推荐结果数目是不固定的。

    # 正向推荐
    items_score_direct = graph.probs(target_user, args)
    # 定新物品候选集
    candidates = items_score_direct.keys()
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None

    # 反向推荐
    items_score_inverse = {}
    # 种子用户
    items_users_score = args.get('items_users_score', None)
    # 生成结果
    for (item_new, score_old) in items_score_direct.items():
        # 为每一个item_new做更新
        # boundary condition
        if item_new not in items_users_score:
            items_score_inverse[item_new] = score_old  # 不变
            continue
        max_seed = max(items_users_score[item_new].values())
        max_scores = max([max_seed, score_old])

        items_score_inverse[item_new] = (score_old) / (max_scores)

    # 重排序。波达计数法
    # 排名值Rank属于[1, candidates_length]，和评分值Score一致，越大越好
    items_score_direct_sorted = sorted(items_score_direct.items(), key=lambda a: a[1], reverse=True)
    items_score_inverse_sorted = sorted(items_score_inverse.items(), key=lambda a: a[1], reverse=True)
    items_rank_direct = {items_score_direct_sorted[i][0]: (candidates_length - i) for i in range(candidates_length)}
    items_rank_inverse = {items_score_inverse_sorted[i][0]: (candidates_length - i) for i in range(candidates_length)}

    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p
    items_score_reranking = {
        candidate: ((1 - norm_ku) * items_rank_direct[candidate] + norm_ku * items_rank_inverse[candidate])
        for candidate in candidates}

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def probs_reranking_di_topsis(target_user, args):
    # 引入反向推荐思想的重排序。TOPSIS指标方法。
    #
    # // 特别注意，推荐结果数目是不固定的。

    # 正向推荐
    items_score_direct = graph.probs(target_user, args)
    # 定新物品候选集
    candidates = items_score_direct.keys()
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None

    # 反向推荐
    items_score_inverse = {}
    # 种子用户
    items_users_score = args.get('items_users_score', None)
    # 生成结果
    for (item_new, score_old) in items_score_direct.items():
        # 为每一个item_new做更新
        # boundary condition
        if item_new not in items_users_score:
            items_score_inverse[item_new] = score_old  # 不变
            continue
        max_seed = max(items_users_score[item_new].values())
        max_scores = max([max_seed, score_old])
        items_score_inverse[item_new] = score_old / max_scores

    # 重排序。TOPSIS指标方法
    # 只需要对正向推荐的评分值Score标准化
    max_score_direct = max(items_score_direct.values())
    min_score_direct = min(items_score_direct.values())
    # boundary condition
    if math.fabs(max_score_direct - min_score_direct) < 1E-6:
        items_score_direct = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_direct = {candidate: ((items_score_direct[candidate] - min_score_direct)
                                          / (max_score_direct - min_score_direct))
                              for candidate in candidates}

    # 定权

    # 得到了加权之后的评价值
    for candidate in candidates:
        items_score_direct[candidate] *= w_direct
        items_score_inverse[candidate] *= w_inverse

    # 寻找最优最劣解（正、负理想解）
    # 正
    positive_direct = max(items_score_direct.values())
    positive_inverse = max(items_score_inverse.values())
    # 负
    negative_direct = min(items_score_direct.values())
    negative_inverse = min(items_score_inverse.values())
    # boundary condition. 当最优解和最劣解相同时
    if positive_direct == negative_direct and positive_inverse == negative_inverse:
        items_score_reranking = {candidate: 1.0 for candidate in candidates}
        return items_score_reranking

    # 计算新物品与最优最劣解的欧式距离，并根据此得出贴近度（结果）
    items_score_reranking = {}  # 贴近度。越大越好，注意与python版本定义正好相反
    for candidate in candidates:
        rank_direct = items_score_direct[candidate]
        rank_inverse = items_score_inverse[candidate]

        distance_positive = (
            (rank_direct - positive_direct) * (rank_direct - positive_direct)
            + (rank_inverse - positive_inverse) * (rank_inverse - positive_inverse)
        )

        distance_negative = (
            (rank_direct - negative_direct) * (rank_direct - negative_direct)
            + (rank_inverse - negative_inverse) * (rank_inverse - negative_inverse)
        )

        # 靠近1说明靠近positive，靠近0说明靠近negtive
        res = distance_negative / (distance_positive + distance_negative)
        items_score_reranking[candidate] = res

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def probs_reranking_di_topsis_weight(target_user, args):
    # 引入反向推荐思想的重排序。TOPSIS指标方法。
    #
    # // 特别注意，推荐结果数目是不固定的。

    # 正向推荐
    items_score_direct = graph.probs(target_user, args)
    # 定新物品候选集
    candidates = items_score_direct.keys()
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None

    # 反向推荐
    items_score_inverse = {}
    # 种子用户
    items_users_score = args.get('items_users_score', None)
    # 生成结果
    for (item_new, score_old) in items_score_direct.items():
        # 为每一个item_new做更新
        # boundary condition
        if item_new not in items_users_score:
            items_score_inverse[item_new] = score_old  # 不变
            continue
        max_seed = max(items_users_score[item_new].values())
        max_scores = max([max_seed, score_old])
        items_score_inverse[item_new] = score_old / max_scores

    # 重排序。TOPSIS指标方法
    # 只需要对正向推荐的评分值Score标准化
    max_score_direct = max(items_score_direct.values())
    min_score_direct = min(items_score_direct.values())
    # boundary condition
    if math.fabs(max_score_direct - min_score_direct) < 1E-6:
        items_score_direct = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_direct = {candidate: ((items_score_direct[candidate] - min_score_direct)
                                          / (max_score_direct - min_score_direct))
                              for candidate in candidates}

    # 定权
    w_inverse = args.get('weight', None)
    w_direct = 1 - w_inverse
    # 得到了加权之后的评价值
    for candidate in candidates:
        items_score_direct[candidate] *= w_direct
        items_score_inverse[candidate] *= w_inverse

    # 寻找最优最劣解（正、负理想解）
    # 正
    positive_direct = max(items_score_direct.values())
    positive_inverse = max(items_score_inverse.values())
    # 负
    negative_direct = min(items_score_direct.values())
    negative_inverse = min(items_score_inverse.values())
    # boundary condition. 当最优解和最劣解相同时
    if positive_direct == negative_direct and positive_inverse == negative_inverse:
        items_score_reranking = {candidate: 1.0 for candidate in candidates}
        return items_score_reranking

    # 计算新物品与最优最劣解的欧式距离，并根据此得出贴近度（结果）
    items_score_reranking = {}  # 贴近度。越大越好，注意与python版本定义正好相反
    for candidate in candidates:
        rank_direct = items_score_direct[candidate]
        rank_inverse = items_score_inverse[candidate]

        distance_positive = (
            (rank_direct - positive_direct) * (rank_direct - positive_direct)
            + (rank_inverse - positive_inverse) * (rank_inverse - positive_inverse)
        )

        distance_negative = (
            (rank_direct - negative_direct) * (rank_direct - negative_direct)
            + (rank_inverse - negative_inverse) * (rank_inverse - negative_inverse)
        )

        # 靠近1说明靠近positive，靠近0说明靠近negtive
        res = distance_negative / (distance_positive + distance_negative)
        items_score_reranking[candidate] = res

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的


def probsb_reranking_di_topsis(target_user, args):
    # 引入反向推荐思想的重排序。TOPSIS指标方法。
    #
    # // 特别注意，推荐结果数目是不固定的。

    # 正向推荐
    items_score_direct = graph.probs_randomwalk(target_user, args)
    # 定新物品候选集
    candidates = items_score_direct.keys()
    candidates_length = len(candidates)
    # boundary condition
    if candidates_length == 0:
        return None

    # 反向推荐
    items_score_inverse = {}

    # 种子用户
    items_users_score = args.get('items_users_score', None)

    # 生成结果
    for (item_new, score_old) in items_score_direct.items():
        # 为每一个item_new做更新
        # boundary condition
        if item_new not in items_users_score:
            items_score_inverse[item_new] = score_old  # 不变
            continue
        max_seed = max(items_users_score[item_new].values())
        max_scores = max(max_seed, score_old)
        items_score_inverse[item_new] = (score_old) / (max_scores)

    # 重排序。TOPSIS指标方法
    # 只需要对正向推荐的评分值Score标准化
    max_score_direct = max(items_score_direct.values())
    min_score_direct = min(items_score_direct.values())
    # boundary condition
    if math.fabs(max_score_direct - min_score_direct) < 1E-6:
        items_score_direct = {candidate: 1.0 for candidate in candidates}
    else:
        items_score_direct = {candidate: ((items_score_direct[candidate] - min_score_direct)
                                          / (max_score_direct - min_score_direct))
                              for candidate in candidates}

    users_degree = args.get('users_degree', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_ku = ku / max_userdegreelog1p
    # 定权
    w_direct = 1 - norm_ku
    w_inverse = norm_ku
    # 得到了加权之后的评价值
    for candidate in candidates:
        items_score_direct[candidate] *= w_direct
        items_score_inverse[candidate] *= w_inverse

    # 寻找最优最劣解（正、负理想解）
    # 正
    positive_direct = max(items_score_direct.values())
    positive_inverse = max(items_score_inverse.values())
    # 负
    negative_direct = min(items_score_direct.values())
    negative_inverse = min(items_score_inverse.values())
    # boundary condition. 当最优解和最劣解相同时
    if positive_direct == negative_direct and positive_inverse == negative_inverse:
        items_score_reranking = {candidate: 1.0 for candidate in candidates}
        return items_score_reranking

    # 计算新物品与最优最劣解的欧式距离，并根据此得出贴近度（结果）
    items_score_reranking = {}  # 贴近度。越大越好，注意与python版本定义正好相反
    for candidate in candidates:
        rank_direct = items_score_direct[candidate]
        rank_inverse = items_score_inverse[candidate]

        distance_positive = (
            (rank_direct - positive_direct) * (rank_direct - positive_direct)
            + (rank_inverse - positive_inverse) * (rank_inverse - positive_inverse)
        )

        distance_negative = (
            (rank_direct - negative_direct) * (rank_direct - negative_direct)
            + (rank_inverse - negative_inverse) * (rank_inverse - negative_inverse)
        )

        # 靠近1说明靠近positive，靠近0说明靠近negtive
        res = distance_negative / (distance_positive + distance_negative)
        items_score_reranking[candidate] = res

    return items_score_reranking  # 特别注意，推荐结果数目是不固定的
