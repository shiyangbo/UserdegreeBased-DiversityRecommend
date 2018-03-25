"""
属于CF协同过滤分类下的推荐算法。
"""

import numpy as np
from collections import defaultdict
import math
import sys

sys.path.append(r'D:\recommender')
import recommend.similarity as sim


def ucf_q(target_user, args):
    # User CF q。
    # // 原论文中的实验表明，一般最优参数q=9。

    users_pair = args.get('users_pair', None)
    q = args.get('q', None)
    similarities_usercf = args.get('similarities_usercf', None)

    # boundary condition
    if users_pair is None or q is None:
        print('\nError. The function ucf_q() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的。
    items_hasrated = users_pair[target_user]

    # # 方法1。先确定候选物品集合，再执行协同过滤算法（集合的交并操作很费时，可以改成在for循环里面加boundary condition条件进行判断。）
    # items_pair = kwargs.get('items_pair', None)  # 新添加
    # items_unrated = items_pair.keys() - users_pair[target_user]
    # for item in items_unrated:
    #     for user in items_pair[item]:
    #         if user not in similarities_usercf[target_user]:  # user和target_user之间的相似度为0时，continue
    #             continue
    #         similarity = similarities_usercf[target_user][user]
    #         # 协同过滤推荐算法
    #         items_score[item] += math.pow(similarity, q)

    # 方法2。先使用最近邻用户（全部用户）进行推荐，再判断是否属于target_user的新物品

    # boundary condition。在线（实时）计算相似度，并直接得到items_score
    if similarities_usercf is None:
        items_pair = args.get('items_pair', None)

        neighbors = {}
        items1 = users_pair[target_user]
        for other_user in users_pair.keys():
            if target_user == other_user:
                continue
            items2 = users_pair[other_user]
            count_u1 = len(items1)
            count_u2 = len(items2)
            count_common = len(items1 & items2)
            if count_common == 0:
                continue
            similarity_with_q = math.pow(count_common, q) / math.pow(count_u1 * count_u2, q / 2.0)
            for item2 in items2:
                if item2 in items_hasrated:  # boundary condition
                    continue
                # 协同过滤推荐算法
                items_score[item2] += similarity_with_q

        return items_score  # 特别注意，推荐结果数目是不固定的

    neighbors = similarities_usercf[target_user]
    for (neighboruser, similarity) in neighbors.items():
        for item in users_pair[neighboruser]:
            if item in items_hasrated:  # boundary condition
                continue
            # 协同过滤推荐算法
            items_score[item] += math.pow(similarity, q)

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的


def ucf_q_neighbordegree(target_user, args):
    # f(w) = w^q / kv^Norm(ku)。

    users_pair = args.get('users_pair', None)
    similarities_usercf = args.get('similarities_usercf', None)

    # boundary condition
    if users_pair is None:
        print('\nError. The function ucf_q_neighbordegree3() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    users_degree = args.get('users_degree', None)  # 用户度的统计值
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    q = args.get('q', None)

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的
    items_hasrated = users_pair[target_user]

    # 方法2。先使用最近邻用户（全部用户）进行推荐，再判断是否属于target_user的新物品
    neighbors = similarities_usercf[target_user]
    for (neighboruser, similarity) in neighbors.items():
        kv = math.log1p(users_degree[neighboruser])

        for item in users_pair[neighboruser]:
            if item in items_hasrated:  # boundary condition
                continue
            # 协同过滤推荐算法
            items_score[item] += math.pow(similarity, q) / math.pow(kv, ku / max_userdegreelog1p)

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的


def ucf_userdegree(target_user, args):
    # q变成q(u)。q(u)=lambda * ku。

    userdegree_lambda = args.get('userdegree_lambda', None)
    # boundary condition
    if userdegree_lambda is None:
        print('\nError. The function ucf_userdegree() is wrong, please correct the code!')
        exit(1)
    users_degree = args.get('users_degree', None)  # 用户度的统计值
    q = userdegree_lambda * math.log1p(users_degree[target_user])
    args.update({'q': q})
    return ucf_q(target_user, args)


def ucf_userdegree_neighbordegree(target_user, args):
    # f(w) = w^(lambda * ku) / kv^Norm(ku)。

    users_pair = args.get('users_pair', None)
    similarities_usercf = args.get('similarities_usercf', None)

    # boundary condition
    if users_pair is None:
        print('\nError. The function ucf_userdegree_neighbordegree3() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    users_degree = args.get('users_degree', None)  # 用户度的统计值
    userdegree_lambda = args.get('userdegree_lambda', None)
    ku = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的
    items_hasrated = users_pair[target_user]

    # 方法2。先使用最近邻用户（全部用户）进行推荐，再判断是否属于target_user的新物品

    neighbors = similarities_usercf[target_user]
    for (neighboruser, similarity) in neighbors.items():
        kv = math.log1p(users_degree[neighboruser])

        for item in users_pair[neighboruser]:
            if item in items_hasrated:  # boundary condition
                continue
            # 协同过滤推荐算法
            items_score[item] += math.pow(similarity, userdegree_lambda * ku) / math.pow(kv, ku / max_userdegreelog1p)

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的


# 分割线
def ucf_knn(target_user, args):
    """User CF kNN。参数k。
    // 特别注意，推荐结果数目是不固定的。
    """

    user_knn = args.get('user_knn', None)
    users_pair = args.get('users_pair', None)
    similarities_usercf = args.get('similarities_usercf', None)

    # boundary condition
    if users_pair is None or user_knn is None:
        print('\nError. The function ucf_neighbor() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的。
    items_hasrated = users_pair[target_user]

    # boundary condition。在线（实时）计算相似度，并直接得到items_score
    if similarities_usercf is None:
        items_pair = args.get('items_pair', None)

        neighbors = {}
        items1 = users_pair[target_user]
        m1 = len(items1)
        for item in items1:
            for other_user in items_pair[item]:
                if target_user == other_user:
                    continue
                if other_user in neighbors:  # 最近邻用户other_user
                    continue

                # 得到每一个最近邻用户other_user
                items2 = users_pair[other_user]
                m2 = len(items2)

                similarity = len(items1 & items2) / math.sqrt(m1 * m2)
                neighbors[other_user] = similarity

        # 找到用户target_user的最近邻，然后使用最近邻用户进行推荐
        if len(neighbors) == 0:  # boundary condition
            return None
        neighbors_nearest = sorted(neighbors.items(), key=lambda a: a[1], reverse=True)[0:user_knn]
        for (neighboruser, similarity) in neighbors_nearest:
            for item in users_pair[neighboruser]:
                if item in items_hasrated:  # boundary condition
                    continue
                items_score.setdefault(item, 0.0)
                items_score[item] += similarity

        # 推荐结果
        return items_score  # 特别注意，推荐结果数目是不固定的

    # 找到用户target_user的最近邻
    neighbors = similarities_usercf[target_user]
    if len(neighbors) == 0:  # boundary condition
        return None
    neighbors_nearest = sorted(neighbors.items(), key=lambda a: a[1], reverse=True)[0:user_knn]

    # 方法1。先确定候选物品集合，再使用最近邻用户进行推荐
    # 因为集合的交并操作很费时，所以我们直接使用法2从而避免了集合的交并操作。
    # 方法2。先使用最近邻用户进行推荐，再判断是否属于target_user的新物品
    for (neighboruser, similarity) in neighbors_nearest:
        for item in users_pair[neighboruser]:
            if item in items_hasrated:  # boundary condition
                continue
            items_score.setdefault(item, 0.0)
            items_score[item] += similarity

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的


def ucf_userdegree_itemdegree_fixed(target_user, args):
    # f(w) = w^ 2*ku / ki。

    users_pair = args.get('users_pair', None)
    items_pair = args.get('items_pair', None)
    similarities_usercf = args.get('similarities_usercf', None)
    userdegree_lambda = args.get('userdegree_lambda', None)

    # boundary condition
    if users_pair is None or items_pair is None or userdegree_lambda is None:
        print('\nError. The function ucf_userdegree_itemdegree_fixed() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    # 物品度
    items_degree = args.get('items_degree', None)
    # 用户度
    users_degree = args.get('users_degree', None)

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的。
    items_hasrated = users_pair[target_user]

    # 方法2。先使用最近邻用户（全部用户）进行推荐，再判断是否属于target_user的新物品
    # boundary condition。在线（实时）计算相似度，并直接得到items_score
    if similarities_usercf is None:
        items1 = users_pair[target_user]
        for other_user in users_pair.keys():
            if target_user == other_user:
                continue
            items2 = users_pair[other_user]
            count_u1 = len(items1)
            count_u2 = len(items2)
            count_common = len(items1 & items2)
            if count_common == 0:
                continue
            ku = math.log1p(users_degree[target_user])  # 可以改进。放到循环外面
            similarity_with_q = math.pow(count_common, userdegree_lambda * ku) / math.pow(count_u1 * count_u2,
                                                                                          userdegree_lambda * ku / 2.0)
            for item2 in items2:
                if item2 in items_hasrated:  # boundary condition
                    continue
                # 协同过滤推荐算法
                ki = math.log1p(items_degree[item2])
                items_score[item2] += similarity_with_q / ki

        return items_score  # 特别注意，推荐结果数目是不固定的

    neighbors = similarities_usercf[target_user]
    for (neighboruser, similarity) in neighbors.items():
        for item in users_pair[neighboruser]:
            if item in items_hasrated:  # boundary condition
                continue
            # 协同过滤推荐算法
            ku = math.log1p(users_degree[target_user])
            ki = math.log1p(items_degree[item])
            items_score[item] += math.pow(similarity, userdegree_lambda * ku) / ki

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的


def ucf_userdegree_itemdegree_flexible(target_user, args):
    # f(w) = w^ 2*ku / ki^Norm(ku)。

    users_pair = args.get('users_pair', None)
    items_pair = args.get('items_pair', None)
    similarities_usercf = args.get('similarities_usercf', None)
    userdegree_lambda = args.get('userdegree_lambda', None)

    # boundary condition
    if users_pair is None or items_pair is None or userdegree_lambda is None:
        print('\nError. The function ucf_userdegree_itemdegree_flexible() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    # 物品度
    items_degree = args.get('items_degree', None)
    # 用户度
    users_degree = args.get('users_degree', None)
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的。
    items_hasrated = users_pair[target_user]

    # 方法2。先使用最近邻用户（全部用户）进行推荐，再判断是否属于target_user的新物品
    # boundary condition。在线（实时）计算相似度，并直接得到items_score
    if similarities_usercf is None:
        items1 = users_pair[target_user]
        for other_user in users_pair.keys():
            if target_user == other_user:
                continue
            items2 = users_pair[other_user]
            count_u1 = len(items1)
            count_u2 = len(items2)
            count_common = len(items1 & items2)
            if count_common == 0:
                continue
            ku = math.log1p(users_degree[target_user])  # 可以改进。放到循环外面
            similarity_with_q = math.pow(count_common, userdegree_lambda * ku) / math.pow(count_u1 * count_u2,
                                                                                          userdegree_lambda * ku / 2.0)
            for item2 in items2:
                if item2 in items_hasrated:  # boundary condition
                    continue
                # 协同过滤推荐算法
                ki = math.log1p(items_degree[item2])
                items_score[item2] += similarity_with_q / math.pow(ki, ku / max_userdegreelog1p)

        return items_score  # 特别注意，推荐结果数目是不固定的

    neighbors = similarities_usercf[target_user]
    for (neighboruser, similarity) in neighbors.items():
        for item in users_pair[neighboruser]:
            if item in items_hasrated:  # boundary condition
                continue
            # 协同过滤推荐算法
            ku = math.log1p(users_degree[target_user])
            ki = math.log1p(items_degree[item])
            items_score[item] += math.pow(similarity, userdegree_lambda * ku) / math.pow(ki, ku / max_userdegreelog1p)

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的


def ucf_userdegree_itemdegree_factorized(target_user, args):
    # f(w) = w^lam*(ku*ki)。lam in [0.01, 0.02, 0.03, ...]。

    users_pair = args.get('users_pair', None)
    items_pair = args.get('items_pair', None)
    similarities_usercf = args.get('similarities_usercf', None)
    factorized_lambda = args.get('factorized_lambda', None)

    # boundary condition
    if users_pair is None or items_pair is None or factorized_lambda is None:
        print('\nError. The function ucf_userdegree_itemdegree_factorized() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    # 物品度
    items_degree = args.get('items_degree', None)
    # 用户度
    users_degree = args.get('users_degree', None)

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的。
    items_hasrated = users_pair[target_user]

    # 方法2。先使用最近邻用户（全部用户）进行推荐，再判断是否属于target_user的新物品
    # boundary condition。在线（实时）计算相似度，并直接得到items_score
    if similarities_usercf is None:
        items1 = users_pair[target_user]
        for other_user in users_pair.keys():
            if target_user == other_user:
                continue
            items2 = users_pair[other_user]
            count_u1 = len(items1)
            count_u2 = len(items2)
            count_common = len(items1 & items2)
            if count_common == 0:
                continue
            ku = math.log1p(users_degree[target_user])
            for item2 in items2:
                if item2 in items_hasrated:  # boundary condition
                    continue
                # 协同过滤推荐算法
                ki = math.log1p(items_degree[item2])
                similarity_with_weight = math.pow(count_common, factorized_lambda * ku * ki) / math.pow(
                    count_u1 * count_u2, factorized_lambda * ku * ki / 2.0)
                items_score[item2] += similarity_with_weight

        return items_score  # 特别注意，推荐结果数目是不固定的

    neighbors = similarities_usercf[target_user]
    for (neighboruser, similarity) in neighbors.items():
        for item in users_pair[neighboruser]:
            if item in items_hasrated:  # boundary condition
                continue
            # 协同过滤推荐算法
            ku = math.log1p(users_degree[target_user])  # 可以改进。放到循环外面
            ki = math.log1p(items_degree[item])
            items_score[item] += math.pow(similarity, factorized_lambda * (ku * ki))

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的


def ucfq_neighbordegree2(target_user, args):
    # f(w) = (1/2)^|klog(u)-klog(v)| * w^q。

    users_pair = args.get('users_pair', None)
    similarities_usercf = args.get('similarities_usercf', None)

    # boundary condition
    if users_pair is None:
        print('\nError. The function ucf_q_neighbordegree3() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    users_degree = args.get('users_degree', None)  # 用户度的统计值
    klogu = math.log1p(users_degree[target_user])
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    q = args.get('q', None)

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的
    items_hasrated = users_pair[target_user]

    # 方法2。先使用最近邻用户（全部用户）进行推荐，再判断是否属于target_user的新物品
    neighbors = similarities_usercf[target_user]
    for (neighboruser, similarity) in neighbors.items():
        klogv = math.log1p(users_degree[neighboruser])

        for item in users_pair[neighboruser]:
            if item in items_hasrated:  # boundary condition
                continue
            # 协同过滤推荐算法
            items_score[item] += math.pow(0.5, math.fabs(klogu-klogv)) * math.pow(similarity, q)

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的


def icf_knn_norm(target_user, args):
    # Item CF kNN Norm。反向邻居物品，结合相似度标准化（SNorm+）。
    # 很常用。
    # 参数item_knn。一般取10就很好了。
    #
    # // 特别注意，推荐结果数目是不固定的。

    users_pair = args.get('users_pair', None)
    similarities_itemcf = args.get('similarities_itemcf', None)
    item_knn = args.get('item_knn', None)
    # boundary condition
    if users_pair is None or similarities_itemcf is None or item_knn is None:
        print('The function icf_knn_norm() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    # 算法部分
    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的。
    items_hasrated = users_pair[target_user]

    for itemj in items_hasrated:
        # 找到物品itemj的最近邻
        neighbors = similarities_itemcf[itemj]
        if len(neighbors) == 0:  # boundary condition
            continue

        maxsimilarity = max(neighbors.values())
        neighbors_nearest = sorted(neighbors.items(), key=lambda a: a[1], reverse=True)[0:item_knn]
        for (item_new, similarity) in neighbors_nearest:
            if item_new in items_hasrated:  # boundary condition
                continue
            items_score[item_new] += (similarity / maxsimilarity)

    return items_score  # 特别注意，推荐结果数目是不固定的


def icf_knn_norm_itemdegree(target_user, args):
    # f(w) = w / kj^Norm(ku)。或者多样性更好的Norm(kj) = (kj - minJ) / (maxJ - minJ)。
    # 参数item_knn。
    #
    # // 特别注意，推荐结果数目是不固定的。

    item_knn = args.get('item_knn', None)
    # 物品度
    items_degree = args.get('items_degree', None)
    # 用户度
    users_degree = args.get('users_degree', None)
    max_userdegreelog = max([math.log1p(degree) for degree in users_degree.values()])
    norm_klogu = math.log1p(users_degree[target_user]) / max_userdegreelog

    users_pair = args.get('users_pair', None)
    similarities_itemcf = args.get('similarities_itemcf', None)
    # boundary condition
    if users_pair is None or similarities_itemcf is None or item_knn is None:
        print('\nError. The function icf_knn_norm_itemdegree() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    # 算法部分
    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的。
    items_hasrated = users_pair[target_user]

    for itemj in items_hasrated:
        neighbors = similarities_itemcf[itemj]
        if len(neighbors) == 0:  # boundary condition
            continue

        klogj = math.log1p(items_degree[itemj])
        maxsimilarity = max(neighbors.values())
        neighbors_nearest = sorted(neighbors.items(), key=lambda a: a[1], reverse=True)[0:item_knn]
        for (item_new, similarity) in neighbors_nearest:  # need change
            if item_new in items_hasrated:  # boundary condition
                continue
            items_score[item_new] += (similarity / maxsimilarity) / math.pow(klogj, norm_klogu)

    return items_score  # 特别注意，推荐结果数目是不固定的
