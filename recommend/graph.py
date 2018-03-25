"""
属于图模型分类下的推荐算法。
"""

from collections import defaultdict
import math


def probs(target_user, args):
    # 概率传播模型。ProbS。

    # # version 1。按照最终的公式形式，一个一个结点进行传播来计算，速度很慢！
    # items_hasrated = users_pair[target_user]
    # items_unrated = items_pair.keys() - items_hasrated  # 集合的交并操作很费时，可以改成在for循环里面加boundary condition条件进行判断
    # energy_node_items_before = dict(zip(list(items_hasrated), [1.0 for i in range(len(items_hasrated))]))
    #
    # items_score = {}  # 特别注意，推荐结果数目是不固定的。
    #
    # for item_unrated in items_unrated:
    #     users_connect_itemunrated = items_pair[item_unrated]
    #
    #     sum_score = 0.0
    #     for (item_hasrated, value) in energy_node_items_before.items():
    #         users_connect_itemhasrated = items_pair[item_hasrated]
    #         users_valid = users_connect_itemhasrated & users_connect_itemunrated  # 集合的交并操作很费时，可以修改优化
    #         user_energy = value * 1.0 / len(users_connect_itemhasrated)
    #         for user in users_valid:
    #             sum_score += user_energy
    #
    #     items_score[item_unrated] = sum_score
    #
    # return items_score

    # version 2。不按照最终的公式形式思考，而是从最初的图传播思想来入手，按照结点传播的思路模拟一遍，来写程序
    users_pair = args.get('users_pair', None)
    items_pair = args.get('items_pair', None)
    users_degree = args.get('users_degree', None)
    items_degree = args.get('items_degree', None)

    # boundary condition
    if users_pair is None or items_pair is None:
        print('\nError. The function probs_convention() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    energy_node_items = defaultdict(lambda: 0.0)
    energy_node_users = defaultdict(lambda: 0.0)

    # 初始化。把能量赋予物品结点
    items_hasrated = users_pair[target_user]
    for itemj in items_hasrated:
        energy_node_items[itemj] = 1.0

    # 第一次传播。能量从物品结点传到用户结点
    for (item, value) in energy_node_items.items():
        energy = value / items_degree[item]
        for user in items_pair[item]:
            energy_node_users[user] += energy
    energy_node_items.clear()

    # 第二次传播。能量从用户结点传回物品结点
    for (user, value) in energy_node_users.items():
        energy = value / users_degree[user]
        for item in users_pair[user]:
            energy_node_items[item] += energy
    #energy_node_users.clear()  # 没有必要再清空这个，因为只进行两步的传播就在这里结束

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的
    for (item, energy) in energy_node_items.items():
        if item in items_hasrated:  # boundary condition
            continue
        if item not in energy_node_items:  # boundary condition
            continue
        items_score[item] = energy

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的。
def heats(target_user, args):
    # 热传导模型。HeatS。

    # version 2。不按照最终的公式形式思考，而是从最初的图传播思想来入手，按照结点传播的思路模拟一遍，来写程序
    users_pair = args.get('users_pair', None)
    items_pair = args.get('items_pair', None)
    users_degree = args.get('users_degree', None)
    items_degree = args.get('items_degree', None)

    # boundary condition
    if users_pair is None or items_pair is None:
        print('\nError. The function heats() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    energy_node_items = defaultdict(lambda: 0.0)
    energy_node_users = defaultdict(lambda: 0.0)

    # 初始化。把能量赋予物品结点
    items_hasrated = users_pair[target_user]
    for itemj in items_hasrated:
        energy_node_items[itemj] = 1.0

    # 第一次传播。能量从物品结点传到用户结点
    for (item, value) in energy_node_items.items():
        energy = value
        for user in items_pair[item]:
            energy_node_users[user] += energy / users_degree[user]
    energy_node_items.clear()

    # 第二次传播。能量从用户结点传回物品结点
    for (user, value) in energy_node_users.items():
        energy = value
        for item in users_pair[user]:
            energy_node_items[item] += energy / items_degree[item]
    # energy_node_users.clear()  # 没有必要再清空这个，因为只进行两步的传播就在这里结束

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的
    for (item, energy) in energy_node_items.items():
        if item in items_hasrated:  # boundary condition
            continue
        if item not in energy_node_items:  # boundary condition
            continue
        items_score[item] = energy

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的。
def hph(target_user, args):
    # 融合模型HPH。ProbS+HeatS。
    # 参数itemdegree_lambda。

    # version 2。不按照最终的公式形式思考，而是从最初的图传播思想来入手，按照结点传播的思路模拟一遍，来写程序
    users_pair = args.get('users_pair', None)
    items_pair = args.get('items_pair', None)
    users_degree = args.get('users_degree', None)
    items_degree = args.get('items_degree', None)
    itemdegree_lambda = args.get('itemdegree_lambda', None)

    # boundary condition
    if users_pair is None or items_pair is None:
        print('\nError. The function hph() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    energy_node_items = defaultdict(lambda: 0.0)
    energy_node_users = defaultdict(lambda: 0.0)

    # 初始化。把能量赋予物品结点
    items_hasrated = users_pair[target_user]
    for itemj in items_hasrated:
        energy_node_items[itemj] = 1.0

    # 第一次传播。能量从物品结点传到用户结点
    for (item, value) in energy_node_items.items():
        energy = value / math.pow(items_degree[item], itemdegree_lambda)  # k(j)^lambda
        for user in items_pair[item]:
            energy_node_users[user] += energy
    energy_node_items.clear()

    # 第二次传播。能量从用户结点传回物品结点
    for (user, value) in energy_node_users.items():
        energy = value / users_degree[user]  # k(v)
        for item in users_pair[user]:
            energy_node_items[item] += energy
    # 为了改进，只取一次key键，所以把对应操作放到了外面
    for (item, energy) in energy_node_items.items():
        energy_node_items[item] = energy / math.pow(items_degree[item], 1 - itemdegree_lambda)  # k(i)^(1-lambda)
    # energy_node_users.clear()  # 没有必要再清空这个，因为只进行两步的传播就在这里结束

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的
    for (item, energy) in energy_node_items.items():
        if item in items_hasrated:  # boundary condition
            continue
        if item not in energy_node_items:  # boundary condition
            continue
        items_score[item] = energy

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的。
def pd(target_user, args):
    # 优先传播模型PD。
    # 参数itemdegree_lambda。

    # version 2。不按照最终的公式形式思考，而是从最初的图传播思想来入手，按照结点传播的思路模拟一遍，来写程序
    users_pair = args.get('users_pair', None)
    items_pair = args.get('items_pair', None)
    users_degree = args.get('users_degree', None)
    items_degree = args.get('items_degree', None)
    itemdegree_x = args.get('itemdegree_x', None)

    # boundary condition
    if users_pair is None or items_pair is None:
        print('\nError. The function hph() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    energy_node_items = defaultdict(lambda: 0.0)
    energy_node_users = defaultdict(lambda: 0.0)

    # 初始化。把能量赋予物品结点
    items_hasrated = users_pair[target_user]
    for itemj in items_hasrated:
        energy_node_items[itemj] = 1.0

    # 第一次传播。能量从物品结点传到用户结点
    for (item, value) in energy_node_items.items():
        energy = value / items_degree[item]  # k(j)
        for user in items_pair[item]:
            energy_node_users[user] += energy
    energy_node_items.clear()

    # 第二次传播。能量从用户结点传回物品结点
    for (user, value) in energy_node_users.items():
        weight = 0.0
        for item in users_pair[user]:
            weight += math.pow(items_degree[item], itemdegree_x)

        energy = value / weight  # weight(v)
        for item in users_pair[user]:
            energy_node_items[item] += energy
    # 为了改进，只取一次key键，所以把对应操作放到了外面
    for (item, energy) in energy_node_items.items():
        energy_node_items[item] = energy / math.pow(items_degree[item], (-1) * itemdegree_x)  # k(i)^ -x
    # energy_node_users.clear()  # 没有必要再清空这个，因为只进行两步的传播就在这里结束

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的
    for (item, energy) in energy_node_items.items():
        if item in items_hasrated:  # boundary condition
            continue
        if item not in energy_node_items:  # boundary condition
            continue
        items_score[item] = energy

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的。


def probs_likeitemcf(users_pair, items_pair, target_user, init=1, **kwargs):
    similarities_probs = kwargs.get('similarities_probs', None)

    # boundary condition
    if similarities_probs is None:
        print(' The function probs_likeitemcf() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    similarities = similarities_probs

    # version2。直接调用icf()方法，不用再把重复代码写一遍了
    # version1。先确定候选物品集合，再执行Item CF协同过滤算法
    items_unrated = items_pair.keys() - users_pair[target_user]  # 集合的交并操作很费时，可以改成在for循环里面加boundary condition条件进行判断
    items_score = {}  # 特别注意，推荐结果数目是不固定的。

    for item in items_unrated:
        sum_score = 0.0
        for item_hasrated in users_pair[target_user]:
            sum_score += similarities[item][item_hasrated]

        if sum_score == 0.0:  # boundary condition
            continue

        items_score[item] = sum_score

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的。


def probs_randomwalk(target_user, args):
    # 加入随机转移概率。

    users_pair = args.get('users_pair', None)
    items_pair = args.get('items_pair', None)
    users_degree = args.get('users_degree', None)
    items_degree = args.get('items_degree', None)
    ku = users_degree[target_user]
    klogu = math.log1p(ku)
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_klogu = klogu / max_userdegreelog1p

    # boundary condition
    if users_pair is None or items_pair is None:
        print('\nError. The function probs_convention() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    energy_node_items = defaultdict(lambda: 0.0)
    energy_node_users = defaultdict(lambda: 0.0)

    # 初始化。把能量赋予物品结点
    items_hasrated = users_pair[target_user]
    for itemj in items_hasrated:
        energy_node_items[itemj] = 1 * (1 / ku / math.pow(math.log1p(items_degree[itemj]), norm_klogu))  # 加入random walk

    # 第一次传播。能量从物品结点传到用户结点
    for (item, value) in energy_node_items.items():
        for user in items_pair[item]:
            energy_node_users[user] += value * (1 / items_degree[item] / math.pow(math.log1p(users_degree[user]), norm_klogu))
    energy_node_items.clear()

    # 第二次传播。能量从用户结点传回物品结点
    for (user, value) in energy_node_users.items():
        for item in users_pair[user]:
            energy_node_items[item] += value * (1 / users_degree[user] / math.pow(math.log1p(items_degree[item]), norm_klogu))
    # energy_node_users.clear()  # 没有必要再清空这个，因为只进行两步的传播就在这里结束

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的
    for (item, energy) in energy_node_items.items():
        if item in items_hasrated:  # boundary condition
            continue
        if item not in energy_node_items:  # boundary condition
            continue
        items_score[item] = energy

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的。


def probs_1(target_user, args):
    # 加入第一步的随机转移概率。

    users_pair = args.get('users_pair', None)
    items_pair = args.get('items_pair', None)
    users_degree = args.get('users_degree', None)
    items_degree = args.get('items_degree', None)
    ku = users_degree[target_user]
    klogu = math.log1p(ku)
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_klogu = klogu / max_userdegreelog1p

    # boundary condition
    if users_pair is None or items_pair is None:
        print('\nError. The function probs_convention() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    energy_node_items = defaultdict(lambda: 0.0)
    energy_node_users = defaultdict(lambda: 0.0)

    # 初始化。把能量赋予物品结点
    items_hasrated = users_pair[target_user]
    for itemj in items_hasrated:
        energy_node_items[itemj] = 1 * (1 / ku / math.pow(math.log1p(items_degree[itemj]), norm_klogu))  # 加入random walk

    # 第一次传播。能量从物品结点传到用户结点
    for (item, value) in energy_node_items.items():
        for user in items_pair[item]:
            energy_node_users[user] += value * (1 / items_degree[item])
    energy_node_items.clear()

    # 第二次传播。能量从用户结点传回物品结点
    for (user, value) in energy_node_users.items():
        for item in users_pair[user]:
            energy_node_items[item] += value * (1 / users_degree[user])
    # energy_node_users.clear()  # 没有必要再清空这个，因为只进行两步的传播就在这里结束

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的
    for (item, energy) in energy_node_items.items():
        if item in items_hasrated:  # boundary condition
            continue
        if item not in energy_node_items:  # boundary condition
            continue
        items_score[item] = energy

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的。


def probs_12(target_user, args):
    # 加入一、二步叠加后的随机转移概率。

    users_pair = args.get('users_pair', None)
    items_pair = args.get('items_pair', None)
    users_degree = args.get('users_degree', None)
    items_degree = args.get('items_degree', None)
    ku = users_degree[target_user]
    klogu = math.log1p(ku)
    max_userdegreelog1p = max([math.log1p(degree) for degree in users_degree.values()])
    norm_klogu = klogu / max_userdegreelog1p

    # boundary condition
    if users_pair is None or items_pair is None:
        print('\nError. The function probs_convention() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    energy_node_items = defaultdict(lambda: 0.0)
    energy_node_users = defaultdict(lambda: 0.0)

    # 初始化。把能量赋予物品结点
    items_hasrated = users_pair[target_user]
    for itemj in items_hasrated:
        energy_node_items[itemj] = 1 * (1 / ku / math.pow(math.log1p(items_degree[itemj]), norm_klogu))  # 加入random walk

    # 第一次传播。能量从物品结点传到用户结点
    for (item, value) in energy_node_items.items():
        for user in items_pair[item]:
            energy_node_users[user] += value * (1 / items_degree[item] / math.pow(math.log1p(users_degree[user]), norm_klogu))
    energy_node_items.clear()

    # 第二次传播。能量从用户结点传回物品结点
    for (user, value) in energy_node_users.items():
        for item in users_pair[user]:
            energy_node_items[item] += value * (1 / users_degree[user])
    # energy_node_users.clear()  # 没有必要再清空这个，因为只进行两步的传播就在这里结束

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的
    for (item, energy) in energy_node_items.items():
        if item in items_hasrated:  # boundary condition
            continue
        if item not in energy_node_items:  # boundary condition
            continue
        items_score[item] = energy

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的。


def probs_step1(target_user, args):
    # 在第一个部分做改进。

    users_pair = args.get('users_pair', None)
    items_pair = args.get('items_pair', None)
    users_degree = args.get('users_degree', None)
    items_degree = args.get('items_degree', None)
    ku = users_degree[target_user]
    klogu = math.log1p(ku)
    max_kloguser = max([math.log1p(degree) for degree in users_degree.values()])
    norm_klogu = klogu / max_kloguser

    # boundary condition
    if users_pair is None or items_pair is None:
        print('\nError. The function probs_step1() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    energy_node_items = defaultdict(lambda: 0.0)
    energy_node_users = defaultdict(lambda: 0.0)

    # 第一步
    items_hasrated = users_pair[target_user]
    prb_old = 1 / ku  # 优化后的代码
    for itemj in items_hasrated:
        prb = prb_old + norm_klogu * (1 / items_degree[itemj])
        energy_node_items[itemj] = prb  # 优化后的代码

    # 第二步
    for (item, value) in energy_node_items.items():
        prb = 1 / items_degree[item]  # 优化后的代码
        for user in items_pair[item]:
            energy_node_users[user] += value * prb
    energy_node_items.clear()

    # 第三步
    for (user, value) in energy_node_users.items():
        prb = 1 / users_degree[user]  # 优化后的代码
        for item in users_pair[user]:
            energy_node_items[item] += value * prb
    # energy_node_users.clear()  # 没有必要再清空这个，因为只进行两步的传播就在这里结束

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的
    for (item, energy) in energy_node_items.items():
        if item in items_hasrated:  # boundary condition
            continue
        if item not in energy_node_items:  # boundary condition
            continue
        items_score[item] = energy

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的。


def probs_step3(target_user, args):
    # 在第二个部分做改进。

    users_pair = args.get('users_pair', None)
    items_pair = args.get('items_pair', None)
    users_degree = args.get('users_degree', None)
    items_degree = args.get('items_degree', None)
    ku = users_degree[target_user]
    klogu = math.log1p(ku)
    max_kloguser = max([math.log1p(degree) for degree in users_degree.values()])
    norm_klogu = klogu / max_kloguser

    # boundary condition
    if users_pair is None or items_pair is None:
        print('\nError. The function probs_step2_v1() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    energy_node_items = defaultdict(lambda: 0.0)
    energy_node_users = defaultdict(lambda: 0.0)

    # 第一步
    items_hasrated = users_pair[target_user]
    prb_old = 1 / ku  # 优化后的代码
    for itemj in items_hasrated:
        energy_node_items[itemj] = prb_old  # 优化后的代码

    # 第二步
    for (item, value) in energy_node_items.items():
        prb_old = 1 / items_degree[item]  # 优化后的代码
        for user in items_pair[item]:
            energy_node_users[user] += value * prb_old
    energy_node_items.clear()

    # 第三步
    for (user, value) in energy_node_users.items():
        prb_old = 1 / users_degree[user]  # 优化后的代码
        for item in users_pair[user]:
            prb = prb_old + norm_klogu * (1 / items_degree[item])
            energy_node_items[item] += value * prb
    # energy_node_users.clear()  # 没有必要再清空这个，因为只进行两步的传播就在这里结束

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的
    for (item, energy) in energy_node_items.items():
        if item in items_hasrated:  # boundary condition
            continue
        if item not in energy_node_items:  # boundary condition
            continue
        items_score[item] = energy

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的。


def probs_step1_step3(target_user, args):
    # 在第一个部分和第二个部分做改进。

    users_pair = args.get('users_pair', None)
    items_pair = args.get('items_pair', None)
    users_degree = args.get('users_degree', None)
    items_degree = args.get('items_degree', None)
    userdegree_lambda = args.get('userdegree_lambda', None)
    ku = users_degree[target_user]
    klogu = math.log1p(ku)

    # boundary condition
    if users_pair is None or items_pair is None:
        print('\nError. The function probs_step2_v1() is wrong, please correct the code!')
        exit(1)
    if target_user not in users_pair:
        return None

    energy_node_items = defaultdict(lambda: 0.0)
    energy_node_users = defaultdict(lambda: 0.0)

    # 第一步
    items_hasrated = users_pair[target_user]
    prb_old = 1 / ku  # 优化后的代码
    for itemj in items_hasrated:
        prb = prb_old + userdegree_lambda * klogu / items_degree[itemj]
        energy_node_items[itemj] = prb  # 优化后的代码

    # 第二步
    for (item, value) in energy_node_items.items():
        prb_old = 1 / items_degree[item]  # 优化后的代码
        for user in items_pair[item]:
            energy_node_users[user] += value * prb_old
    energy_node_items.clear()

    # 第三步
    for (user, value) in energy_node_users.items():
        prb_old = 1 / users_degree[user]  # 优化后的代码
        for item in users_pair[user]:
            prb = prb_old + userdegree_lambda * klogu / items_degree[item]
            energy_node_items[item] += value * prb
    # energy_node_users.clear()  # 没有必要再清空这个，因为只进行两步的传播就在这里结束

    items_score = defaultdict(lambda: 0.0)  # 特别注意，推荐结果数目是不固定的
    for (item, energy) in energy_node_items.items():
        if item in items_hasrated:  # boundary condition
            continue
        if item not in energy_node_items:  # boundary condition
            continue
        items_score[item] = energy

    # 推荐结果
    return items_score  # 特别注意，推荐结果数目是不固定的。
