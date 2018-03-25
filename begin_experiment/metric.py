"""
离线评测。
"""


import os
import sys
sys.path.append(r'D:\recommender')
sys.path.append(r'F:\recommender')
import numpy as np
import math
import recommend.read as read
import recommend.recommend as recommend
import recommend.similarity as sim
from collections import defaultdict
import pickle


def get_nDCG(users_test, users_topn, topn):
    # 排序准确率指标。nDCG=DCG/IDCG。

    nDCG = 0.0
    count_validuser = 0

    for (user, items_test) in users_test.items():
        # boundary condition
        if users_topn[user] == []:
            continue
        if len(users_topn[user]) < topn:  # boundary condition. Jester数据集
            continue

        # 计算DCG
        DCG = 0.0
        count_hit = 0
        items_topnlist = users_topn[user][:topn]
        for i in range(topn):
            index = i + 1
            if items_topnlist[i] in items_test:
                DCG += 1 / math.log2(1 + index)
                count_hit += 1
            else:
                DCG += 0.0

        # 计算IDCG
        IDCG = sum([1 / math.log2(1 + index) for index in range(1, count_hit + 1)])

        # 最后得到nDCG
        if IDCG == 0.0:
            nDCG += 0.0
            count_validuser += 1
        else:
            nDCG += (DCG / IDCG)
            count_validuser += 1

    return nDCG / count_validuser


def get_PrecisionRecall(users_test, users_topn, topn):
    # 计算准确率和召回率。
    # // 召回率有两种不同的计算公式。这里我们采用的是《推荐系统实践》一书中所采用的第二种。
    # 与topn有关。

    hit = 0
    n_recall = 0
    n_precision = 0
    for user in users_test:
        # boudnary condition
        if users_topn[user] == []:
            continue
        if len(users_topn[user]) < topn:  # boundary condition. Jester数据集
            continue
        items_topnlist = set(users_topn[user][:topn])
        hit += len(items_topnlist & users_test[user])
        n_recall += len(users_test[user])
        n_precision += topn

    return (hit * 1.0 / n_precision, hit * 1.0 / n_recall)


def get_ILD(ii_similarities, users_test, users_topn, topn):
    # 计算多样性指标ILD(ILS)。
    # // ILD和ILS实质上是一样的。一个是距离d(i,j)的度量，一个是相似度s(i,j)的度量，ILD=1-ILS。
    # 与topn有关。

    intra_dissimilarity = 0.0
    count = 0

    for user in users_test:
        if users_topn[user] == []:  # boundary condition
            continue
        if len(users_topn[user]) < topn:  # boundary condition. Jester数据集
            continue
        items_topnlist = users_topn[user][:topn]
        for i in range(0, topn - 1):
            for j in range(i + 1, topn):
                item1 = items_topnlist[i]
                item2 = items_topnlist[j]
                if item2 in ii_similarities[item1]:
                    intra_dissimilarity += ii_similarities[item1][item2]
                    count += 1
                else:
                    intra_dissimilarity += 0.0
                    count += 1

    return 1 - (intra_dissimilarity / count)


def get_coverage(users_test, users_topn, topn, dataset_file):
    # 这里，使用最简单的覆盖率定义。除此之外还有信息熵和基尼系数两种更精细的覆盖率定义。
    # 返回值有两个。第一个是训练集全体用户作为分子得出的覆盖率，第二个是仅测试集中的用户作为分子得出的覆盖率。
    # 与topn有关。

    # 系统中的物品全体
    (users_pair, items_pair) = read.read_data(dataset_file)
    items_all = set(items_pair.keys())

    # 开始计算
    items_recommended_allusers = set()
    items_recommended_testusers = set()
    for user in users_topn:
        if users_topn[user] == []:  # boundary condition
            continue
        if len(users_topn[user]) < topn:  # boundary condition. Jester数据集
            continue
        items_recommended_allusers |= set(users_topn[user][:topn])
        if user in users_test:
            items_recommended_testusers |= set(users_topn[user][:topn])

    coverage_allusers = len(items_recommended_allusers) * 1.0 / len(items_all)
    coverage_testusers = len(items_recommended_testusers) * 1.0 / len(items_all)

    return (coverage_allusers, coverage_testusers)


def get_gini(users_test, users_topn, topn, dataset_file):
    # giniNew=1-giniOld。判断均匀程序。越不均匀，马太效应越高，“贫富差距越大”，giniNew系数越小；越均匀，“贫富越平等”，giniNew系数越大。
    # // 注意，这里我们计算的是整个数据集中的（系统中的）所有物品出现在推荐列表中的频数（概率）的gini系数指标。
    # 与topn有关。

    # 找到出现在推荐列表中的所有物品。并计算它们出现在推荐列表中的频数（概率）
    items_counts = defaultdict(lambda: 0)
    total_counts = 0
    for user in users_test:
        if users_topn[user] == []:  # boundary condition
            continue
        if len(users_topn[user]) < topn:  # boundary condition. Jester数据集
            continue

        items_topnlist = users_topn[user][:topn]
        for item in items_topnlist:
            items_counts[item] += 1
            total_counts += 1

    items_probability = {item: count * 1.0 / total_counts for (item, count) in
                         items_counts.items()}  # 可以改进。把除法操作拖延到放到函数最后

    # 从小到大递增排序
    items_probability_sorted = sorted(items_probability.items(), key=lambda a: a[1])

    # 系统中的物品全体
    (users_pair, items_pair) = read.read_data(dataset_file)
    n_all = len(items_pair)
    n_test = len(items_probability)

    gini = 0.0
    for i in range(len(items_probability)):
        probability = items_probability_sorted[i][1]
        index = i + (n_all - n_test) + 1  # 难点。易错，因为有(n_all - n_test)个物品在推荐列表中都没有出现过。覆盖率不为100%时肯定会出现这种情况
        gini += (2 * index - n_all - 1) * probability
    gini = 1 - gini / (n_all - 1)

    # gini = 0.0
    # n_test = len(items_probability_sorted)  # 这里只计算当前测试集当中的全体物品的排名，我认为这样更合适。（我在原来的java程序中是计算的数据集中的全体物品（Database.items）出现在推荐列表中的频数的排名）
    # for i in range(n_test):
    #     index = i + 1
    #     p_item_at_index = items_probability_sorted[i][1]
    #     gini += (2*index-n_test-1) * p_item_at_index
    # gini = 1 - gini / (n_test - 1)

    return gini


def get_novelty(users_test, users_topn, topn, dataset_file):
    # 计算推荐结果的新颖性（物品的平均流行度）。
    # // 取对数后，流行度的平均值更加稳定。
    # 返回值有两个。第一个是训练集全体用户作为分子得出的新颖性，第二个是仅测试集中的用户作为分子得出的新颖性。
    # 与topn有关。

    (users_pair, items_pair) = read.read_data(dataset_file)
    items_degree = {item: len(users) for (item, users) in items_pair.items()}

    res_allusers = 0.0
    count_allusers = 0
    res_testusers = 0.0
    count_testusers = 0

    for user in users_topn:
        if users_topn[user] == []:  # boundary condition
            continue
        if len(users_topn[user]) < topn:  # boundary condition. Jester数据集
            continue
        if user in users_test:
            for item in users_topn[user][:topn]:
                degreevalue = np.log(1 + items_degree[item])
                res_allusers += degreevalue
                res_testusers += degreevalue
                count_allusers += 1
                count_testusers += 1
        else:
            for item in users_topn[user][:topn]:
                degreevalue = np.log(1 + items_degree[item])
                res_allusers += degreevalue
                count_allusers += 1
    return (res_allusers / count_allusers, res_testusers / count_testusers)


def get_HD(users_test, users_topn, topn):
    # 衡量用户与用户之间的推荐列表的不同程度。
    # 与topn有关。
    # // 在各个数据集的实验中，效果不好，所以决定不使用它。

    haming_distance = 0.0
    count = 0

    users_list = list(users_test.keys())
    for i in range(0, len(users_list) - 1):
        user1 = users_list[i]
        if users_topn[user1] == []:  # boundary condition
            continue
        if len(users_topn[user1]) < topn:  # boundary condition. Jester数据集
            continue
        for j in range(i + 1, len(users_list)):
            user2 = users_list[j]
            if users_topn[user2] == []:  # boundary condition
                continue
            if len(users_topn[user2]) < topn:  # boundary condition. Jester数据集
                continue
            items1_set = set(users_topn[user1][:topn])
            items2_set = set(users_topn[user2][:topn])
            haming_distance += (1.0 - len(items1_set & items2_set) * 1.0 / topn)
            count += 1

    return haming_distance / count


def get_all(users_test, users_topn, topn, ii_similarities=None, items_pair=None, args=None):
    # 统一计算所有评价指标。

    if ii_similarities is None:
        ii_similarities = args.get('similarities_itemswholedataset', None)
    items_degree = None
    items_degreelog1p = None
    if items_pair is None:
        items_degree = args.get('items_degree', None)
        items_degreelog1p = {item: math.log1p(degree) for (item, degree) in items_degree.items()}
    else:
        items_degreelog1p = {item: math.log1p(len(users)) for (item, users) in items_pair.items()}

    nDCG = 0.0
    count_validuser = 0

    hit = 0
    count_recall = 0
    count_precision = 0

    ILD = 0.0
    count_ILD = 0

    novelty_testusers = 0.0
    count_testusers = 0

    gini = 0.0
    items_counts = defaultdict(lambda: 0)
    total_counts = 0

    coverage_testusers = 0.0
    items_recommended_testusers = set()

    HD_HamingDistance = 0.0
    count_HD = 0

    for (user, items_test) in users_test.items():
        # boudnary condition
        if users_topn[user] == []:
            continue
        if len(users_topn[user]) < topn:  # boundary condition. Jester数据集
            continue

        items_topnlist = users_topn[user][:topn]
        items_topnlist_set = set(items_topnlist)

        # 计算DCG
        DCG = 0.0
        count_hit = 0
        for i in range(topn):
            index = i + 1
            if items_topnlist[i] in items_test:
                DCG += 1 / math.log2(1 + index)
                count_hit += 1
            else:
                DCG += 0.0
        # 计算IDCG
        IDCG = sum([1 / math.log2(1 + index) for index in range(1, count_hit + 1)])
        # 最后得到nDCG
        if IDCG == 0.0:
            nDCG += 0.0
            count_validuser += 1
        else:
            nDCG += (DCG / IDCG)
            count_validuser += 1

        # 计算Precision和Recall
        hit += len(items_topnlist_set & items_test)
        count_recall += len(items_test)
        count_precision += topn

        # 计算ILD
        for i in range(0, topn - 1):
            for j in range(i + 1, topn):
                item1 = items_topnlist[i]
                item2 = items_topnlist[j]
                if item2 in ii_similarities[item1]:
                    ILD += ii_similarities[item1][item2]
                    count_ILD += 1
                else:
                    ILD += 0.0
                    count_ILD += 1

        # 计算Novelty和Gini
        for item in items_topnlist:
            dlog1p = items_degreelog1p[item]
            novelty_testusers += dlog1p
            count_testusers += 1

            items_counts[item] += 1
            total_counts += 1

        # 计算Coverage
        items_recommended_testusers |= items_topnlist_set

    # 计算HD
    users_list = list(users_test.keys())
    for i in range(0, len(users_list) - 1):
        user1 = users_list[i]
        if users_topn[user1] == []:  # boundary condition
            continue
        if len(users_topn[user1]) < topn:  # boundary condition. Jester数据集
            continue

        for j in range(i + 1, len(users_list)):
            user2 = users_list[j]
            if users_topn[user2] == []:  # boundary condition
                continue
            if len(users_topn[user2]) < topn:  # boundary condition. Jester数据集
                continue

            items1_set = set(users_topn[user1][:topn])
            items2_set = set(users_topn[user2][:topn])
            HD_HamingDistance += (1.0 - len(items1_set & items2_set) * 1.0 / topn)
            count_HD += 1

    nDCG /= count_validuser
    precision = hit * 1.0 / count_precision
    recall = hit * 1.0 / count_recall
    ILD = 1 - (ILD / count_ILD)
    novelty_testusers /= count_testusers

    items_probability = {item: count for (item, count) in items_counts.items()}  # 已改进。把除法操作（/total_counts）拖延到放到最后再做运算
    items_probability_sorted = sorted(items_probability.items(), key=lambda a: a[1])  # 从小到大递增排序
    n_all = 0
    if items_pair is None:
        n_all = len(items_degree)
    else:
        n_all = len(items_pair)
    n_test = len(items_probability)
    for i in range(n_test):
        probability = items_probability_sorted[i][1]
        index = i + (n_all - n_test) + 1  # 难点。易错，因为有(n_all - n_test)个物品在推荐列表中都没有出现过。覆盖率不为100%时肯定会出现这种情况
        gini += (2 * index - n_all - 1) * probability
    gini = 1 - gini / total_counts / (n_all - 1)

    coverage_testusers = len(items_recommended_testusers) * 1.0 / n_all
    HD_HamingDistance /= count_HD

    metrics = {'nDCG': nDCG, 'Precision': precision, 'Recall': recall,
               'ILD': ILD, 'Novelty': novelty_testusers, 'Gini': gini, 'Coverage': coverage_testusers,
               'HD': HD_HamingDistance}

    return metrics


def get_hds(users_test, users_topn, topn, args):
    # 计算并返回各种HD指标。
    # 与topn有关。
    # // 在各个数据集的实验中，效果不好，所以决定不使用它。

    users_degree = args.get('users_degree', None)
    max_userdegreelog1p = max(math.log1p(degree) for degree in users_degree.values())
    max_square = max_userdegreelog1p * max_userdegreelog1p

    hdold_sum = 0.0
    weightold_sum = 0.0
    hd1_sum = 0.0
    weight1_sum = 0.0
    hd2_sum = 0.0
    weight2_sum = 0.0

    # 列表化遍历，只用计算一半数量的用户u用户v对
    users_list = list(users_test.keys())
    for i in range(0, len(users_list) - 1):
        useru = users_list[i]
        if users_topn[useru] == []:  # boundary condition
            continue
        klogu = math.log1p(users_degree[useru])

        for j in range(i + 1, len(users_list)):
            userv = users_list[j]
            if users_topn[userv] == []:  # boundary condition
                continue
            klogv = math.log1p(users_degree[userv])

            items1_set = set(users_topn[useru][:topn])
            items2_set = set(users_topn[userv][:topn])

            weightold = 1
            hdold = (1 - len(items1_set & items2_set) / topn)
            weightold_sum += weightold
            hdold_sum += hdold * weightold

            # weight1 = min(klogu, klogv) / max_userdegreelog1p * math.pow(0.5, math.fabs(klogu - klogv))
            # hd1 = hdold
            # weight1_sum += weight1
            # hd1_sum += hd1 * weight1

            # weight1 =  min(klogu, klogv) / max_userdegreelog1p
            # hd1 = hdold * math.pow(0.5, math.fabs(klogu - klogv))
            # weight1_sum += weight1
            # hd1_sum += hd1 * weight1

            weight2 = math.pow(0.5, math.fabs(klogu - klogv))
            hd2 = hdold * min(klogu, klogv) / max_userdegreelog1p
            weight2_sum += weight2
            hd2_sum += hd2 * weight2

    return (hdold_sum / weightold_sum, hd2_sum / weight2_sum)


def get_novelty_new(users_test, users_topn, topn, args):
    # 计算新的评价指标Novelty New。

    items_degree = args.get('items_degree', None)
    max_itemdegreelog1p = max(math.log1p(degree) for degree in items_degree.values())
    min_itemdegreelog1p = min(math.log1p(degree) for degree in items_degree.values())

    res = 0.0
    count_validuser = 0
    for (user, items_test) in users_test.items():
        # boundary condition
        if users_topn[user] == []:
            continue

        novelty_norm = 0.0
        items_topnlist = users_topn[user][:topn]
        for i in range(topn):
            index = i + 1
            item_new = items_topnlist[i]
            # 多样性
            klogi = math.log1p(items_degree[item_new])
            novelty_norm += klogi / max_itemdegreelog1p
        novelty_norm = 1 - novelty_norm / topn
        res += novelty_norm
        count_validuser += 1

    return (res / count_validuser)


def get_pd(users_test, users_topn, topn, args):
    # Precision和Diversity结合在一起。

    users_degree = args.get('users_degree', None)
    max_userdegreelog1p = max(math.log1p(degree) for degree in users_degree.values())
    min_userdegreelog1p = min(math.log1p(degree) for degree in users_degree.values())
    items_degree = args.get('items_degree', None)
    max_itemdegreelog1p = max(math.log1p(degree) for degree in items_degree.values())

    pd = 0.0

    count_validuser = 0
    for (user, items_test) in users_test.items():
        # boundary condition
        if users_topn[user] == []:
            continue
        if len(users_topn[user]) < topn:  # boundary condition. Jester数据集
            continue

        nDCG = 0.0
        DCG = 0.0
        novelty_norm = 0.0
        count_hit = 0

        items_topnlist = users_topn[user][:topn]
        for i in range(topn):
            index = i + 1
            item_new = items_topnlist[i]

            # 准确率
            if item_new in items_test:
                DCG += 1 / math.log2(1 + index)
                count_hit += 1
            else:
                DCG += 0.0

            # 多样性
            klogi = math.log1p(items_degree[item_new])
            norm_klogi = klogi / max_itemdegreelog1p
            novelty_norm += norm_klogi

        # 计算IDCG
        IDCG = sum([1 / math.log2(1 + index) for index in range(1, count_hit + 1)])
        # 最后得到nDCG
        if IDCG == 0.0:
            nDCG += 0.0
        else:
            nDCG += (DCG / IDCG)

        novelty_norm = 1 - novelty_norm / topn

        norm_klogu = math.log1p(users_degree[user]) / max_userdegreelog1p
        pd += (1 - norm_klogu) * nDCG + norm_klogu * novelty_norm
        count_validuser += 1

    return pd / count_validuser


def get_pds(users_test, users_topn, topn, args):
    # 返回PD复合评价指标。

    users_degree = args.get('users_degree', None)
    max_userdegreelog1p = max(math.log1p(degree) for degree in users_degree.values())
    items_degree = args.get('items_degree', None)
    max_itemdegreelog1p = max(math.log1p(degree) for degree in items_degree.values())
    max_itemdegree = max(items_degree.values())
    similarities_itemswholedataset = args.get('similarities_itemswholedataset', None)

    pd1 = 0.0
    pd2 = 0.0

    count_validuser = 0
    for (user, items_test) in users_test.items():
        # boundary condition
        if users_topn[user] == []:
            continue

        nDCG = 0.0
        DCG = 0.0
        count_hit = 0
        noveltynorm = 0.0
        ILD = 0.0

        items_topnlist = users_topn[user][:topn]
        for i in range(topn):
            index = i + 1
            item_new = items_topnlist[i]

            # 计算DCG
            if item_new in items_test:
                DCG += 1 / math.log2(1 + index)
                count_hit += 1
            else:
                DCG += 0.0

            # 计算Novelty_Norm
            # klogi = math.log1p(items_degree[item_new])
            # norm_klogi = klogi / max_itemdegreelog1p
            ki = items_degree[item_new]
            norm_ki = ki / max_itemdegree
            noveltynorm += norm_ki

        # 计算ILD
        for i in range(0, topn - 1):
            for j in range(i + 1, topn):
                item1 = items_topnlist[i]
                item2 = items_topnlist[j]
                if item2 in similarities_itemswholedataset[item1]:
                    ILD += similarities_itemswholedataset[item1][item2]
        ILD = 1 - ILD / (topn * (topn - 1))

        # 计算nDCG
        IDCG = sum([1 / math.log2(1 + index) for index in range(1, count_hit + 1)])
        if IDCG == 0.0:
            nDCG += 0.0
        else:
            nDCG += (DCG / IDCG)

        # 计算Novelty_Norm
        noveltynorm = 1 - noveltynorm / topn

        norm_klogu = math.log1p(users_degree[user]) / max_userdegreelog1p
        pd1 += (1 - norm_klogu) * nDCG + norm_klogu * ILD
        pd2 += (1 - norm_klogu) * nDCG + norm_klogu * noveltynorm
        count_validuser += 1

    return (pd1/count_validuser, pd2/count_validuser)


def get_pds2(users_test, users_topn, topn, args):
    # 返回PD复合评价指标。Metric = nDCG + HD。

    users_degree = args.get('users_degree', None)
    max_userdegreelog1p = max(math.log1p(degree) for degree in users_degree.values())
    # items_degree = args.get('items_degree', None)
    # max_itemdegreelog1p = max(math.log1p(degree) for degree in items_degree.values())
    # max_itemdegree = max(items_degree.values())

    pd = 0.0
    HD_all = 0.0
    nDCG_all = 0.0

    count_validuser = 0
    for (user, items_test) in users_test.items():
        # # 进度条
        # if count_validuser % 1000 == 0:
        #     print('已处理{0}用户...'.format(count_validuser))

        klogu = math.log1p(users_degree[user])
        # boundary condition
        if users_topn[user] == []:
            continue

        nDCG = 0.0
        DCG = 0.0
        count_hit = 0
        HD = 0.0
        HD_weight = 0.0

        items_topnlist = users_topn[user][:topn]
        items_topnlist_u = set(items_topnlist)
        for i in range(topn):
            index = i + 1
            item_new = items_topnlist[i]

            # 计算DCG
            if item_new in items_test:
                DCG += 1 / math.log2(1 + index)
                count_hit += 1
            else:
                DCG += 0.0

        # 计算Novelty_Norm
        count_validuserv = 0
        for (userv, items_test) in users_test.items():
            # boundary condition
            if userv == user:
                continue
            if users_topn[userv] == []:
                continue
            klogv = math.log1p(users_degree[userv])
            hd_weight = math.pow(0.5, math.fabs(klogu - klogv))
            items_topnlist_v = set(users_topn[userv][:topn])
            hd = 1 - len(items_topnlist_u & items_topnlist_v) / topn
            HD_weight += hd_weight
            HD += hd * hd_weight
            count_validuserv += 1
        HD /= HD_weight

        # 计算nDCG
        IDCG = sum([1 / math.log2(1 + index) for index in range(1, count_hit + 1)])
        if IDCG == 0.0:
            nDCG += 0.0
        else:
            nDCG += (DCG / IDCG)

        HD_all += HD
        nDCG_all += nDCG
        norm_klogu = math.log1p(users_degree[user]) / max_userdegreelog1p
        pd += (1 - norm_klogu) * nDCG + norm_klogu * HD
        count_validuser += 1

    return (nDCG_all / count_validuser, HD_all / count_validuser, pd / count_validuser)


def get_threemetrics(users_test, users_topn, topn, args):
    # 返回nDCG、ILD和Noveltynorm三个指标。

    users_degree = args.get('users_degree', None)
    max_userdegreelog1p = max(math.log1p(degree) for degree in users_degree.values())
    items_degree = args.get('items_degree', None)
    max_itemdegreelog1p = max(math.log1p(degree) for degree in items_degree.values())
    similarities_itemswholedataset = args.get('similarities_itemswholedataset', None)

    nDCG_all = 0.0
    ILD_all = 0.0
    Novelty_all = 0.0
    count_validuser = 0
    for (user, items_test) in users_test.items():
        # boundary condition
        if users_topn[user] == []:
            continue

        nDCG = 0.0
        DCG = 0.0
        count_hit = 0
        noveltynorm = 0.0
        ILD = 0.0

        items_topnlist = users_topn[user][:topn]
        for i in range(topn):
            index = i + 1
            item_new = items_topnlist[i]

            # 计算DCG
            if item_new in items_test:
                DCG += 1 / math.log2(1 + index)
                count_hit += 1
            else:
                DCG += 0.0

            # 计算Novelty_Norm
            klogi = math.log1p(items_degree[item_new])
            norm_klogi = klogi / max_itemdegreelog1p
            noveltynorm += norm_klogi

        # 计算ILD
        for i in range(0, topn - 1):
            for j in range(i + 1, topn):
                item1 = items_topnlist[i]
                item2 = items_topnlist[j]
                if item2 in similarities_itemswholedataset[item1]:
                    ILD += similarities_itemswholedataset[item1][item2]
        ILD = 1 - ILD / (topn * (topn - 1) / 2)

        # 计算nDCG
        IDCG = sum([1 / math.log2(1 + index) for index in range(1, count_hit + 1)])
        if IDCG == 0.0:
            nDCG += 0.0
        else:
            nDCG += (DCG / IDCG)

        # 计算Novelty_Norm
        noveltynorm = 1 - noveltynorm / topn

        nDCG_all += nDCG
        ILD_all += ILD
        Novelty_all += noveltynorm
        count_validuser += 1

    return (nDCG_all/count_validuser, ILD_all/count_validuser, Novelty_all/count_validuser)


def get_twometrics(users_test, users_topn, topn, args):
    # 计算并返回nDCG和经过改进后的HD指标。

    users_degree = args.get('users_degree', None)
    max_userdegreelog1p = max(math.log1p(degree) for degree in users_degree.values())
    # items_degree = args.get('items_degree', None)
    # max_itemdegreelog1p = max(math.log1p(degree) for degree in items_degree.values())
    # max_itemdegree = max(items_degree.values())

    HD_all = 0.0
    nDCG_all = 0.0

    count_validuser = 0
    for (user, items_test) in users_test.items():
        # # 进度条
        # if count_validuser % 1000 == 0:
        #     print('已处理{0}用户...'.format(count_validuser))

        klogu = math.log1p(users_degree[user])
        # boundary condition
        if users_topn[user] == []:
            continue

        nDCG = 0.0
        DCG = 0.0
        count_hit = 0
        HD = 0.0
        HD_weight = 0.0

        items_topnlist = users_topn[user][:topn]
        items_topnlist_u = set(items_topnlist)
        for i in range(topn):
            index = i + 1
            item_new = items_topnlist[i]

            # 计算DCG
            if item_new in items_test:
                DCG += 1 / math.log2(1 + index)
                count_hit += 1
            else:
                DCG += 0.0

        # 计算Novelty_Norm
        count_validuserv = 0
        for (userv, items_test) in users_test.items():
            # boundary condition
            if userv == user:
                continue
            if users_topn[userv] == []:
                continue
            klogv = math.log1p(users_degree[userv])
            hd_weight = math.pow(0.5, math.fabs(klogu - klogv))
            items_topnlist_v = set(users_topn[userv][:topn])
            hd = 1 - len(items_topnlist_u & items_topnlist_v) / topn
            HD_weight += hd_weight
            HD += hd * hd_weight
            count_validuserv += 1
        HD /= HD_weight

        # 计算nDCG
        IDCG = sum([1 / math.log2(1 + index) for index in range(1, count_hit + 1)])
        if IDCG == 0.0:
            nDCG += 0.0
        else:
            nDCG += (DCG / IDCG)

        HD_all += HD
        nDCG_all += nDCG
        count_validuser += 1

    return (HD_all / count_validuser, nDCG_all / count_validuser)


def diversity_with_topn(current_directory):
    # 画出折线图。横坐标表示topn[20, 100]，纵坐标表示ILD、Novelty、Coverage和HD评价指标。
    # 以MovieLens-1m为主要数据集。以User CF KNN 固定最优参数k为主要推荐算法。

    dataset = 'movielens1m'
    type = '-v1'
    methods = ['User CF Neighbor Optimal']
    arguments = ['user_knn=60']

    topn_files = begin_recommendation(current_directory, dataset, type, methods, arguments)
    topn_file = topn_files[0]  # 只有这一个
    users_topn = {}
    with open(topn_file, 'r') as fr:
        for row in fr.readlines():
            line = row.strip().split(',')
            user = int(line[0])
            if line[1] == '':  # boundary condition
                items = []
                users_topn[user] = items
            else:
                items = [int(item) for item in line[1].split(' ')]
                users_topn[user] = items
    similarities_itemswholedataset_file = r'{0}\{1}\ii_similarities_wholedataset{2}.pickle'.format(current_directory,
                                                                                                   dataset, type)
    testall_file = r'{0}\{1}\{1}_out\testall{2}'.format(current_directory, dataset, type)
    dataset_file = r'{0}\{1}\dataset{2}'.format(current_directory, dataset, type)
    similarities_itemswholedataset = sim.get_iisimilarity_wholedataset(dataset_file,
                                                                       similarities_itemswholedataset_file)
    users_pair_testall = read.read_test(testall_file)
    users_test = users_pair_testall

    print('\n计算多样性评价指标')
    topns = list(range(10, 110, 10))
    ILDs = []
    NVs = []
    CVs = []
    HDs = []
    for topn in topns:
        # 进度条
        print('\r[' + '*' * int(10 * topn / topns[-1]) + '-' * (10 - int(10 * topn / topns[-1])) + ']', end='')

        ILD = get_ILD(similarities_itemswholedataset, users_test, users_topn, topn)
        (novelty_allusers, novelty_testusers) = get_novelty(users_test, users_topn, topn, dataset_file)
        (coverage_allusers, coverage_testusers) = get_coverage(users_test, users_topn, topn, dataset_file)
        HD = get_HD(users_test, users_topn, topn)
        ILDs.append(ILD)
        NVs.append(novelty_testusers)
        CVs.append(coverage_testusers)
        HDs.append(HD)

    # 画图并保存到外存磁盘
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    ax = plt.subplot(111)
    ax.plot(topns, ILDs, 'rx-', mfc='none', label='ILD')
    ax.plot(topns, CVs, 'b*-', mfc='none', label='Coverage')
    ax.plot(topns, HDs, 'c*-', mfc='none', label='HD')
    ax.legend(loc='upper right')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 1])
    ax.set_xlabel('topn')
    ax.set_ylabel('多样性指标')
    plt.grid(True)
    plt.title('topn和多样性指标的关系({0}，{1}推荐算法)'.format(dataset + type, methods[0] + '-' + arguments[0]))
    plt.savefig(current_directory + r'\topn和多样性指标的关系图像.png')
    plt.close('all')

    ax = plt.subplot(111)
    ax.plot(topns, NVs, 'mx-', mfc='none', label='Novelty')
    ax.legend(loc='upper right')
    ax.set_xlim([0, 100])
    ax.set_xlabel('topn')
    ax.set_ylabel('多样性指标')
    plt.grid(True)
    plt.title('topn和多样性指标的关系({0}，{1}推荐算法)'.format(dataset + type, methods[0] + '-' + arguments[0]))
    plt.savefig(current_directory + r'\topn和多样性指标的关系图像2.png')
    plt.close('all')
    return


def begin_recommendation(current_directory, dataset, type, methods, arguments, need_cfsimilarity=True):
    # 做推荐，保存topn_files到外存磁盘。

    train_file = r'{0}\{1}\train{2}'.format(current_directory, dataset, type)
    similarity_usercf_file = r'{0}\{1}\{1}_out\similarities_usercf{2}.pickle'.format(current_directory, dataset, type)
    similarity_itemcf_file = r'{0}\{1}\{1}_out\similarities_itemcf{2}.pickle'.format(current_directory, dataset, type)
    pickle_filepath = r'{0}\{1}\train-seedusers{2}'.format(current_directory, dataset, type)

    #
    (users_pair, items_pair) = read.read_data(train_file)  # 大规模数据集上可能跑不成。可以改成一行一行读入
    users_degree = {user: len(items) for (user, items) in users_pair.items()}  # 新添加
    items_degree = {item: len(users) for (item, users) in items_pair.items()}  # 新添加
    print('暂时不计算用户的协同过滤相似度')
    # similarities_usercf = sim.get_uusimilarity(similarity_usercf_file, users_pair, items_pair)
    similarities_usercf = None
    similarities_itemcf = sim.get_iisimilarity(similarity_itemcf_file, users_pair, items_pair)
    users_seed = None
    # if os.path.exists(pickle_filepath):  # boundary condition
    #     print('probs相似度文件已存在，直接序列化读取它，而不用再重新计算了')
    #     f = open(pickle_filepath, 'rb')
    #     users_seed = pickle.load(f)
    #     f.close()
    # else:
    #     print('Error. 种子用户不存在!')
    #     exit(1)

    #
    topn_files = []
    count = 0
    for method in methods:
        if len(arguments) == 0:  # boundary condition
            # 当前进度
            count += 1
            print('\n当前进度：第{0}种 / 共{1}'.format(count, len(methods)))

            # 确定初始参数
            args = {'users_pair': users_pair,
                    'items_pair': items_pair,
                    'similarities_usercf': similarities_usercf,
                    'similarities_itemcf': similarities_itemcf,
                    'users_degree': users_degree,
                    'items_degree': items_degree,
                    'users_seed': users_seed
                    }

            # 生成推荐列表，保存到外存磁盘
            topn_file = r'{0}\{1}\{1}_out\topn_{2}-{3}-无参数'.format(current_directory, dataset, type, method)
            topn_files.append(topn_file)
            recommend.recommend_onebyone(method, topn_file, current_directory, dataset, type, args)
            continue

        for argument in arguments:
            # 当前进度
            count += 1
            print('\n当前进度：第{0}种 / 共{1}'.format(count, len(methods) * len(arguments)))

            # 确定初始参数
            args = {'users_pair': users_pair,
                    'items_pair': items_pair,
                    'similarities_usercf': similarities_usercf,
                    'similarities_itemcf': similarities_itemcf,
                    'users_degree': users_degree,
                    'items_degree': items_degree
                    }
            if '=' in argument:  # 把新参数添加到args
                argument = argument.split(',')
                for a in argument:
                    kv = a.split('=')
                    if kv[1].isdigit():
                        args[kv[0]] = int(kv[1])
                    else:
                        args[kv[0]] = float(kv[1])
            else:
                argument = "['" + argument + "']"

            # 生成推荐列表，保存到外存磁盘
            topn_file = r'{0}\{1}\{1}_out\topn_{2}-{3}-{4}'.format(current_directory, dataset, type, method,
                                                                   str(argument)[1:-1].strip('\''))
            topn_files.append(topn_file)
            recommend.recommend_onebyone(method, topn_file, current_directory, dataset, type, args)

    # 清除冗余变量，为电脑内存腾空间
    import gc
    del users_pair
    del items_pair
    del similarities_usercf
    del similarities_itemcf
    del users_degree
    del items_degree
    gc.collect()

    return topn_files


def begin_recommendation_fixargument(current_directory, dataset, type, methods_and_arguments, need_cfsimilarity=True):
    # 固定参数。
    # 做推荐，保存topn_files到外存磁盘。

    train_file = r'{0}\{1}\train{2}'.format(current_directory, dataset, type)
    similarity_usercf_file = r'{0}\{1}\{1}_out\similarities_usercf{2}.pickle'.format(current_directory, dataset, type)
    similarity_itemcf_file = r'{0}\{1}\{1}_out\similarities_itemcf{2}.pickle'.format(current_directory, dataset, type)
    pickle_filepath = r'{0}\{1}\train-seedusers-0.01{2}'.format(current_directory, dataset, type)

    #
    (users_pair, items_pair) = read.read_data(train_file)  # 大规模数据集上可能跑不成。可以改成一行一行读入
    users_degree = {user: len(items) for (user, items) in users_pair.items()}  # 新添加
    items_degree = {item: len(users) for (item, users) in items_pair.items()}  # 新添加
    similarities_usercf = sim.get_uusimilarity(similarity_usercf_file, users_pair, items_pair)
    similarities_itemcf = sim.get_iisimilarity(similarity_itemcf_file, users_pair, items_pair)
    users_seed = None
    if os.path.exists(pickle_filepath):  # boundary condition
        print('种子用户文件已存在，直接序列化读取它')
        f = open(pickle_filepath, 'rb')
        users_seed = pickle.load(f)
        f.close()
    else:
        print('Error. 种子用户不存在!')
        exit(1)

    #
    topn_files = []
    count = 0
    for (method, argument) in methods_and_arguments:
        # 当前进度
        count += 1
        print('\n当前进度：第{0}种 / 共{1}'.format(count,
                                           len(methods_and_arguments))
              )

        # 确定初始参数
        args = {'users_pair': users_pair,
                'items_pair': items_pair,
                'similarities_usercf': similarities_usercf,
                'similarities_itemcf': similarities_itemcf,
                'users_degree': users_degree,
                'items_degree': items_degree,
                'users_seed': users_seed
                }
        if '=' in argument:  # 把新参数添加到args
            argument = argument.split(',')
            for a in argument:
                kv = a.split('=')
                if kv[1].isdigit():
                    args[kv[0]] = int(kv[1])
                else:
                    args[kv[0]] = float(kv[1])
        else:
            argument = "['" + argument + "']"

        # 生成推荐列表，保存到外存磁盘
        topn_file = r'{0}\{1}\{1}_out\topn_{2}-{3}-{4}'.format(current_directory, dataset, type, method,
                                                               str(argument)[1:-1].strip('\''))
        topn_files.append(topn_file)
        recommend.recommend_onebyone(method, topn_file, current_directory, dataset, type, args)

    # 清除冗余变量，为电脑内存腾空间
    import gc
    del users_pair
    del items_pair
    del similarities_usercf
    del similarities_itemcf
    del users_degree
    del items_degree
    gc.collect()

    return topn_files


def begin_metric(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files):
    # 开始计算评价指标。

    similarities_itemswholedataset_file = r'{0}\{1}\ii_similarities_wholedataset{2}.pickle'.format(current_directory,
                                                                                                   dataset, type)
    testall_file = r'{0}\{1}\{1}_out\testall{2}'.format(current_directory, dataset, type)
    testbig_file = r'{0}\{1}\{1}_out\testbig{2}'.format(current_directory, dataset, type)
    testmedium_file = r'{0}\{1}\{1}_out\testmedium{2}'.format(current_directory, dataset, type)
    testsmall_file = r'{0}\{1}\{1}_out\testsmall{2}'.format(current_directory, dataset, type)

    # 准备将离线评价指标的计算结果保存在外存磁盘里
    if os.path.exists(r'{0}\{1}\result.txt'.format(current_directory, dataset)):
        with open(r'{0}\{1}\result.txt'.format(current_directory, dataset), 'a') as fw:
            fw.write('\n')
            fw.write('数据集是{0}, 类别是{1}。推荐算法{2}，参数{3}\n'.format(dataset, type, methods, arguments))
            fw.write('nDCG\tPrecision\tRecall\tILD\tNovelty(越小越好)\tGini\tCoverage\tHD\n')
        print('\n{0}\{1}目录下result结果文件已存在，不需要再创建'.format(current_directory, dataset))
    else:
        with open(r'{0}\{1}\result.txt'.format(current_directory, dataset), 'a') as fw:
            fw.write('数据集是{0}，类别是{1}。推荐算法{2}，参数{3}\n'.format(dataset, type, methods, arguments))
            fw.write('nDCG\tPrecision\tRecall\tILD\tNovelty(越小越好)\tGini\tCoverage\tHD\n')
        print('\n已在{0}\{1}目录下result结果文件创建成功'.format(current_directory, dataset))

    # 读入推荐列表，计算评价指标
    dataset_file = r'{0}\{1}\dataset{2}'.format(current_directory, dataset, type)
    (users_pair, items_pair) = read.read_data(dataset_file)
    similarities_itemswholedataset = sim.get_iisimilarity_wholedataset(dataset_file,
                                                                       similarities_itemswholedataset_file)
    users_pair_testall = read.read_test(testall_file)
    users_pair_testbig = read.read_test(testbig_file)
    users_pair_testmedium = read.read_test(testmedium_file)
    users_pair_testsmall = read.read_test(testsmall_file)

    print()
    count = 0
    for test in test_type:
        # 当前进度
        count += 1
        print('{0}测试集用户：计算评价指标...'.format(test))

        users_test = None
        if test == 'all':
            users_test = users_pair_testall
        elif test == 'big':
            users_test = users_pair_testbig
        elif test == 'medium':
            if users_pair_testmedium == None:  # boundary condition
                print('Error. test medium is None!')
                exit(1)
            users_test = users_pair_testmedium
        elif test == 'small':
            users_test = users_pair_testsmall
        else:
            print('Error. test type is wrong!')
            exit(1)

        count_topn_file = 0
        for topn_file in topn_files:
            # 图形进度条
            count_topn_file += 1
            print('\r[' + '*' * int(10 * count_topn_file / len(topn_files)) + '-' * (
            10 - int(10 * count_topn_file / len(topn_files))) + ']', end='')

            users_topn = {}
            with open(topn_file, 'r') as fr:
                for row in fr.readlines():
                    line = row.strip().split(',')
                    user = int(line[0])
                    if line[1] == '':  # boundary condition
                        items = []
                        users_topn[user] = items
                    else:
                        items = [int(item) for item in line[1].split(' ')]
                        users_topn[user] = items

            # 开始计算评价指标
            # ndcg = get_nDCG(users_test, users_topn, topn)
            # (p, r) = get_PrecisionRecall(users_test, users_topn, topn)
            # intra_dissimilarity = get_ILD(similarities_itemswholedataset, users_test, users_topn, topn)
            # gini = get_gini(users_test, users_topn, topn, dataset_file)
            # (coverage_allusers, coverage_testusers) = get_coverage(users_test, users_topn, topn, dataset_file)
            # (novelty_allusers, novelty_testusers) = get_novelty(users_test, users_topn, topn, dataset_file)
            # haming = get_HD(users_test, users_topn, topn)
            metrics = get_all(users_test, users_topn, topn, ii_similarities=similarities_itemswholedataset,
                              items_pair=items_pair)

            with open(r'{0}\{1}\result.txt'.format(current_directory, dataset), 'a') as fw:
                fw.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'.format(round(metrics['nDCG'], 3),
                                                                           round(metrics['Precision'], 3),
                                                                           round(metrics['Recall'], 3),
                                                                           round(metrics['ILD'], 3),
                                                                           round(metrics['Novelty'], 3),
                                                                           round(metrics['Gini'], 3),
                                                                           round(metrics['Coverage'], 3),
                                                                           round(metrics['HD'], 3))
                         )

        print()
        with open(r'{0}\{1}\result.txt'.format(current_directory, dataset), 'a') as fw:
            fw.write('\n')
    return


def begin_metric_hds(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files):
    # 计算各种改进后的新指标。

    testall_file = r'{0}\{1}\{1}_out\testall{2}'.format(current_directory, dataset, type)
    testbig_file = r'{0}\{1}\{1}_out\testbig{2}'.format(current_directory, dataset, type)
    testmedium_file = r'{0}\{1}\{1}_out\testmedium{2}'.format(current_directory, dataset, type)
    testsmall_file = r'{0}\{1}\{1}_out\testsmall{2}'.format(current_directory, dataset, type)

    # 读入推荐列表，计算评价指标
    dataset_file = r'{0}\{1}\dataset{2}'.format(current_directory, dataset, type)
    (users_pair, items_pair) = read.read_data(dataset_file)
    users_degree = {user: len(items) for (user, items) in users_pair.items()}  # 新添加
    items_degree = {item: len(users) for (item, users) in items_pair.items()}  # 新添加

    # 清除冗余变量，为电脑内存腾空间
    import gc
    del users_pair
    del items_pair
    gc.collect()

    args = {'users_degree': users_degree,
            'items_degree': items_degree
            }
    users_pair_testall = read.read_test(testall_file)
    users_pair_testbig = read.read_test(testbig_file)
    users_pair_testmedium = read.read_test(testmedium_file)
    users_pair_testsmall = read.read_test(testsmall_file)

    count = 0
    for test in test_type:
        # 当前进度
        count += 1
        print('{0}测试集用户：计算评价指标...'.format(test))

        users_test = None
        if test == 'all':
            users_test = users_pair_testall
        elif test == 'big':
            users_test = users_pair_testbig
        elif test == 'medium':
            if users_pair_testmedium == None:  # boundary condition
                print('Error. test medium is None!')
                exit(1)
            users_test = users_pair_testmedium
        elif test == 'small':
            users_test = users_pair_testsmall
        else:
            print('Error. test type is wrong!')
            exit(1)

        for topn_file in topn_files:
            # 读入推荐列表
            users_topn = {}
            with open(topn_file, 'r') as fr:
                for row in fr.readlines():
                    line = row.strip().split(',')
                    user = int(line[0])
                    if line[1] == '':  # boundary condition
                        items = []
                        users_topn[user] = items
                    else:
                        items = [int(item) for item in line[1].split(' ')]
                        users_topn[user] = items

            # 开始计算评价指标

            (hdold, hd1) = get_hds(users_test, users_topn, topn, args)
            hdold = round(hdold, 3)
            hd1 = round(hd1, 3)
            print('{0}\t{1}'.format(hdold, hd1))

            # (pd1, pd2) = get_pds(users_test, users_topn, topn, args)
            # pd1 = round(pd1, 3)
            # pd2 = round(pd2, 3)
            # print('{0}\t{1}'.format(pd1, pd2))

            # (nDCG, ILD, Noveltynorm) = get_threemetrics(users_test, users_topn, topn, args)
            # nDCG = round(nDCG, 3)
            # ILD = round(ILD, 3)
            # Noveltynorm = round(Noveltynorm, 3)
            # print('{0}\t{1}\t{2}'.format(nDCG, ILD, Noveltynorm))

    return


def begin_metric_mts(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files):
    # 计算各种改进后的新指标。

    testall_file = r'{0}\{1}\{1}_out\testall{2}'.format(current_directory, dataset, type)
    testbig_file = r'{0}\{1}\{1}_out\testbig{2}'.format(current_directory, dataset, type)
    testmedium_file = r'{0}\{1}\{1}_out\testmedium{2}'.format(current_directory, dataset, type)
    testsmall_file = r'{0}\{1}\{1}_out\testsmall{2}'.format(current_directory, dataset, type)

    # 读入推荐列表，计算评价指标
    dataset_file = r'{0}\{1}\dataset{2}'.format(current_directory, dataset, type)
    (users_pair, items_pair) = read.read_data(dataset_file)
    users_degree = {user: len(items) for (user, items) in users_pair.items()}  # 新添加
    items_degree = {item: len(users) for (item, users) in items_pair.items()}  # 新添加

    # 清除冗余变量，为电脑内存腾空间
    import gc
    del users_pair
    del items_pair
    gc.collect()

    similarities_itemswholedataset_file = r'{0}\{1}\ii_similarities_wholedataset{2}.pickle'.format(current_directory, dataset, type)
    similarities_itemswholedataset = sim.get_iisimilarity_wholedataset(dataset_file, similarities_itemswholedataset_file)
    args = {'users_degree': users_degree,
            'items_degree': items_degree,
            'similarities_itemswholedataset': similarities_itemswholedataset
            }
    users_pair_testall = read.read_test(testall_file)
    users_pair_testbig = read.read_test(testbig_file)
    users_pair_testmedium = read.read_test(testmedium_file)
    users_pair_testsmall = read.read_test(testsmall_file)

    count = 0
    for test in test_type:
        # 当前进度
        count += 1
        print('{0}测试集用户：计算评价指标...'.format(test))

        users_test = None
        if test == 'all':
            users_test = users_pair_testall
        elif test == 'big':
            users_test = users_pair_testbig
        elif test == 'medium':
            if users_pair_testmedium == None:  # boundary condition
                print('Error. test medium is None!')
                exit(1)
            users_test = users_pair_testmedium
        elif test == 'small':
            users_test = users_pair_testsmall
        else:
            print('Error. test type is wrong!')
            exit(1)

        for topn_file in topn_files:
            # 读入推荐列表
            users_topn = {}
            with open(topn_file, 'r') as fr:
                for row in fr.readlines():
                    line = row.strip().split(',')
                    user = int(line[0])
                    if line[1] == '':  # boundary condition
                        items = []
                        users_topn[user] = items
                    else:
                        items = [int(item) for item in line[1].split(' ')]
                        users_topn[user] = items

            # 开始计算评价指标

            # (hdold, hd1) = get_hds(users_test, users_topn, topn, args)
            # hdold = round(hdold, 3)
            # hd1 = round(hd1, 3)
            # print('{0}\t{1}'.format(hdold, hd1))

            (pd1, pd2) = get_pds(users_test, users_topn, topn, args)
            pd1 = round(pd1, 3)
            pd2 = round(pd2, 3)
            print('{0}\t{1}'.format(pd1, pd2))

            # (nDCG, ILD, Noveltynorm) = get_threemetrics(users_test, users_topn, topn, args)
            # nDCG = round(nDCG, 3)
            # ILD = round(ILD, 3)
            # Noveltynorm = round(Noveltynorm, 3)
            # print('{0}\t{1}\t{2}'.format(nDCG, ILD, Noveltynorm))

    return


def begin_metric_mts2(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files):
    # 计算各种改进后的新指标。

    testall_file = r'{0}\{1}\{1}_out\testall{2}'.format(current_directory, dataset, type)
    testbig_file = r'{0}\{1}\{1}_out\testbig{2}'.format(current_directory, dataset, type)
    testmedium_file = r'{0}\{1}\{1}_out\testmedium{2}'.format(current_directory, dataset, type)
    testsmall_file = r'{0}\{1}\{1}_out\testsmall{2}'.format(current_directory, dataset, type)

    # 读入推荐列表，计算评价指标
    dataset_file = r'{0}\{1}\dataset{2}'.format(current_directory, dataset, type)
    (users_pair, items_pair) = read.read_data(dataset_file)
    users_degree = {user: len(items) for (user, items) in users_pair.items()}  # 新添加
    items_degree = {item: len(users) for (item, users) in items_pair.items()}  # 新添加

    # 清除冗余变量，为电脑内存腾空间
    import gc
    del users_pair
    del items_pair
    gc.collect()

    args = {'users_degree': users_degree,
            'items_degree': items_degree
            }
    users_pair_testall = read.read_test(testall_file)
    users_pair_testbig = read.read_test(testbig_file)
    users_pair_testmedium = read.read_test(testmedium_file)
    users_pair_testsmall = read.read_test(testsmall_file)

    count = 0
    for test in test_type:
        # 当前进度
        count += 1
        print('{0}测试集用户：计算评价指标...'.format(test))

        users_test = None
        if test == 'all':
            users_test = users_pair_testall
        elif test == 'big':
            users_test = users_pair_testbig
        elif test == 'medium':
            if users_pair_testmedium == None:  # boundary condition
                print('Error. test medium is None!')
                exit(1)
            users_test = users_pair_testmedium
        elif test == 'small':
            users_test = users_pair_testsmall
        else:
            print('Error. test type is wrong!')
            exit(1)

        for topn_file in topn_files:
            # 读入推荐列表
            users_topn = {}
            with open(topn_file, 'r') as fr:
                for row in fr.readlines():
                    line = row.strip().split(',')
                    user = int(line[0])
                    if line[1] == '':  # boundary condition
                        items = []
                        users_topn[user] = items
                    else:
                        items = [int(item) for item in line[1].split(' ')]
                        users_topn[user] = items

            # 开始计算评价指标

            # (hdold, hd1) = get_hds(users_test, users_topn, topn, args)
            # hdold = round(hdold, 3)
            # hd1 = round(hd1, 3)
            # print('{0}\t{1}'.format(hdold, hd1))

            (nDCG, HD, Metric) = get_pds2(users_test, users_topn, topn, args)
            nDCG = round(nDCG, 3)
            HD = round(HD, 3)
            Metric = round(Metric, 3)
            print('{0}\t{1}\t{2}'.format(nDCG, HD, Metric))

            # (nDCG, ILD, Noveltynorm) = get_threemetrics(users_test, users_topn, topn, args)
            # nDCG = round(nDCG, 3)
            # ILD = round(ILD, 3)
            # Noveltynorm = round(Noveltynorm, 3)
            # print('{0}\t{1}\t{2}'.format(nDCG, ILD, Noveltynorm))

    return


def begin_metric_three(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files):
    # 计算各种改进后的新指标。

    testall_file = r'{0}\{1}\{1}_out\testall{2}'.format(current_directory, dataset, type)
    testbig_file = r'{0}\{1}\{1}_out\testbig{2}'.format(current_directory, dataset, type)
    testmedium_file = r'{0}\{1}\{1}_out\testmedium{2}'.format(current_directory, dataset, type)
    testsmall_file = r'{0}\{1}\{1}_out\testsmall{2}'.format(current_directory, dataset, type)

    # 读入推荐列表，计算评价指标
    dataset_file = r'{0}\{1}\dataset{2}'.format(current_directory, dataset, type)
    (users_pair, items_pair) = read.read_data(dataset_file)
    users_degree = {user: len(items) for (user, items) in users_pair.items()}  # 新添加
    items_degree = {item: len(users) for (item, users) in items_pair.items()}  # 新添加

    # 清除冗余变量，为电脑内存腾空间
    import gc
    del users_pair
    del items_pair
    gc.collect()

    similarities_itemswholedataset_file = r'{0}\{1}\ii_similarities_wholedataset{2}.pickle'.format(current_directory, dataset, type)
    similarities_itemswholedataset = sim.get_iisimilarity_wholedataset(dataset_file, similarities_itemswholedataset_file)
    args = {'users_degree': users_degree,
            'items_degree': items_degree,
            'similarities_itemswholedataset': similarities_itemswholedataset
            }
    users_pair_testall = read.read_test(testall_file)
    users_pair_testbig = read.read_test(testbig_file)
    users_pair_testmedium = read.read_test(testmedium_file)
    users_pair_testsmall = read.read_test(testsmall_file)

    count = 0
    for test in test_type:
        # 当前进度
        count += 1
        print('{0}测试集用户：计算评价指标...'.format(test))

        users_test = None
        if test == 'all':
            users_test = users_pair_testall
        elif test == 'big':
            users_test = users_pair_testbig
        elif test == 'medium':
            if users_pair_testmedium == None:  # boundary condition
                print('Error. test medium is None!')
                exit(1)
            users_test = users_pair_testmedium
        elif test == 'small':
            users_test = users_pair_testsmall
        else:
            print('Error. test type is wrong!')
            exit(1)

        for topn_file in topn_files:
            # 读入推荐列表
            users_topn = {}
            with open(topn_file, 'r') as fr:
                for row in fr.readlines():
                    line = row.strip().split(',')
                    user = int(line[0])
                    if line[1] == '':  # boundary condition
                        items = []
                        users_topn[user] = items
                    else:
                        items = [int(item) for item in line[1].split(' ')]
                        users_topn[user] = items

            # 开始计算评价指标

            # (hdold, hd1) = get_hds(users_test, users_topn, topn, args)
            # hdold = round(hdold, 3)
            # hd1 = round(hd1, 3)
            # print('{0}\t{1}'.format(hdold, hd1))

            # (pd1, pd2) = get_pds(users_test, users_topn, topn, args)
            # pd1 = round(pd1, 3)
            # pd2 = round(pd2, 3)
            # print('{0}\t{1}'.format(pd1, pd2))

            (nDCG, ILD, Noveltynorm) = get_threemetrics(users_test, users_topn, topn, args)
            nDCG = round(nDCG, 3)
            ILD = round(ILD, 3)
            Noveltynorm = round(Noveltynorm, 3)
            print('{0}\t{1}\t{2}'.format(nDCG, ILD, Noveltynorm))

    return


def begin_metric_two(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files):
    # 计算各种改进后的新指标。

    testall_file = r'{0}\{1}\{1}_out\testall{2}'.format(current_directory, dataset, type)
    testbig_file = r'{0}\{1}\{1}_out\testbig{2}'.format(current_directory, dataset, type)
    testmedium_file = r'{0}\{1}\{1}_out\testmedium{2}'.format(current_directory, dataset, type)
    testsmall_file = r'{0}\{1}\{1}_out\testsmall{2}'.format(current_directory, dataset, type)

    # 读入推荐列表，计算评价指标
    dataset_file = r'{0}\{1}\dataset{2}'.format(current_directory, dataset, type)
    (users_pair, items_pair) = read.read_data(dataset_file)
    users_degree = {user: len(items) for (user, items) in users_pair.items()}  # 新添加
    items_degree = {item: len(users) for (item, users) in items_pair.items()}  # 新添加

    # 清除冗余变量，为电脑内存腾空间
    import gc
    del users_pair
    del items_pair
    gc.collect()

    args = {'users_degree': users_degree,
            'items_degree': items_degree
            }
    users_pair_testall = read.read_test(testall_file)
    users_pair_testbig = read.read_test(testbig_file)
    users_pair_testmedium = read.read_test(testmedium_file)
    users_pair_testsmall = read.read_test(testsmall_file)

    count = 0
    for test in test_type:
        # 当前进度
        count += 1
        print('{0}测试集用户：计算评价指标...'.format(test))

        users_test = None
        if test == 'all':
            users_test = users_pair_testall
        elif test == 'big':
            users_test = users_pair_testbig
        elif test == 'medium':
            if users_pair_testmedium == None:  # boundary condition
                print('Error. test medium is None!')
                exit(1)
            users_test = users_pair_testmedium
        elif test == 'small':
            users_test = users_pair_testsmall
        else:
            print('Error. test type is wrong!')
            exit(1)

        for topn_file in topn_files:
            # 读入推荐列表
            users_topn = {}
            with open(topn_file, 'r') as fr:
                for row in fr.readlines():
                    line = row.strip().split(',')
                    user = int(line[0])
                    if line[1] == '':  # boundary condition
                        items = []
                        users_topn[user] = items
                    else:
                        items = [int(item) for item in line[1].split(' ')]
                        users_topn[user] = items

            # 开始计算评价指标

            # (hdold, hd1) = get_hds(users_test, users_topn, topn, args)
            # hdold = round(hdold, 3)
            # hd1 = round(hd1, 3)
            # print('{0}\t{1}'.format(hdold, hd1))

            # (pd1, pd2) = get_pds(users_test, users_topn, topn, args)
            # pd1 = round(pd1, 3)
            # pd2 = round(pd2, 3)
            # print('{0}\t{1}'.format(pd1, pd2))

            (HD, nDCG) = get_twometrics(users_test, users_topn, topn, args)
            HD = round(HD, 3)
            nDCG = round(nDCG, 3)
            print('{0}\t{1}'.format(nDCG, HD))

    return


def begin_metric_all(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files):
    # 计算各种改进后的新指标。

    testall_file = r'{0}\{1}\{1}_out\testall{2}'.format(current_directory, dataset, type)
    testbig_file = r'{0}\{1}\{1}_out\testbig{2}'.format(current_directory, dataset, type)
    testmedium_file = r'{0}\{1}\{1}_out\testmedium{2}'.format(current_directory, dataset, type)
    testsmall_file = r'{0}\{1}\{1}_out\testsmall{2}'.format(current_directory, dataset, type)

    # 读入推荐列表，计算评价指标
    dataset_file = r'{0}\{1}\dataset{2}'.format(current_directory, dataset, type)
    (users_pair, items_pair) = read.read_data(dataset_file)
    users_degree = {user: len(items) for (user, items) in users_pair.items()}  # 新添加
    items_degree = {item: len(users) for (item, users) in items_pair.items()}  # 新添加

    # 清除冗余变量，为电脑内存腾空间
    import gc
    del users_pair
    del items_pair
    gc.collect()

    similarities_itemswholedataset_file = r'{0}\{1}\ii_similarities_wholedataset{2}.pickle'.format(current_directory, dataset, type)
    similarities_itemswholedataset = sim.get_iisimilarity_wholedataset(dataset_file, similarities_itemswholedataset_file)
    args = {'users_degree': users_degree,
            'items_degree': items_degree,
            'similarities_itemswholedataset': similarities_itemswholedataset
            }
    users_pair_testall = read.read_test(testall_file)
    users_pair_testbig = read.read_test(testbig_file)
    users_pair_testmedium = read.read_test(testmedium_file)
    users_pair_testsmall = read.read_test(testsmall_file)

    count = 0
    for test in test_type:
        # 当前进度
        count += 1
        print('{0}测试集用户：计算评价指标...'.format(test))

        users_test = None
        if test == 'all':
            users_test = users_pair_testall
        elif test == 'big':
            users_test = users_pair_testbig
        elif test == 'medium':
            if users_pair_testmedium == None:  # boundary condition
                print('Error. test medium is None!')
                exit(1)
            users_test = users_pair_testmedium
        elif test == 'small':
            users_test = users_pair_testsmall
        else:
            print('Error. test type is wrong!')
            exit(1)

        for topn_file in topn_files:
            # 读入推荐列表
            users_topn = {}
            with open(topn_file, 'r') as fr:
                for row in fr.readlines():
                    line = row.strip().split(',')
                    user = int(line[0])
                    if line[1] == '':  # boundary condition
                        items = []
                        users_topn[user] = items
                    else:
                        items = [int(item) for item in line[1].split(' ')]
                        users_topn[user] = items

            # 开始计算评价指标

            # (hdold, hd1) = get_hds(users_test, users_topn, topn, args)
            # hdold = round(hdold, 3)
            # hd1 = round(hd1, 3)
            # print('{0}\t{1}'.format(hdold, hd1))

            # (pd1, pd2) = get_pds(users_test, users_topn, topn, args)
            # pd1 = round(pd1, 3)
            # pd2 = round(pd2, 3)
            # print('{0}\t{1}'.format(pd1, pd2))

            metrics = get_all(users_test, users_topn, topn, args=args)
            nDCG = metrics['nDCG']
            Precision = metrics['Precision']
            Recall = metrics['Recall']
            ILD = metrics['ILD']
            Novelty = metrics['Novelty']
            Gini = metrics['Gini']
            Coverage = metrics['Coverage']
            HD = metrics['HD']

            nDCG = round(nDCG, 3)
            Precision = round(Precision, 3)
            Recall = round(Recall, 3)
            ILD = round(ILD, 3)
            Novelty = round(Novelty, 3)
            Gini = round(Gini, 3)
            Coverage = round(Coverage, 3)
            HD = round(HD, 3)

            print('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}'.format(nDCG, Precision, Recall,
                                                                  ILD, Novelty,
                                                                  Gini, Coverage, HD))

    return


def score_itemdegree(current_directory, dataset, type, topn_file):
    # 得到结果并打印。
    #  methods中只能有一个元素。

    # 读入推荐列表，长度为100
    users_topn = {}
    with open(topn_file, 'r') as fr:
        for row in fr.readlines():
            line = row.strip().split(',')
            user = int(line[0])
            if line[1] == '':  # boundary condition
                items = []
                users_topn[user] = items
            else:
                items = [int(item) for item in line[1].split(' ')]
                users_topn[user] = items

    # 读入系统中的物品流行度
    dataset_file = r'{0}\{1}\dataset{2}'.format(current_directory, dataset, type)
    (users_pair, items_pair) = read.read_data(dataset_file)  # 大规模数据集上可能跑不成。可以改成一行一行读入
    items_degree = {item: len(users) for (item, users) in items_pair.items()}  # 新添加

    testbig_file = r'{0}\{1}\{1}_out\testbig{2}'.format(current_directory, dataset, type)
    testmedium_file = r'{0}\{1}\{1}_out\testmedium{2}'.format(current_directory, dataset, type)
    testsmall_file = r'{0}\{1}\{1}_out\testsmall{2}'.format(current_directory, dataset, type)
    users_pair_testbig = read.read_test(testbig_file)
    users_pair_testmedium = read.read_test(testmedium_file)
    users_pair_testsmall = read.read_test(testsmall_file)

    # 计算High、Medium和Low三部分的前100推荐列表的平均物品流行度
    logdegree_testbig = 0
    count_big = 0
    for user in users_pair_testbig:
        items_list = users_topn[user]
        logdegree_testbig += sum([math.log1p(items_degree[item]) for item in items_list])
        count_big += 1
    logdegree_testbig /= (count_big * 100)

    logdegree_testmedium = 0
    count_medium = 0
    for user in users_pair_testmedium:
        items_list = users_topn[user]
        logdegree_testmedium += sum(math.log1p(items_degree[item]) for item in items_list)
        count_medium += 1
    logdegree_testmedium /= (count_medium * 100)

    logdegree_testsmall = 0
    count_small = 0
    for user in users_pair_testsmall:
        items_list = users_topn[user]
        logdegree_testsmall += sum([math.log1p(items_degree[item]) for item in items_list])
        count_small += 1
    logdegree_testsmall /= (count_small * 100)

    print('High: {0}, Medium: {1}, Low: {2}'.format(logdegree_testbig, logdegree_testmedium, logdegree_testsmall))
    return


if __name__ == '__main__':
    type = '-v1'
    current_directory = r'D:\recommender_data'

    # # need change
    # import time
    # print('ICF系列：等待1小时...')
    # time.sleep(3600)

    for dataset in [  # need change
        ''
        # 'netflix', 'movielens1m', 'msd'
    ]:
        # dataset = 'msd'
        # print(dataset, ': 统计推荐结果的平均物品流行度')
        # methods = ['User CF q', 'Item CF kNN Norm']
        # arguments = ['q=3', 'item_knn=10']
        # for (method, argument) in list(zip(methods, arguments)):
        #     print(method)
        #     topn_file = r'{0}\{1}\{1}_out\topn_-v1-{2}-{3}'.format(current_directory, dataset, method, argument)
        #     score_itemdegree(current_directory, dataset, '-v1', topn_file)
        #     print()

        # dataset = 'msd'
        # print(dataset, ': nDCG, HD和Metric')
        # methods = [
        #     'HPH', 'HPH'
        # ]
        # arguments = ['itemdegree_lambda=0.9']
        # topn_files = []
        # for (method, argument) in list(zip(methods, arguments)):
        #     topn_file = r'{0}\{1}\{1}_out\topn_{2}-{3}-{4}'.format(current_directory, dataset, type, method, argument)
        #     topn_files.append(topn_file)
        # topn = 50
        # test_type = ['all']  # need change 可以加上'all'
        # begin_metric_mts2(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files)
        # print()

        # dataset = 'movielens100k'
        # print(dataset, '：复合指标')
        # methods = [
        #     'ProbS',
        #     'HPH',
        #     'PD',
        #     'ProbS Reranking DI TOPSIS'
        # ]
        # arguments = None
        # if dataset == 'movielens100k':
        #     arguments = ['无参数', 'itemdegree_lambda=0.3', 'itemdegree_x=-0.9', '无参数']
        # if dataset == 'movielens1m':
        #     arguments = ['无参数', 'itemdegree_lambda=0.2', 'itemdegree_x=-0.9', '无参数']
        # if dataset == 'netflix':
        #     arguments = ['无参数', 'itemdegree_lambda=0.2', 'itemdegree_x=-0.8', '无参数']
        # if dataset == 'msd':
        #     arguments = ['无参数', 'itemdegree_lambda=0.7', 'itemdegree_x=-0.3', '无参数']
        # topn_files = []
        # for (method, argument) in list(zip(methods, arguments)):
        #     topn_file = r'{0}\{1}\{1}_out\topn_{2}-{3}-{4}'.format(current_directory, dataset, type, method, argument)
        #     topn_files.append(topn_file)
        # topn = 50
        # test_type = ['all']  # need change 可以加上'all'
        # begin_metric_mts2(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files)
        # print()

        dataset = 'msd'
        print(dataset, '各种HD评价指标的对比结果')
        methods = [
            'ProbS',
            'HeatS',
            'PD',
            'HPH',
            'ProbS Reranking DI TOPSIS'
        ]
        arguments = None
        if dataset == 'movielens1m':
            arguments = ['无参数', '无参数', 'itemdegree_x=-0.9', 'itemdegree_lambda=0.2', '无参数']
        if dataset == 'netflix':
            arguments = ['无参数', '无参数', 'itemdegree_x=-0.8', 'itemdegree_lambda=0.2', '无参数']
        if dataset == 'msd':
            arguments = ['无参数', '无参数', 'itemdegree_x=-0.3', 'itemdegree_lambda=0.7', '无参数']
        topn_files = []
        for (method, argument) in list(zip(methods, arguments)):
            topn_file = r'{0}\{1}\{1}_out\topn_{2}-{3}-{4}'.format(current_directory, dataset, type, method, argument)
            topn_files.append(topn_file)
        topn = 50
        test_type = ['all']  # need change 可以加上'all'
        begin_metric_hds(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files)
        print()

        # print(dataset, '：all metrics')
        # methods = [
        #     'User CF Userdegree'
        # ]
        # arguments = ['userdegree_lambda=3']
        # topn_files = []
        # for (method, argument) in list(zip(methods, arguments)):
        #     topn_file = r'{0}\{1}\{1}_out\topn_{2}-{3}-{4}'.format(current_directory, dataset, type, method, argument)
        #     topn_files.append(topn_file)
        # topn = 50
        # test_type = ['big', 'medium', 'small']  # need change 可以加上'all'
        # begin_metric_all(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files)
        # print()

        # # UCF和ICF算法产生的推荐列表中的平均物品流行度
        # methods = ['ProbS']
        # topn_file = r'{0}\{1}\{1}_out\topn_{2}-{3}-无参数'.format(current_directory, dataset, type, methods[0])
        # score_itemdegree(current_directory, dataset, type, topn_file)

        # dataset = 'netflix'
        # print('PD')
        # methods = ['PD']
        # arguments = ['itemdegree_x=-0.2', 'itemdegree_x=-0.4', 'itemdegree_x=-0.6', 'itemdegree_x=-0.8']
        # topn_files = begin_recommendation(current_directory, dataset, type, methods, arguments, need_cfsimilarity=True)
        # topn = 50
        # test_type = ['all']  # need change 可以加上'all'
        # begin_metric(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files)
        #
        # print('HPH')
        # methods = ['HPH']
        # arguments = ['itemdegree_lambda=0.2', 'itemdegree_lambda=0.4', 'itemdegree_lambda=0.6', 'itemdegree_lambda=0.8']
        # topn_files = begin_recommendation(current_directory, dataset, type, methods, arguments, need_cfsimilarity=True)
        # topn = 50
        # test_type = ['all']  # need change 可以加上'all'
        # begin_metric(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files)

        # dataset = 'netflix'
        # print(dataset, ': ProbS系列')
        # methods = ['ProbS Step1+3']
        # arguments = ['userdegree_lambda=0.002', 'userdegree_lambda=0.02', 'userdegree_lambda=0.03', 'userdegree_lambda=0.04', 'userdegree_lambda=0.05', 'userdegree_lambda=0.07', 'userdegree_lambda=0.09', 'userdegree_lambda=0.1', 'userdegree_lambda=0.11', 'userdegree_lambda=0.5']
        # topn_files = begin_recommendation(current_directory, dataset, type, methods, arguments)
        # topn = 50
        # test_type = ['big', 'medium', 'small']  # need change 可以加上'all'
        # begin_metric(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files)
        # # dataset = 'movielens1m'
        # # print(dataset, ': ProbS系列')
        # # methods = ['ProbS Step1+3']
        # # arguments = ['userdegree_lambda=0.01', 'userdegree_lambda=0.03', 'userdegree_lambda=0.05',
        # #              'userdegree_lambda=0.07', 'userdegree_lambda=0.09', 'userdegree_lambda=0.11']
        # # topn_files = begin_recommendation(current_directory, dataset, type, methods, arguments)
        # # topn = 50
        # # test_type = ['big', 'medium', 'small']  # need change 可以加上'all'
        # # begin_metric(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files)
        # # dataset = 'msd'
        # # print(dataset, ': ProbS系列')
        # # methods = ['ProbS Step1+3']
        # # arguments = ['userdegree_lambda=0.001', 'userdegree_lambda=0.003', 'userdegree_lambda=0.005',
        # #              'userdegree_lambda=0.007', 'userdegree_lambda=0.009', 'userdegree_lambda=0.012']
        # # topn_files = begin_recommendation(current_directory, dataset, type, methods, arguments)
        # # topn = 50
        # # test_type = ['big', 'medium', 'small']  # need change 可以加上'all'
        # # begin_metric(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files)

        # dataset = 'movielens1m'
        # print(dataset, ': 反向推荐系列')
        # methods = ['TS(ProbS) Weight']
        # arguments = ['weight=0', 'weight=0.2', 'weight=0.4', 'weight=0.6', 'weight=0.8', 'weight=1']
        # topn_files = begin_recommendation(current_directory, dataset, type, methods, arguments)
        # # 计算评价指标
        # topn = 50
        # test_type = ['big', 'medium', 'small']  # need change 可以加上'all'
        # begin_metric(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files)

        # dataset = 'movielens100k'
        # print(dataset, ': 反向推荐系列')
        # methods = [
        #     'UCF Reranking DI TOPSIS',
        #     'ProbS Reranking DI TOPSIS'
        # ]
        # arguments = None
        # if dataset == 'movielens1m':
        #     arguments = ['q=10', '无参数']
        # if dataset == 'msd':
        #     arguments = ['q=3', '无参数']
        # methods_and_arguments = list(zip(methods, arguments))
        # topn_files = begin_recommendation_fixargument(current_directory, dataset, type, methods_and_arguments)
        # # 计算评价指标
        # topn = 50
        # test_type = ['big', 'medium', 'small']  # need change 可以加上'all'
        # begin_metric(current_directory, dataset, type, methods, arguments, topn, test_type, topn_files)
