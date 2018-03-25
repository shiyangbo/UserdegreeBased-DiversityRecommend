"""
计算用户与用户之间的相似度；
或者，计算物品与物品之间的相似度。
// 如果外存磁盘里已经计算有，则直接读取即可，不用重复计算。（用到了序列化pickle技术。）
"""


import recommend.read as read
import pickle
import os
import sys
import numpy as np
from collections import defaultdict
import math


def get_iisimilarity_wholedataset(dataset_file, pickle_filepath):
    # 使用dataset计算物品与物品之间的相似度。
    # // 使用余弦相似度。

    if os.path.exists(pickle_filepath):  # boundary condition
        print('wholedataset物品相似度文件已存在，直接序列化读取它，而不用再重新计算了')
        f = open(pickle_filepath, 'rb')
        ii_similarities_wholedataset = pickle.load(f)
        f.close()
        return ii_similarities_wholedataset

    (users_pair, items_pair) = read.read_data(dataset_file)
    print('正在计算 whole dataset 物品相似度矩阵，为之后的多样性评价指标计算提供帮助...')
    ii_similarities_wholedataset = get_similarity(users_pair, items_pair, need_exchange=True)
    f = open(pickle_filepath, 'wb')
    pickle.dump(ii_similarities_wholedataset, f)
    f.close()
    print()
    return ii_similarities_wholedataset
def get_uusimilarity(pickle_filepath, users_pair, items_pair):
    #

    if os.path.exists(pickle_filepath):  # boundary condition
        print('用户相似度文件已存在，直接序列化读取它，而不用再重新计算了')
        f = open(pickle_filepath, 'rb')
        similarities = pickle.load(f)
        f.close()
        return similarities

    uu_similarities = get_similarity(users_pair, items_pair)

    f = open(pickle_filepath, 'wb')
    pickle.dump(uu_similarities, f)
    f.close()
    return uu_similarities
def get_iisimilarity(pickle_filepath, users_pair, items_pair):
    #

    if os.path.exists(pickle_filepath):  # boundary condition
        print('物品相似度文件已存在，直接序列化读取它，而不用再重新计算了')
        f = open(pickle_filepath, 'rb')
        similarities = pickle.load(f)
        f.close()
        return similarities

    ii_similarities = get_similarity(users_pair, items_pair, need_exchange=True)

    f = open(pickle_filepath, 'wb')
    pickle.dump(ii_similarities, f)
    f.close()
    return ii_similarities
class NeighborPair:
    # 自定义元组类型。

    def __init__(self, i, j):
        # t形如(u1, u2)、(i1, i2)等。

        self.o1 = i  # 报错说明元素i不是int类型
        self.o2 = j

    def __hash__(self):
        return self.o1 + self.o2

    def __eq__(self, other):
        if self.o1 == other.o1:
            if self.o2 == other.o2:  # (o1, o2) == (other.o1, other.o2)
                return True
            else:
                return False
        elif self.o1 == other.o2:
            if self.o2 == other.o1:  # (o1, o2) == (other.o2, other.o1)
                return True
            else:
                return False
        else:
            return False

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y。
        # True at the same time。

        return not(self == other)
def get_similarity(users_pair, items_pair, need_exchange=False):
    # 模板函数。

    user_items = None
    item_users = None
    if need_exchange:
        user_items = items_pair
        item_users = users_pair
    else:
        user_items = users_pair
        item_users = items_pair

    # 相似度计算
    # version 2.0。我自己写的相似度计算方法，而不是《推荐系统实践》上写的相似度计算方法
    similarities = defaultdict(dict)
    count = 0
    for (u1, items1) in user_items.items():
        # 进度条
        count += 1
        sys.stdout.write('\r相似度计算：{0} / {1}'.format(count, len(user_items)))
        sys.stdout.flush()

        for item in items1:
            for u2 in item_users[item]:
                if u1 == u2:
                    continue
                if u2 in similarities[u1].keys():
                    continue

                items2 = user_items[u2]

                val = len(items1.intersection(items2)) / math.sqrt(len(items1)*len(items2))
                similarities[u1][u2] = val
                similarities[u2][u1] = val  # 可以改进，更节省内存。不再使用嵌套字典，而是使用元组作为key值

    return similarities
def get_similarity2(users_pair, items_pair, need_exchange=False):
    # 模板函数。只保存相似度矩阵一半的元素，更节省内存。

    user_items = None
    item_users = None
    if need_exchange:
        user_items = items_pair
        item_users = users_pair
    else:
        user_items = users_pair
        item_users = items_pair

    # 相似度计算
    # version 2.0。我自己写的相似度计算方法，而不是《推荐系统实践》上写的相似度计算方法
    similarities = defaultdict(dict)
    count = 0
    for (u1, items1) in user_items.items():
        # 进度条
        count += 1
        sys.stdout.write('\r相似度计算：{0} / {1}'.format(count, len(user_items)))
        sys.stdout.flush()

        for item in items1:
            for u2 in item_users[item]:
                if u1 == u2:
                    continue

                p = NeighborPair(u1, u2)
                if p in similarities:
                    continue

                items2 = user_items[u2]
                val = len(items1.intersection(items2)) / math.sqrt(len(items1) * len(items2))
                similarities[p] = val  # 更节省内存。不再使用嵌套字典，而是使用元组作为key值

    return similarities


def get_similarity_probs(pickle_filepath, users_pair, items_pair, need_exchange=False):
    # 注意到ProbS算法的该相似度矩阵是一个稠密的矩阵，此计算过程是ProbS算法中最费时的一部分！
    # 定义similarities[u1][u2]表示“u2**到**u1的相似度”，注意到它是非对称的（和similarities[u2][u1]不相等）。

    if os.path.exists(pickle_filepath):  # boundary condition
        print('probs相似度文件已存在，直接序列化读取它，而不用再重新计算了')
        f = open(pickle_filepath, 'rb')
        similarities = pickle.load(f)
        f.close()
        return similarities

    user_items = None
    item_users = None
    if need_exchange:
        user_items = items_pair
        item_users = users_pair
    else:
        user_items = users_pair
        item_users = items_pair

    # 相似度计算
    similarities = {}
    count = 0
    for (u1, items1) in user_items.items():
        # 进度条
        count += 1
        sys.stdout.write('\rprobs相似度计算：{0} / {1}'.format(count, len(user_items)))
        sys.stdout.flush()

        similarities.setdefault(u1, {})
        for (u2, items2) in user_items.items():
            if u2 in similarities[u1].keys():  # boundary condition
                continue

            if u1 == u2:  # boundary condition
                sum_numerator = 0.0
                for item in items1:
                    sum_numerator += len(item_users[item])
                similarities[u1][u1] = sum_numerator * 1.0 / len(user_items[u1])
                continue

            common_items = items1 & items2  # 集合的交、并操作很费时，可以考虑修改优化该代码
            sum_numerator = 0.0
            for item in common_items:
                sum_numerator += 1.0 / len(item_users[item])

            similarities[u1][u2] = sum_numerator * 1.0 / len(user_items[u2])
            similarities.setdefault(u2, {})  # boundary condition
            similarities[u2][u1] = sum_numerator * 1.0 / len(user_items[u1])

    f = open(pickle_filepath, 'wb')
    pickle.dump(similarities, f)
    f.close()
    return similarities
def get_similarity_probsdouble(users_pair, items_pair):
    # 注意到此方法得到的相似度有可能大于1.

    similarities_double_file = r'D:\recommender_data\ml100k\ml100k_out\ii_similarities_probsdouble.2017_08_14-fake.dat'

    # 获取用户相似度矩阵
    f = open(r'D:\recommender_data\ml100k\ml100k_out\uu_similarities_probs.2017_08_14-fake.dat', 'rb')
    uu_similarities = pickle.load(f)
    f.close()

    # 进一步计算物品相似度矩阵
    ii_similarities_double = {}
    count = 0
    for item1 in items_pair.keys():
        # 进度条
        count += 1
        sys.stdout.write('\rprobs double相似度计算：{0} / {1}'.format(count, len(items_pair)))
        sys.stdout.flush()

        users1 = items_pair[item1]
        for item2 in items_pair.keys():
            users2 = items_pair[item2]
            sum = 0.0
            for user1 in users1:
                for user2 in users2:
                    sum += uu_similarities[user1][user2]
            ii_similarities_double.setdefault(item1, {})
            ii_similarities_double[item1][item2] = sum

    f = open(similarities_double_file, 'wb')
    pickle.dump(ii_similarities_double, f)
    f.close()
    return
