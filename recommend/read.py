"""
读取训练集。
"""

import pandas as pd
from collections import defaultdict


def read_data(train_path):
    """仅能读取隐式二值评分数据。
    """

    train = pd.read_csv(train_path, sep=',', engine='python')

    user_items = defaultdict(set)
    item_users = defaultdict(set)
    for row in train.values:
        user = int(row[0])
        item = int(row[1])

        user_items[user].add(item)
        item_users[item].add(user)

    return (user_items, item_users)
def read_test(test_file):
    """读入各种测试集test文件
    """

    # boundary condition
    testfile = open(test_file)
    for line in testfile:
        if line.strip() == 'null':
            return None
        else:
            break
    testfile.close()

    test = pd.read_csv(test_file, sep=',', engine='python')

    users_pair = defaultdict(set)
    for row in test.values:
        user = int(row[0])
        item = int(row[1])
        users_pair[user].add(item)

    return users_pair