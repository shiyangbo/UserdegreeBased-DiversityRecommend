"""
把数据集data划分成训练集和测试集。
"""

import random
import pandas as pd
import sys
import sys
sys.path.append(r'D:\recommender')
sys.path.append(r'F:\recommender')
import preprocess.degree_distribution as myutil2
from collections import defaultdict


def read_data(file):
    # 使用DataFrame数据结构，读入数据集。

    print('正在读入数据集...', end='')
    df = pd.read_csv(file, sep=',', engine='python', header=None)
    users_pair = defaultdict(set)
    items_pair = defaultdict(set)
    for row in df.values:
        user = int(row[0])
        item = int(row[1])
        users_pair[user].add(item)
        items_pair[item].add(user)
    print('\t数据集已读入完毕')
    return (users_pair, items_pair)
def read_data_onebyone(file):
    # 一行一行读入。适用于大规模数据集。
    # // 我预计有可能会因为内存不够的原因，而报错。

    users_pair = defaultdict(set)
    items_pair = defaultdict(set)
    with open(file, 'r') as file_read:
        count = 0
        for line in file_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r读入数据集：已处理{0}万行'.format(count/10000))
                sys.stdout.flush()
            count += 1
            
            row = line.strip().split(',')
            user = int(row[0])  # 报错说明数据集不规范
            item = int(row[1])
            users_pair[user].add(item)
            items_pair[item].add(user)
    print()
    return (dict(users_pair), dict(items_pair))
def get_userdegree(file):
    # 统计用户度。

    users_degree = defaultdict(lambda: 0)
    with open(file, 'r') as file_read:
        count = 0
        for line in file_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r统计数据集中的用户度：已处理{0}万行'.format(count / 10000))
                sys.stdout.flush()
            count += 1

            row = line.split(',')
            user = int(row[0])  # 报错说明数据集不规范
            users_degree[user] += 1
    print()
    return users_degree


def split(dataset_file, train_file, testall_file, seed=47):
    # 按近似9:1的比例划分出train和testall。
    # // 主要到虽然是随机，但是只要seed相同，结果就可以重现。

    data = []
    train = []
    testall = []
    users_train = set()
    users_test = set()

    df = pd.read_csv(dataset_file, sep=',', engine='python', header=None)
    for row in df.values:
        user = int(row[0])
        item = int(row[1])
        data.append((user, item))

    random.seed(seed)
    random.shuffle(data)
    for (user, item) in data:
        if random.randint(0, 8) == 8:  # 0 <= random <= 8 < 9
            testall.append((str(user), str(item)))
            users_test.add(str(user))
        else:
            train.append((str(user), str(item)))
            users_train.add(str(user))

    # 重要。边界条件“用户只出现在test而未出现在train”时，要把对应数据从test转移到train中
    users_new = users_test - users_train
    if len(users_new) != 0:
        needchange = [(user_string, item_string) for (user_string, item_string) in testall if user_string in users_new]
        testall = [(user_string, item_string) for (user_string, item_string) in testall if user_string not in users_new]
        train += needchange
        count_changelines = len(needchange)
        print('Warning! 出现边界条件"用户只出现在test而未出现在train"，总共影响了测试集test里面的{}行\n'.format(count_changelines))
    else:
        print('恭喜。数据集划分时未出现异常的边界条件')

    print('把训练集和测试集写出到外存...')
    with open(train_file, 'w') as f:
        for row in train:
            f.write(','.join(row))
            f.write('\n')
    with open(testall_file, 'w') as f:
        for row in testall:
            f.write(','.join(row))
            f.write('\n')
    return
def divide_users(users_pair_dataset, testall_file, testbig_file, testmedium_file, testsmall_file, userdegree):
    # 按照数据集data中用户度的分布，从测试集中选出属于原书数据集的前20%用户流行度的用户构成testbig，
    # 后40%用户流行度的用户构成testsmall，中间剩余的40%用户流行度的用户构成testmedium。

    testall = pd.read_csv(testall_file, sep=',', engine='python', header=None)

    #
    print('对训练集testall分类，并把结果写出到外存...')
    # P(用户流行度X >= userdegree_big)=20%，属于testbig
    # P(userdegree_small <= 用户流行度X < userdegree_big)=60%，属于testmedium
    # P(用户流行度X < userdegree_small)=20%，属于testsmall
    testbig = {}
    testmedium = {}
    testsmall = {}

    for row in testall.values:
        user = int(row[0])
        item = int(row[1])

        degree = len(users_pair_dataset[user])

        if degree >= userdegree[0]:
            testbig.setdefault(user, set())
            testbig[user].add(item)
        elif degree >= userdegree[1]:
            testmedium.setdefault(user, set())
            testmedium[user].add(item)
        else:
            testsmall.setdefault(user, set())
            testsmall[user].add(item)

    with open(testbig_file, 'w') as f:
        for (user, items) in testbig.items():
            for item in items:
                line = [str(user), str(item)]
                f.write(','.join(line))
                f.write('\n')

    with open(testmedium_file, 'w') as f:
        for (user, items) in testmedium.items():
            for item in items:
                line = [str(user), str(item)]
                f.write(','.join(line))
                f.write('\n')

    with open(testsmall_file, 'w') as f:
        for (user, items) in testsmall.items():
            for item in items:
                line = [str(user), str(item)]
                f.write(','.join(line))
                f.write('\n')

    print('testall测试集共{0}用户。其中：\n\tbig测试集用户共{1}\n\tmedium测试集用户共{2}\n\tsmall测试集用户共{3}'.format((len(testbig)+len(testmedium)+len(testsmall)), len(testbig), len(testmedium), len(testsmall)))
    return
def find_percent(X, Y, top):
    # 给定一个比例值top，返回对应的thrd值（用户流行度）。其中，|{u|Ku>=thrd}| / |U| = top。

    p = 1.0 - top

    # 二分查找定下界low_bound
    low = 0
    high = len(Y)
    while low < high:
        mid = low + int((high - low) / 2)
        if Y[mid] > p:
            high = mid
        elif Y[mid] == p:
            high = mid
        else:
            low = mid + 1

    thrd = X[low]
    return thrd

def get_seedusers(pickle_filepath, users_pair_train, userdegree, seed=47):
    # 使用分层抽样，生成种子用户。
    # 抽样比例定为10%。
    # 为反向推荐做准备。

    users_seed = set()

    # 按用户活跃度前20%，中60%，后40%划分成3层
    users_big = set()
    users_medium = set()
    users_small = set()
    for (user, items) in users_pair_train.items():
        degree = len(items)

        if degree >= userdegree[0]:
            users_big.add(user)
        elif degree >= userdegree[1]:
            users_medium.add(user)
        else:
            users_small.add(user)

    # 分层抽样
    random.seed(seed)
    sample_ratio = 0.01
    sample_big = random.sample(users_big, int(round(sample_ratio * len(users_big))))
    sample_medium = random.sample(users_medium, int(round(sample_ratio * len(users_medium))))
    sample_small = random.sample(users_small, int(round(sample_ratio * len(users_small))))
    users_seed = set(sample_big + sample_medium + sample_small)
    # 可视化输出
    print('层1共{0}个用户'.format(len(sample_big)))
    print('层2共{0}个用户'.format(len(sample_medium)))
    print('层3共{0}个用户'.format(len(sample_small)))

    import pickle
    f = open(pickle_filepath, 'wb')
    pickle.dump(users_seed, f)
    f.close()
    return users_seed


if __name__ == '__main__':
    current_directory = r'D:\recommender_data'
    dataset = 'netflix'  # need change
    types = ['-v1']  # need change

    for type in types:
        dataset_file = r'{0}\{1}\dataset{2}'.format(current_directory, dataset, type)
        train_file = r'{0}\{1}\train{2}'.format(current_directory, dataset, type)
        testall_file = r'{0}\{1}\{1}_out\testall{2}'.format(current_directory, dataset, type)
        testbig_file = r'{0}\{1}\{1}_out\testbig{2}'.format(current_directory, dataset, type)
        testmedium_file = r'{0}\{1}\{1}_out\testmedium{2}'.format(current_directory, dataset, type)
        testsmall_file = r'{0}\{1}\{1}_out\testsmall{2}'.format(current_directory, dataset, type)

        # split(dataset_file, train_file, testall_file)
        #
        # (users_pair, items_pair) = read_data(dataset_file)
        # users_degree = {user: len(items) for (user, items) in users_pair.items()}
        # (X, Y) = myutil2.degree_pdf_and_cdf(users_degree, dataset=None, degreetype=None, get_pdf=False, get_cdf=True, savecdf=False)
        # thrd_big = find_percent(X, Y, 0.10)  # Last.fm对应比值改为前20%， 即0.20
        # thrd_small = find_percent(X, Y, 0.80)  # Last.fm对应比值改为后80%，即0.20
        # print('\n大度用户>={0}\t小度用户<={1}\n'.format(thrd_big, thrd_small))
        # divide_users(users_pair, testall_file, testbig_file, testmedium_file, testsmall_file, (thrd_big,thrd_small))
        # print()

        # 生成种子用户
        (users_pair, items_pair) = read_data(train_file)
        users_degree = {user: len(items) for (user, items) in users_pair.items()}
        (X, Y) = myutil2.degree_pdf_and_cdf(users_degree, dataset=None, degreetype=None, get_pdf=False, get_cdf=True, savecdf=False)
        thrd_big = find_percent(X, Y, 0.20)
        thrd_small = find_percent(X, Y, 0.80)
        print('\n大度用户>={0}\t小度用户<={1}\n'.format(thrd_big, thrd_small))
        pickle_filepath = r'{0}\{1}\train-seedusers{2}'.format(current_directory, dataset, type)
        get_seedusers(pickle_filepath, users_pair, (thrd_big,thrd_small))
        print()
