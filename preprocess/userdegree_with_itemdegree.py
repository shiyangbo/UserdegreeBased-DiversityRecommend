"""
一、数据集预处理
用户度与物品度的关系。
// 物品度指的是（具有某个用户度的所有用户）（所购买过的物品的平均度）
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import gc


def userdegree_with_itemdegree(dataset, suffix):
    # 画出用户度和物品平均度的关系图像，保存到外存磁盘。

    data_file = r'D:\recommender_data\{0}\dataset{1}'.format(dataset, suffix)

    users_pair = defaultdict(set)
    users_degree = defaultdict(lambda: 0)
    items_degree = defaultdict(lambda: 0)
    count = 0
    with open(data_file, 'r') as file_read:
        for line in file_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r读入给定的数据集：已处理{0}万行'.format(count / 10000))
                sys.stdout.flush()
            count += 1

            row = line.strip().split(',')
            user = int(row[0])  # 报错说明数据集不规范
            item = int(row[1])
            users_pair[user].add(item)
            users_degree[user] += 1
            items_degree[item] += 1
    print('\r---------------')
    print("统计信息：\t用户数：{0}\t物品数：{1}".format(
        len(users_degree), len(items_degree))
    )

    # 统计，并画图
    users_degree_sorted = sorted(users_degree.items(), key=lambda a: a[1])
    X = [degree for (user, degree) in users_degree_sorted]
    Y = []

    degrees_user = dict_transpose(users_degree)
    count = 0
    # for degree in X:  # 方法1。然后得到对应的纵坐标的值。注意到**平均化了两次**才得到单值
    #     # 进度条
    #     if count % 10 == 0:
    #         sys.stdout.write('\r计算纵坐标：{}%'.format(
    #             round(100*count/len(X), 1))
    #         )
    #         sys.stdout.flush()
    #     count += 1
    #
    #     res = [[items_degree[item]] for user in degrees_user[degree] for item in users_pair[user]]
    #     mean1 = [np.mean(degrees) for degrees in res]
    #     mean2 = np.mean(mean1)
    #     Y.append(mean2)
    # print('\r---------------')
    for degree in X:  # 方法2。**统一平均化一次**，等效于方法1
        # 进度条
        if count % 100 == 0:
            sys.stdout.write('\r计算纵坐标：{}%'.format(
                round(100 * count / len(X), 1))
            )
            sys.stdout.flush()
        count += 1

        # # 方法1
        # res = [items_degree[item] for user in degrees_user[degree] for item in users_pair[user]]
        # mean = np.mean(res)
        # Y.append(mean)

        # 方法2
        sumvalue = 0
        users = degrees_user[degree]
        for user in users:  # 该用户度下的所属用户
            for item in users_pair[user]:  # 上述用户所交互的物品
                sumvalue += items_degree[item]  # 计算上述这些物品的平均度

        Y.append(sumvalue * 1.0 / degree / len(users))
    print('\r---------------')

    # 画图
    del users_pair
    del users_degree
    del items_degree
    gc.collect()
    print('完成垃圾回收，最后保存图像到外存磁盘...')

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    ax = plt.subplot(111)
    ax.scatter(X, Y, s=90, facecolors='none', edgecolors='k')
    ax.set_xlabel('用户活跃度')
    ax.set_ylabel('物品平均流行度')
    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    plt.title('Netflix数据集'.format(dataset, suffix))
    plt.show()
    # plt.savefig(r'D:\recommender_data\{0}{1}数据集-用户度和物品平均度的关系图像'.format(dataset, suffix))
    # plt.close('all')  # 结果表明有一定趋势显示“用户度越大，其购买记录中物品度的平均值越低一些”
    # print('图像已保存完毕')


def dict_transpose(d):
    # 对普通字典做翻转。

    d_new = defaultdict(set)
    for (key, value) in d.items():
        d_new[value].add(key)
    return d_new


if __name__ == '__main__':
    datasets = ['netflix']
    suffixes = ['-v1']
    for (dataset, suffix) in zip(datasets, suffixes):
        userdegree_with_itemdegree(dataset, suffix)
