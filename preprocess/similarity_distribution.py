"""
一、数据集预处理
不同用户度的相似度分布。
// 0到1.0区间手动分成100个bin, 所以对应有100个散点，散点图表示。
"""


import sys
sys.path.append(r'D:\recommender')
sys.path.append(r'F:\recommender')
import preprocess.util_split_traintest as myutil
import preprocess.degree_distribution as myutil2
import preprocess.util_split_traintest as myutil3
import recommend.similarity as sim
import numpy as np
import matplotlib.pyplot as plt

keys = [i for i in np.arange(0, 1.01, 0.01)]
def plot_distribution(specific_users, similarities_usercf, title):
    # 画出散点图。
    # // 按照0.11:[0.105, 0.114], 0.12:[0.115, 0.124], ...。分成大约100个小区间bin。对应也有大约100个散点。

    xy = {i: 0 for i in np.arange(0, 1.01, 0.01)}

    average_term = len(specific_users)
    for user in specific_users:
        for (neighbor, simvalue) in similarities_usercf[user].items():
            key = get_key(simvalue)
            xy[key] += 1
    X = []
    Y = []
    for (similarity, count) in xy.items():
        X.append(similarity)
        Y.append(count * 1.0 / average_term)

    # 画图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    ax = plt.subplot(111)
    ax.scatter(X, Y, s=50, facecolors='none', edgecolors='r', label='$degree userNumbers$')
    ax.set_xlabel('邻居用户的相似度')
    ax.set_ylabel('频数(频率值)')
    ax.set_xlim([0, 1])
    ax.set_ylim([1, None])  # 注意，我们不画频数为0的散点
    plt.title('{0}{1}数据集-{2}的相似度分布（散点图表示）'.format(dataset, type, title))
    plt.savefig(r'D:\recommender_data\{0}{1}实证分析“不同用户度的相似度分布的差异度（散点图表示）-{2}”'.format(dataset, type, title))
    plt.close('all')  # 结果表明有一定趋势显示“用户度越大，其购买记录中物品度的平均值越低一些”
    print('{0}图像已保存完毕'.format(title))
    return
def get_key(simvalue):
    global keys

    # 二分查找定下界low_bound
    low = 0
    high = len(keys)
    while low < high:
        mid = low + int((high - low) / 2)
        if keys[mid] > simvalue:
            high = mid
        elif keys[mid] == simvalue:
            high = mid
        else:
            low = mid + 1

    if simvalue == keys[low]:
        return keys[low]
    else:
        if simvalue < keys[low] - 0.005:
            return keys[low - 1]
        else:
            return keys[low]


if __name__ == '__main__':
    current_directory = r'D:\recommender_data'
    dataset = 'Netflix'
    type = '-v1'  # -v3, -v33, -v1

    # 不同用户度
    dataset_file = r'{0}\{1}\dataset{2}'.format(current_directory, dataset, type)
    train_file = r'{0}\{1}\train{2}'.format(current_directory, dataset, type)
    (users_pair, items_pair) = myutil.read_data(dataset_file)
    users_degree = {user: len(items) for (user, items) in users_pair.items()}
    (X, Y) = myutil2.degree_pdf_and_cdf(users_degree, dataset=None, degreetype=None, get_pdf=False, get_cdf=True, savecdf=False)
    thrd_big = myutil3.find_percent(X, Y, 0.20)
    thrd_small = myutil3.find_percent(X, Y, 0.80)
    users_big = {user for (user, degree) in users_degree.items() if degree >= thrd_big}
    users_small = {user for (user, degree) in users_degree.items() if degree <= thrd_small}

    #
    similarity_usercf_file = r'{0}\{1}\{1}_out\similarities_usercf{2}.pickle'.format(current_directory, dataset, type)
    similarities_usercf = sim.get_uusimilarity(similarity_usercf_file, users_pair, items_pair)

    print('\n画出相似度similarity分布图像')
    plot_distribution(users_big, similarities_usercf, title='big userdegree')
    plot_distribution(users_small, similarities_usercf, title='small userdegree')
