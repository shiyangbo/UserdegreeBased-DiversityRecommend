"""
一、数据集预处理
用户度的分布和物品度的分布(主要看用户度的分布)。有PDF分布和CDF分布。
// 如果原始数据集没有经过预处理（没有预先清除很多稀疏的数据），那么它中所蕴含的**用户行为**应该服从长尾分布（从PDF图像观察）。
"""

from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import math

def get_userdegree(file):
    # 统计用户度。

    users_degree = defaultdict(lambda: 0)
    with open(file, 'r') as file_read:
        count = 0
        for line in file_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r统计数据集中的用户度分布：已处理{0}万行'.format(count / 10000))
                sys.stdout.flush()
            count += 1

            row = line.split(',')
            user = int(row[0])  # 报错说明数据集不规范
            users_degree[user] += 1
    print()
    return users_degree
def get_itemdegree(file):
    # 统计物品度。

    items_degree = defaultdict(lambda: 0)
    with open(file, 'r') as file_read:
        count = 0
        for line in file_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r统计数据集中的物品度：已处理{0}万行'.format(count / 10000))
                sys.stdout.flush()
            count += 1

            row = line.split(',')
            item = int(row[1])  # 报错说明数据集不规范
            items_degree[item] += 1
    print()
    return items_degree
    
    
def degree_pdf_and_cdf(keys_degree, dataset, degreetype, get_pdf, get_cdf, savecdf=False):
    # 给定全体用户，或者全体物品。计算概率分布PDF和累计分布CDF，并根据需要把图像保存到外存磁盘。

    degrees_keynum = defaultdict(lambda: 0)
    for (key, degree) in keys_degree.items():
        degrees_keynum[degree] += 1
    
    # key流行度的概率密度PDF
    if get_pdf:
        # # 频率直方图
        # degrees = list(keys_degree.values())
        #
        # #num_bins = 10  # 直方图的柱数，默认为10
        # #(MAX, MIN) = (max(degrees), min(degrees))
        # #bins = range(MIN, MAX, int((MAX - MIN) / num_bins))
        #
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        # ax = plt.subplot(111)
        # #(n, bins, patches) = ax.hist(degrees, bins, facecolor='r')
        # (n, bins, patches) = ax.hist(degrees, 7, facecolor='r')
        # ax.set_xlim([0, 9000])  # need change
        # ax.set_ylim([0, None])
        # ax.set_xlabel('{0}流行度'.format(degreetype))
        # ax.set_ylabel('频率(频数)')
        # plt.title('{0}数据集:{1}流行度的概率密度分布PDF（频率直方图表示）'.format(dataset, degreetype))
        # plt.savefig(r'D:\recommender_data\{0}数据集-{1}流行度的概率分布PDF图像.png'.format(
        #     dataset, degreetype)
        # )  # 结果表明用户流行度（物品流行度）比较符合“长尾分布”
        # plt. close('all')

        # 画图。注意到是离散的散点，因为横坐标degree不是均匀地布满[minDegree, maxDegree]区间
        print('计算用户(物品)度的概率密度分布PDF')
        X = []
        Y = []
        for (degree, keynum) in degrees_keynum.items():
            # 普通坐标轴
            X.append(degree)
            Y.append(keynum)

            # # 双对数坐标轴
            # v1 = np.log10(degree)
            # v2 = np.log10(keynum)
            # X.append(v1)
            # Y.append(v2)

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        ax = plt.subplot(111)
        ax.scatter(X, Y, s=90, facecolors='none', edgecolors='r', label='$degree-numbers$')
        ax.set_xlim([0, None])  # need change
        ax.set_ylim([0, None])
        ax.set_xlabel('{0}流行度'.format(degreetype))
        ax.set_ylabel('频率(频数)')
        plt.title('{0}数据集:{1}流行度的概率密度分布PDF（散点图表示）'.format(dataset, degreetype))
        plt.show()
        # plt.savefig(r'D:\recommender_data\{0}数据集-{1}流行度的概率分布PDF图像.png'.format(
        #     dataset, degreetype)
        # )  # 结果表明用户流行度（物品流行度）比较符合“长尾分布”
        # plt. close('all')
    
    # 统计key流行度的累计分布CDF，并根据需要保存图像到外存磁盘
    if get_cdf:
        X = list(degrees_keynum.keys())  # 注意到是离散的散点，因为横坐标degree不是均匀地布满[minDegree, maxDegree]区间
        Y = []
        total_keys = sum(degrees_keynum.values())
        count = 0
        for degree_in_x in X:
            # 进度条
            if count % 100 == 0:
                sys.stdout.write('\r计算用户(物品)度的累计分布CDF：{0}/{1}'.format(count, len(X)))
                sys.stdout.flush()
            count += 1
    
            hit_keys = 0
            for (degree, keynum) in degrees_keynum.items():  # 可以改进。把degrees_keynum.items()排序后借助二分查找来加速
                if degree <= degree_in_x:
                    hit_keys += keynum
            Y.append(1.0 * hit_keys / total_keys)
        print()
        
        # 画出key流行度的累计分布CDF图像
        if savecdf:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            ax = plt.subplot(111)
            ax.scatter(X, Y, s=5, facecolors='None', edgecolors='r', label='${0}degree-probability$'.format(degreetype))
            if degreetype == '用户' and dataset == 'MovieLens':
                ax.set_xlim([0, 800])
            elif degreetype == '用户' and dataset == 'BookCrossing':
                ax.set_xlim([0, 400])
            elif degreetype == '用户' and dataset == 'Netflix':
                ax.set_xlim([0, 1500])
            else:
                pass
            ax.set_ylim([0, 1])
            ax.set_xlabel('{0}流行度'.format(degreetype))
            ax.set_ylabel('概率')
            plt.title('{0}数据集:{1}流行度的累计分布CDF（散点图表示）'.format(dataset, degreetype))
            plt.savefig(r'D:\recommender_data\{0}数据集-{1}流行度的累计分布CDF图像.png'.format(
                dataset, degreetype)
            )
            plt. close('all')

        return (X, Y)
        
    return None


def degree_rating(pics, degreetype):
    # 近似画一下。最终图像还是需要重新写代码实现，画成连续的接近折线图。
    # pics可以是不同的datasets，也可以是不同的suffixes。

    methods = {'用户': get_userdegree, '物品': get_itemdegree}

    colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k'][:len(pics)]
    color_index = 0
    for (dataset, suffix) in pics:
        # 先借助CDF相关方法得到keys的分布
        keys_degree = methods[degreetype](r'D:\recommender_data\{0}\dataset{1}'.format(dataset, suffix))
        (X, Y) = degree_pdf_and_cdf(keys_degree, dataset, degreetype, get_pdf=False, get_cdf=True)
    
        degrees_keynum = defaultdict(lambda: 0)
        for (key, degree) in keys_degree.items():
            degrees_keynum[degree] += 1
        probs_ratingcumulate = {}
        degrees_prob = dict(zip(X, Y))  # 形如{100: 80%, 10: 20%, ...}
        degrees_ratingnum = {degree: degree * keynum for (degree, keynum) in degrees_keynum.items()}
        degrees_ratingnum_sorted = sorted(degrees_ratingnum.items(), key=lambda a: a[0], reverse=True)
        
        ratingcumulate = 0
        for (degree, ratingnum) in degrees_ratingnum_sorted:
            ratingcumulate += ratingnum
            prob = 1 - degrees_prob[degree]  # 前100prob%最流行的用户（物品）
            probs_ratingcumulate[prob] = ratingcumulate
            
        ratingtotal = sum(degrees_ratingnum.values())
        X = []
        Y = []
        for (prob, ratingcumulate) in probs_ratingcumulate.items():
            X.append(prob)
            Y.append(1.0 * ratingcumulate / ratingtotal)
        
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        ax = plt.subplot(111)
        ax.scatter(X, Y, s=1, facecolors='None', edgecolors=colors[color_index], label='${0}{1}$'.format(dataset, suffix))
        color_index += 1
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('前%最流行的{0}'.format(degreetype))
        ax.set_ylabel('占总评分%')
        ax.legend(loc='lower right')
        plt.title('{0}流行度和评分数的关系（散点图表示）'.format(degreetype))
        
    plt.savefig(r'D:\recommender_data\{0}-{1}流行度和评分数的关系图像.png'.format(
        str(pics), degreetype)
    )
    plt. close('all')
    return


if __name__ == '__main__':
    degreetype = '用户'
    # datasets = ['movielens100k', 'movielens100k']
    dataset = 'movielens100k'
    suffixes = ['-v1']

    # 画出PDF和CDF图像
    methods = {'用户': get_userdegree, '物品': get_itemdegree}
    for suffix in suffixes:
        file = r'D:\recommender_data\{0}\dataset{1}'.format(dataset, suffix)
        keys_degree = methods[degreetype](file)
        # dataset为dataset+suffix
        degree_pdf_and_cdf(keys_degree, dataset+suffix, degreetype, get_pdf=True, get_cdf=False, savecdf=False)
    
    # # 画出rating分布图像
    # print('\n画出rating分布图像')
    # pics = list(zip(datasets, suffixes))
    # degree_rating(pics, degreetype)
