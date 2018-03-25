"""
画图程序。
"""


import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def line_double(x, y1, y2, size=None, save=False):
    # 画双坐标轴折线图（曲线图）。


    fig, ax1 = plt.subplots()  # 创建图像文件。最好写成ax = plt.subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(x, y1, 'k.-', mfc='none', label='Precision')
    ax2.plot(x, y2, 'k.--', mfc='none', label='HD')

    ax1.set_xlim([0, 14])
    ax1.set_ylim([0, 0.3])
    ax2.set_ylim([0, 1])

    ax1.set_xlabel(r'$q$')
    ax1.set_ylabel('准确率指标 Preicison')
    ax2.set_ylabel('多样性指标 HD')

    #
    plt.grid(True)
    plt.title('ML1M数据集，High部分')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()  # 显示图像
    return


def line(x, y1, y2):
    # 画普通折线图。

    font_size = 18

    ax = plt.subplot(111)
    ax.plot(x, y1, 'ko--', mfc='None', linewidth=2.0, label='UCF')
    ax.plot(x, y2, 'k^-', mfc='None', linewidth=2.0, label='UCFu')

    ax.set_xlim([0, 5.5])
    ax.set_ylim([0, 0.6])
    ax.set_xlabel(r'$\lambda$', fontsize=font_size)
    ax.set_ylabel(r'$\rm {0}$'.format('nDCG'), fontsize=font_size)  # \rm表示正体

    # plt.grid(True)
    plt.title('exam', fontsize=font_size)  # change
    ax.legend(loc='upper left', fontsize=font_size)

    plt.tight_layout()  # 使布局更紧凑
    plt.show()
    return


def bar_double(index, y1, y2, size=None, save=False):
    # 画条形图。index有三个分量，分别对应[ML1M, Netflix, MSD]。

    ax = plt.subplot(111)

    bar_width = 0.35
    rects1 = plt.bar(index, y1, bar_width, color='r', fill=False, hatch='/', label='UCF')
    index2 = [i + bar_width for i in index]
    rects2 = plt.bar(index2, y2, bar_width, color='r', fill=False, hatch='o', label='ICF')

    # 给图加text
    for x, y in zip(index, y1):
        plt.text(x + 0.2, y + 0.01, '%d' % y, ha='center', va='bottom')
    for x, y in zip(index2, y2):
        plt.text(x + 0.2, y + 0.01, '%d' % y, ha='center', va='bottom')

    ax.set_xlabel(r'测试集')
    ax.set_ylabel(r'平均物品流行度')  # change
    ax.set_xlim([0, None])
    ax.set_ylim([0, None])

    #
    index3 = [i + (bar_width / 2) + 0.1 for i in index]
    plt.xticks(index3, ['High部分', 'Medium部分', 'Low部分'])  # 注意不能使用ax.set_xticks()
    #plt.title('High部分')  # change
    ax.legend(loc='upper left')

    plt.tight_layout()  # 使布局更紧凑
    #plt.show()  # 显示图像
    plt.savefig(r'C:\Users\shiya\Desktop\pic.png', dpi=220)
    return


def bar(index, y1, y2, y3, size=None, save=False):
    # 画条形图。index有三个分量，分别对应[ML1M, Netflix, MSD]。

    ax = plt.subplot(111)

    bar_width = 0.35
    index2 = [i + bar_width for i in index]
    index3 = [i + 2 * bar_width for i in index]
    rects1 = plt.bar(index, y1, bar_width, color='white', label='测试集High部分')
    rects2 = plt.bar(index2, y2, bar_width, color='dimgray', label='测试集Medium部分')
    rects3 = plt.bar(index3, y3, bar_width, color='black', label='测试集Low部分')

    # # 给图加text
    # plt.text(index2[0] + 0.2, y2[0] + 0.01, '39%', ha='center', va='bottom')
    # plt.text(index2[1] + 0.2, y2[1] + 0.01, '1%', ha='center', va='bottom')
    # plt.text(index2[2] + 0.2, y2[2] + 0.01, '-3%', ha='center', va='bottom')
    #
    # plt.text(index3[0] + 0.2, y3[0] + 0.01, '70%', ha='center', va='bottom')
    # plt.text(index3[1] + 0.2, y3[1] + 0.01, '2%', ha='center', va='bottom')
    # plt.text(index3[2] + 0.2, y3[2] + 0.01, '3%', ha='center', va='bottom')

    ax.set_xlabel(r'各评价指标')  # change
    # ax.set_ylabel(r'值')  # change
    ax.set_xlim([0, None])
    ax.set_ylim([0, 1])

    #
    index3 = [i + (bar_width / 2) for i in index]
    plt.xticks(index3, ['nDCG', 'ILD', 'Novelty'])  # 注意不能使用ax.set_xticks()
    plt.title('MSD数据集')  # change
    ax.legend(loc='upper left')

    plt.tight_layout()  # 使布局更紧凑
    plt.show()  # 显示图像
    # plt.savefig(r'C:\Users\shiya\Desktop\pic.png', dpi=220)
    return


if __name__ == '__main__':
    # # UCF参数q遍历
    # # 高活跃度用户
    # x = [i for i in range(1, 13 + 1)]
    # y1 = [0.168,0.180,0.189,0.196,0.202,0.208,0.212,0.216,0.219,0.221,0.223,0.225,0.226 ]
    # y2 = [0.628,0.657,0.680,0.700,0.717,0.732,0.746,0.759,0.770,0.781,0.790,0.799,0.807]
    # fig, ax1 = plt.subplots()  # 创建图像文件。最好写成ax = plt.subplot(111)
    # font_size = 25
    # ax2 = ax1.twinx()
    # ax1.plot(x, y1, 'ks-', mfc='k', linewidth=2.0, markersize=9, label='Precision')
    # ax2.plot(x, y2, 'kv--', mfc='none', linewidth=2.0, markersize=9, label='HD')
    # ax1.set_xlim([0, 14])
    # ax1.set_ylim([0, 0.3])
    # ax2.set_ylim([0, 1])
    # ax1.set_xlabel(r'$q$', fontsize=font_size)
    # ax1.set_ylabel(r'$\rm Preicison$', fontsize=font_size)
    # ax2.set_ylabel(r'$\rm HD$', fontsize=font_size)
    # # plt.grid(True)
    # plt.title('高活跃度用户群', fontsize=font_size)
    # ax1.legend(loc='upper left', fontsize=font_size)
    # ax2.legend(loc='upper right', fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像
    #
    # # 低活跃度用户
    # x = [i for i in range(1, 13 + 1)]
    # y1 = [0.017,0.021,0.024,0.027,0.029,0.030,0.030,0.029,0.029,0.028,0.027,0.027,0.026]
    # y2 = [0.246,0.346,0.449,0.559,0.655,0.728,0.779,0.815,0.839,0.856,0.869,0.877,0.884 ]
    # fig, ax1 = plt.subplots()  # 创建图像文件。最好写成ax = plt.subplot(111)
    # font_size = 25
    # ax2 = ax1.twinx()
    # ax1.plot(x, y1, 'ks-', mfc='k', linewidth=2.0, markersize=9, label='Precision')
    # ax2.plot(x, y2, 'kv--', mfc='none', linewidth=2.0, markersize=9, label='HD')
    # ax1.set_xlim([0, 14])
    # ax1.set_ylim([0, 0.1])
    # ax2.set_ylim([0, 1])
    # ax1.set_xlabel(r'$q$', fontsize=font_size)
    # ax1.set_ylabel(r'$\rm Preicison$', fontsize=font_size)
    # ax2.set_ylabel(r'$\rm HD$', fontsize=font_size)
    # # plt.grid(True)
    # plt.title('低活跃度用户群', fontsize=font_size)
    # ax1.legend(loc='upper left', fontsize=font_size)
    # ax2.legend(loc='lower right', fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像

    # # biasP3参数lambda遍历
    # # 高活跃度用户
    # x = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11]
    # y1 = [0.634,0.636,0.631,0.622,0.606,0.586]
    # y2 = [0.652,0.664,0.679,0.699,0.723,0.749]
    # fig, ax = plt.subplots()  # 创建图像文件。最好写成ax = plt.subplot(111)
    # font_size = 25
    # ax.plot(x, y1, 'ks-', mfc='k', linewidth=2.0, markersize=9, label='nDCG')
    # ax.plot(x, y2, 'kv--', mfc='none', linewidth=2.0, markersize=9, label='HD')
    # ax.set_xlim([0, None])
    # ax.set_ylim([0.4, 1])
    # ax.set_xlabel(r'$\lambda$', fontsize=font_size)
    # ax.set_ylabel(r'指标值', fontsize=font_size)
    # # plt.grid(True)
    # plt.title('高活跃度用户群', fontsize=font_size)
    # ax.legend(loc='upper left', fontsize=font_size)
    # ax.set_xticks([0.01, 0.03, 0.05, 0.07, 0.09, 0.11])
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像
    #
    # # 低活跃度用户
    # x = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11]
    # y1 = [0.262,0.262,0.263,0.264,0.265,0.265]
    # y2 = [0.325,0.328,0.331,0.334,0.338,0.343]
    # fig, ax = plt.subplots()  # 创建图像文件。最好写成ax = plt.subplot(111)
    # font_size = 25
    # ax.plot(x, y1, 'ks-', mfc='k', linewidth=2.0, markersize=9, label='nDCG')
    # ax.plot(x, y2, 'kv--', mfc='none', linewidth=2.0, markersize=9, label='HD')
    # ax.set_xlim([0, None])
    # ax.set_ylim([0.2, 0.4])
    # ax.set_xlabel(r'$\lambda$', fontsize=font_size)
    # ax.set_ylabel(r'指标值', fontsize=font_size)
    # # plt.grid(True)
    # plt.title('低活跃度用户群', fontsize=font_size)
    # ax.legend(loc='upper left', fontsize=font_size)
    # ax.set_xticks([0.01, 0.03, 0.05, 0.07, 0.09, 0.11])
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像

    # # 高活跃度用户
    # # ML1M
    # x = [i for i in range(1, 5 + 1)]
    # y1 = [0.781, 0.781, 0.781, 0.781, 0.781]
    # y2 = [0.727, 0.794, 0.837, 0.865, 0.883]
    # ax = plt.subplot(111)
    # font_size = 25
    # ax.plot(x, y1, 'kv--', mfc='None', linewidth=2.0, markersize=9, label='UCFQ')
    # ax.plot(x, y2, 'ks-', mfc='k', linewidth=2.0, markersize=9, label='uUCFQ')
    # ax.set_xlim([0, 5.5])
    # ax.set_ylim([0.6, 1])
    # ax.set_xlabel(r'$\lambda$', fontsize=font_size)
    # ax.set_ylabel(r'$\rm HD$', fontsize=font_size)  # \rm表示正体
    # # plt.grid(True)
    # plt.title('高活跃度用户群', fontsize=font_size)  # change
    # ax.legend(loc='lower right', fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()
    #
    # # Netflix
    # x = [i for i in range(1, 5 + 1)]
    # y1 = [0.795, 0.795, 0.795, 0.795, 0.795]
    # y2 = [0.749, 0.803, 0.839, 0.865, 0.883]
    # ax = plt.subplot(111)
    # font_size = 25
    # ax.plot(x, y1, 'kv--', mfc='None', linewidth=2.0, markersize=9, label='UCFQ')
    # ax.plot(x, y2, 'ks-', mfc='k', linewidth=2.0, markersize=9, label='uUCFQ')
    # ax.set_xlim([0, 5.5])
    # ax.set_ylim([0.6, 1])
    # ax.set_xlabel(r'$\lambda$', fontsize=font_size)
    # ax.set_ylabel(r'$\rm HD$', fontsize=font_size)  # \rm表示正体
    # # plt.grid(True)
    # plt.title('高活跃度用户群', fontsize=font_size)  # change
    # ax.legend(loc='lower right', fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()
    #
    # # MSD
    # x = [i for i in range(1, 5 + 1)]
    # y1 = [0.831, 0.831, 0.831, 0.831, 0.831]
    # y2 = [0.921, 0.967, 0.974, 0.976, 0.977]
    # ax = plt.subplot(111)
    # font_size = 25
    # ax.plot(x, y1, 'kv--', mfc='None', linewidth=2.0, markersize=9, label='UCFQ')
    # ax.plot(x, y2, 'ks-', mfc='k', linewidth=2.0, markersize=9, label='uUCFQ')
    # ax.set_xlim([0, 5.5])
    # ax.set_ylim([0.6, 1])
    # ax.set_xlabel(r'$\lambda$', fontsize=font_size)
    # ax.set_ylabel(r'$\rm HD$', fontsize=font_size)  # \rm表示正体
    # # plt.grid(True)
    # plt.title('高活跃度用户群', fontsize=font_size)  # change
    # ax.legend(loc='lower right', fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()

    # # 低活跃度用户
    # # ML1M
    # x = [i for i in range(1, 5 + 1)]
    # y1 = [0.354, 0.354, 0.354, 0.354, 0.354]
    # y2 = [0.311,0.369,0.354,0.331,0.318]
    # ax = plt.subplot(111)
    # font_size = 25
    # ax.plot(x, y1, 'kv--', mfc='None', linewidth=2.0, markersize=9, label='UCFQ')
    # ax.plot(x, y2, 'ks-', mfc='k', linewidth=2.0, markersize=9, label='uUCFQ')
    # ax.set_xlim([0, 5.5])
    # ax.set_ylim([0, 0.6])
    # ax.set_xlabel(r'$\lambda$', fontsize=font_size)
    # ax.set_ylabel(r'$\rm nDCG$', fontsize=font_size)  # \rm表示正体
    # # plt.grid(True)
    # plt.title('低活跃度用户群', fontsize=font_size)  # change
    # ax.legend(loc='upper left', fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()
    # # Netflix
    # x = [i for i in range(1, 5 + 1)]
    # y1 = [0.248, 0.248, 0.248, 0.248, 0.248]
    # y2 = [0.249,0.273,0.262,0.245,0.233]
    # ax = plt.subplot(111)
    # font_size = 25
    # ax.plot(x, y1, 'kv--', mfc='None', linewidth=2.0, markersize=9, label='UCFQ')
    # ax.plot(x, y2, 'ks-', mfc='k', linewidth=2.0, markersize=9, label='uUCFQ')
    # ax.set_xlim([0, 5.5])
    # ax.set_ylim([0, 0.6])
    # ax.set_xlabel(r'$\lambda$', fontsize=font_size)
    # ax.set_ylabel(r'$\rm nDCG$', fontsize=font_size)  # \rm表示正体
    # # plt.grid(True)
    # plt.title('低活跃度用户群', fontsize=font_size)  # change
    # ax.legend(loc='upper left', fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()
    # # MSD
    # x = [i for i in range(1, 5 + 1)]
    # y1 = [0.151, 0.151, 0.151, 0.151, 0.151]
    # y2 = [0.152,0.142,0.129,0.122,0.118]
    # ax = plt.subplot(111)
    # font_size = 25
    # ax.plot(x, y1, 'kv--', mfc='None', linewidth=2.0, markersize=9, label='UCFQ')
    # ax.plot(x, y2, 'ks-', mfc='k', linewidth=2.0, markersize=9, label='uUCFQ')
    # ax.set_xlim([0, 5.5])
    # ax.set_ylim([0, 0.6])
    # ax.set_xlabel(r'$\lambda$', fontsize=font_size)
    # ax.set_ylabel(r'$\rm nDCG$', fontsize=font_size)  # \rm表示正体
    # # plt.grid(True)
    # plt.title('低活跃度用户群', fontsize=font_size)  # change
    # ax.legend(loc='upper left', fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()

    # # ML1M和MSD数据集上ProbSb算法分析
    # # 画条形图。index有两个分量，分别对应[HD, nDCG]。
    # # ML1M
    # y1 = [0.648, 0.633]
    # y2 = [0.653, 0.636]
    # y3 = [0.65, 0.633]
    # y4 = [0.662, 0.639]
    # index = [0.5, 3.5]
    # ax = plt.subplot(111)
    # font_size = 25
    # bar_width = 0.35
    # index2 = [i + bar_width for i in index]
    # index3 = [i + 2 * bar_width for i in index]
    # index4 = [i + 3 * bar_width for i in index]
    # rects1 = plt.bar(index, y1, bar_width, color='white', label='ProbS')
    # rects2 = plt.bar(index2, y2, bar_width, color='silver', label='Step1')
    # rects3 = plt.bar(index3, y3, bar_width, color='dimgray', label='Step1+2')
    # rects4 = plt.bar(index4, y4, bar_width, color='black', label='Step1+2+3')
    # # # 给图加text
    # # plt.text(index2[0] + 0.2, y2[0] + 0.01, '39%', ha='center', va='bottom')
    # # plt.text(index2[1] + 0.2, y2[1] + 0.01, '1%', ha='center', va='bottom')
    # # plt.text(index2[2] + 0.2, y2[2] + 0.01, '-3%', ha='center', va='bottom')
    # #
    # # plt.text(index3[0] + 0.2, y3[0] + 0.01, '70%', ha='center', va='bottom')
    # # plt.text(index3[1] + 0.2, y3[1] + 0.01, '2%', ha='center', va='bottom')
    # # plt.text(index3[2] + 0.2, y3[2] + 0.01, '3%', ha='center', va='bottom')
    # ax.set_xlabel(r'多样性指标和准确率指标', fontsize=font_size)  # change
    # # ax.set_ylabel(r'值')  # change
    # ax.set_xlim([0, None])
    # ax.set_ylim([0, 1])
    # index5 = [i + (bar_width / 2) + 0.4 for i in index]
    # plt.xticks(index5, [r'$\rm HD$', r'$\rm nDCG$'], fontsize=font_size)  # 注意不能使用ax.set_xticks()
    # plt.title('ML1M数据集，高活跃度用户', fontsize=font_size)  # change
    # ax.legend(loc='upper right')
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像
    #
    # # MSD
    # y1 = [0.719, 0.31]
    # y2 = [0.817, 0.302]
    # y3 = [0.846, 0.291]
    # y4 = [0.981, 0.236]
    # index = [0.5, 3.5]
    # ax = plt.subplot(111)
    # font_size = 25
    # bar_width = 0.35
    # index2 = [i + bar_width for i in index]
    # index3 = [i + 2 * bar_width for i in index]
    # index4 = [i + 3 * bar_width for i in index]
    # rects1 = plt.bar(index, y1, bar_width, color='white', label='ProbS')
    # rects2 = plt.bar(index2, y2, bar_width, color='silver', label='Step1')
    # rects3 = plt.bar(index3, y3, bar_width, color='dimgray', label='Step1+2')
    # rects4 = plt.bar(index4, y4, bar_width, color='black', label='Step1+2+3')
    # # # 给图加text
    # # plt.text(index2[0] + 0.2, y2[0] + 0.01, '39%', ha='center', va='bottom')
    # # plt.text(index2[1] + 0.2, y2[1] + 0.01, '1%', ha='center', va='bottom')
    # # plt.text(index2[2] + 0.2, y2[2] + 0.01, '-3%', ha='center', va='bottom')
    # #
    # # plt.text(index3[0] + 0.2, y3[0] + 0.01, '70%', ha='center', va='bottom')
    # # plt.text(index3[1] + 0.2, y3[1] + 0.01, '2%', ha='center', va='bottom')
    # # plt.text(index3[2] + 0.2, y3[2] + 0.01, '3%', ha='center', va='bottom')
    # ax.set_xlabel(r'多样性指标和准确率指标', fontsize=font_size)  # change
    # # ax.set_ylabel(r'值')  # change
    # ax.set_xlim([0, None])
    # ax.set_ylim([0, 1])
    # index5 = [i + (bar_width / 2) + 0.4 for i in index]
    # plt.xticks(index5, [r'$\rm HD$', r'$\rm nDCG$'], fontsize=font_size)  # 注意不能使用ax.set_xticks()
    # plt.title('MSD数据集，高活跃度用户', fontsize=font_size)  # change
    # ax.legend(loc='upper right', fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像

    # # nDCG和Precision指标在不同活跃度用户上值的变动情况
    # # 画条形图。index有两个分量，分别对应[nDCG, Precision]
    # # ML1M
    # y1 = [0.633, 0.177]
    # y2 = [0.421, 0.058]
    # y3 = [0.262, 0.021]
    # index = [0.5, 3.5]
    # ax = plt.subplot(111)
    # font_size = 25
    # bar_width = 0.35
    # index2 = [i + bar_width for i in index]
    # index3 = [i + 2 * bar_width for i in index]
    # index4 = [i + 3 * bar_width for i in index]
    # rects1 = plt.bar(index, y1, bar_width, color='white', label='高活跃度用户群')
    # rects2 = plt.bar(index2, y2, bar_width, color='silver', label='普通活跃度用户群')
    # rects3 = plt.bar(index3, y3, bar_width, color='dimgray', label='低活跃度用户群')
    # # # 给图加text
    # # plt.text(index2[0] + 0.2, y2[0] + 0.01, '39%', ha='center', va='bottom')
    # # plt.text(index2[1] + 0.2, y2[1] + 0.01, '1%', ha='center', va='bottom')
    # # plt.text(index2[2] + 0.2, y2[2] + 0.01, '-3%', ha='center', va='bottom')
    # #
    # # plt.text(index3[0] + 0.2, y3[0] + 0.01, '70%', ha='center', va='bottom')
    # # plt.text(index3[1] + 0.2, y3[1] + 0.01, '2%', ha='center', va='bottom')
    # # plt.text(index3[2] + 0.2, y3[2] + 0.01, '3%', ha='center', va='bottom')
    # ax.set_xlabel(r'评价指标', fontsize=font_size)  # change
    # # ax.set_ylabel(r'值')  # change
    # ax.set_xlim([0, None])
    # ax.set_ylim([0, 1])
    # index5 = [i + (bar_width / 2) + 0.2 for i in index]
    # plt.xticks(index5, [r'$\rm nDCG$', r'$\rm Precision$'], fontsize=font_size)  # 注意不能使用ax.set_xticks()
    # plt.title('ML1M数据集', fontsize=font_size)  # change
    # ax.legend(loc='upper right', fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像
    #
    # # # Netflix
    # # y1 = [0.628, 0.197]
    # # y2 = [0.37, 0.054]
    # # y3 = [0.22, 0.018]
    # # index = [0.5, 3.5]
    # # ax = plt.subplot(111)
    # # font_size = 25
    # # bar_width = 0.35
    # # index2 = [i + bar_width for i in index]
    # # index3 = [i + 2 * bar_width for i in index]
    # # index4 = [i + 3 * bar_width for i in index]
    # # rects1 = plt.bar(index, y1, bar_width, color='white', label='高活跃度用户群')
    # # rects2 = plt.bar(index2, y2, bar_width, color='silver', label='普通活跃度用户群')
    # # rects3 = plt.bar(index3, y3, bar_width, color='dimgray', label='低活跃度用户群')
    # # # # 给图加text
    # # # plt.text(index2[0] + 0.2, y2[0] + 0.01, '39%', ha='center', va='bottom')
    # # # plt.text(index2[1] + 0.2, y2[1] + 0.01, '1%', ha='center', va='bottom')
    # # # plt.text(index2[2] + 0.2, y2[2] + 0.01, '-3%', ha='center', va='bottom')
    # # #
    # # # plt.text(index3[0] + 0.2, y3[0] + 0.01, '70%', ha='center', va='bottom')
    # # # plt.text(index3[1] + 0.2, y3[1] + 0.01, '2%', ha='center', va='bottom')
    # # # plt.text(index3[2] + 0.2, y3[2] + 0.01, '3%', ha='center', va='bottom')
    # # ax.set_xlabel(r'评价指标', fontsize=font_size)  # change
    # # # ax.set_ylabel(r'值')  # change
    # # ax.set_xlim([0, None])
    # # ax.set_ylim([0, 1])
    # # index5 = [i + (bar_width / 2) + 0.2 for i in index]
    # # plt.xticks(index5, [r'$\rm nDCG$', r'$\rm Precision$'], fontsize=font_size)  # 注意不能使用ax.set_xticks()
    # # plt.title('Netflix数据集', fontsize=font_size)  # change
    # # ax.legend(loc='upper right', fontsize=font_size)
    # # plt.tight_layout()  # 使布局更紧凑
    # # plt.show()  # 显示图像
    #
    # # MSD
    # y1 = [0.31,0.04]
    # y2 = [0.189,0.015]
    # y3 = [0.156,0.009]
    # index = [0.5, 3.5]
    # ax = plt.subplot(111)
    # font_size = 25
    # bar_width = 0.35
    # index2 = [i + bar_width for i in index]
    # index3 = [i + 2 * bar_width for i in index]
    # index4 = [i + 3 * bar_width for i in index]
    # rects1 = plt.bar(index, y1, bar_width, color='white', label='高活跃度用户群')
    # rects2 = plt.bar(index2, y2, bar_width, color='silver', label='普通活跃度用户群')
    # rects3 = plt.bar(index3, y3, bar_width, color='dimgray', label='低活跃度用户群')
    # # # 给图加text
    # # plt.text(index2[0] + 0.2, y2[0] + 0.01, '39%', ha='center', va='bottom')
    # # plt.text(index2[1] + 0.2, y2[1] + 0.01, '1%', ha='center', va='bottom')
    # # plt.text(index2[2] + 0.2, y2[2] + 0.01, '-3%', ha='center', va='bottom')
    # #
    # # plt.text(index3[0] + 0.2, y3[0] + 0.01, '70%', ha='center', va='bottom')
    # # plt.text(index3[1] + 0.2, y3[1] + 0.01, '2%', ha='center', va='bottom')
    # # plt.text(index3[2] + 0.2, y3[2] + 0.01, '3%', ha='center', va='bottom')
    # ax.set_xlabel(r'评价指标', fontsize=font_size)  # change
    # # ax.set_ylabel(r'值')  # change
    # ax.set_xlim([0, None])
    # ax.set_ylim([0, 1])
    # index5 = [i + (bar_width / 2) + 0.2 for i in index]
    # plt.xticks(index5, [r'$\rm nDCG$', r'$\rm Precision$'], fontsize=font_size)  # 注意不能使用ax.set_xticks()
    # plt.title('MSD数据集', fontsize=font_size)  # change
    # ax.legend(loc='upper right', fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像

    # # ILD和NoveltyNew指标在不同活跃度用户上值的变动情况
    # # 画条形图。index有两个分量，分别对应[nDCG, Precision]
    # # ML1M
    # y1 = [0.597,0.095]
    # y2 = [0.552,0.075]
    # y3 = [0.538,0.068]
    # index = [0.5, 3.5]
    # ax = plt.subplot(111)
    # font_size = 25
    # bar_width = 0.35
    # index2 = [i + bar_width for i in index]
    # index3 = [i + 2 * bar_width for i in index]
    # index4 = [i + 3 * bar_width for i in index]
    # rects1 = plt.bar(index, y1, bar_width, color='white', label='高活跃度用户群')
    # rects2 = plt.bar(index2, y2, bar_width, color='silver', label='普通活跃度用户群')
    # rects3 = plt.bar(index3, y3, bar_width, color='dimgray', label='低活跃度用户群')
    # # # 给图加text
    # # plt.text(index2[0] + 0.2, y2[0] + 0.01, '39%', ha='center', va='bottom')
    # # plt.text(index2[1] + 0.2, y2[1] + 0.01, '1%', ha='center', va='bottom')
    # # plt.text(index2[2] + 0.2, y2[2] + 0.01, '-3%', ha='center', va='bottom')
    # #
    # # plt.text(index3[0] + 0.2, y3[0] + 0.01, '70%', ha='center', va='bottom')
    # # plt.text(index3[1] + 0.2, y3[1] + 0.01, '2%', ha='center', va='bottom')
    # # plt.text(index3[2] + 0.2, y3[2] + 0.01, '3%', ha='center', va='bottom')
    # ax.set_xlabel(r'评价指标', fontsize=font_size)  # change
    # # ax.set_ylabel(r'值')  # change
    # ax.set_xlim([0, None])
    # ax.set_ylim([0, 1])
    # index5 = [i + (bar_width / 2) + 0.2 for i in index]
    # plt.xticks(index5, [r'$\rm ILD$', r'Novelty-Norm'], fontsize=font_size)  # 注意不能使用ax.set_xticks()
    # plt.title('ML1M数据集', fontsize=font_size)  # change
    # ax.legend(loc='upper right', fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像
    #
    # # # Netflix
    # # y1 = [0.560 ,0.057]
    # # y2 = [0.527 ,0.045]
    # # y3 = [0.528 ,0.046]
    # # index = [0.5, 3.5]
    # # ax = plt.subplot(111)
    # # font_size = 25
    # # bar_width = 0.35
    # # index2 = [i + bar_width for i in index]
    # # index3 = [i + 2 * bar_width for i in index]
    # # index4 = [i + 3 * bar_width for i in index]
    # # rects1 = plt.bar(index, y1, bar_width, color='white', label='高活跃度用户群')
    # # rects2 = plt.bar(index2, y2, bar_width, color='silver', label='普通活跃度用户群')
    # # rects3 = plt.bar(index3, y3, bar_width, color='dimgray', label='低活跃度用户群')
    # # # # 给图加text
    # # # plt.text(index2[0] + 0.2, y2[0] + 0.01, '39%', ha='center', va='bottom')
    # # # plt.text(index2[1] + 0.2, y2[1] + 0.01, '1%', ha='center', va='bottom')
    # # # plt.text(index2[2] + 0.2, y2[2] + 0.01, '-3%', ha='center', va='bottom')
    # # #
    # # # plt.text(index3[0] + 0.2, y3[0] + 0.01, '70%', ha='center', va='bottom')
    # # # plt.text(index3[1] + 0.2, y3[1] + 0.01, '2%', ha='center', va='bottom')
    # # # plt.text(index3[2] + 0.2, y3[2] + 0.01, '3%', ha='center', va='bottom')
    # # ax.set_xlabel(r'评价指标', fontsize=font_size)  # change
    # # # ax.set_ylabel(r'值')  # change
    # # ax.set_xlim([0, None])
    # # ax.set_ylim([0, 1])
    # # index5 = [i + (bar_width / 2) + 0.2 for i in index]
    # # plt.xticks(index5, [r'$\rm ILD$', r'Novelty-Norm'], fontsize=font_size)  # 注意不能使用ax.set_xticks()
    # # plt.title('Netflix数据集', fontsize=font_size)  # change
    # # ax.legend(loc='upper right', fontsize=font_size)
    # # plt.tight_layout()  # 使布局更紧凑
    # # plt.show()  # 显示图像
    #
    # # MSD
    # y1 = [0.884 ,0.296]
    # y2 = [0.895,0.364]
    # y3 = [0.9,0.412]
    # index = [0.5, 3.5]
    # ax = plt.subplot(111)
    # font_size = 25
    # bar_width = 0.35
    # index2 = [i + bar_width for i in index]
    # index3 = [i + 2 * bar_width for i in index]
    # index4 = [i + 3 * bar_width for i in index]
    # rects1 = plt.bar(index, y1, bar_width, color='white', label='高活跃度用户群')
    # rects2 = plt.bar(index2, y2, bar_width, color='silver', label='普通活跃度用户群')
    # rects3 = plt.bar(index3, y3, bar_width, color='dimgray', label='低活跃度用户群')
    # # # 给图加text
    # # plt.text(index2[0] + 0.2, y2[0] + 0.01, '39%', ha='center', va='bottom')
    # # plt.text(index2[1] + 0.2, y2[1] + 0.01, '1%', ha='center', va='bottom')
    # # plt.text(index2[2] + 0.2, y2[2] + 0.01, '-3%', ha='center', va='bottom')
    # #
    # # plt.text(index3[0] + 0.2, y3[0] + 0.01, '70%', ha='center', va='bottom')
    # # plt.text(index3[1] + 0.2, y3[1] + 0.01, '2%', ha='center', va='bottom')
    # # plt.text(index3[2] + 0.2, y3[2] + 0.01, '3%', ha='center', va='bottom')
    # ax.set_xlabel(r'评价指标', fontsize=font_size)  # change
    # # ax.set_ylabel(r'值')  # change
    # ax.set_xlim([0, None])
    # ax.set_ylim([0, 1])
    # index5 = [i + (bar_width / 2) + 0.2 for i in index]
    # plt.xticks(index5, [r'$\rm ILD$', r'Novelty-Norm'], fontsize=font_size)  # 注意不能使用ax.set_xticks()
    # plt.title('MSD数据集', fontsize=font_size)  # change
    # ax.legend(loc='upper right', fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像

    # # nDCG和改进后的HD指标在不同活跃度用户上值的变动情况
    # # 画条形图。index有两个分量，分别对应[nDCG, Precision]
    # # ML1M
    # y1 = [0.633, 0.644]
    # y2 = [0.421, 0.447]
    # y3 = [0.262, 0.323]
    # index = [0.5, 3.5]
    # ax = plt.subplot(111)
    # font_size = 25
    # bar_width = 0.35
    # index2 = [i + bar_width for i in index]
    # index3 = [i + 2 * bar_width for i in index]
    # index4 = [i + 3 * bar_width for i in index]
    # rects1 = plt.bar(index, y1, bar_width, color='white', label='高活跃度用户群')
    # rects2 = plt.bar(index2, y2, bar_width, color='silver', label='普通活跃度用户群')
    # rects3 = plt.bar(index3, y3, bar_width, color='dimgray', label='低活跃度用户群')
    # # # 给图加text
    # # plt.text(index2[0] + 0.2, y2[0] + 0.01, '39%', ha='center', va='bottom')
    # # plt.text(index2[1] + 0.2, y2[1] + 0.01, '1%', ha='center', va='bottom')
    # # plt.text(index2[2] + 0.2, y2[2] + 0.01, '-3%', ha='center', va='bottom')
    # #
    # # plt.text(index3[0] + 0.2, y3[0] + 0.01, '70%', ha='center', va='bottom')
    # # plt.text(index3[1] + 0.2, y3[1] + 0.01, '2%', ha='center', va='bottom')
    # # plt.text(index3[2] + 0.2, y3[2] + 0.01, '3%', ha='center', va='bottom')
    # ax.set_xlabel(r'评价指标', fontsize=font_size)  # change
    # # ax.set_ylabel(r'值')  # change
    # ax.set_xlim([0, None])
    # ax.set_ylim([0, 1])
    # index5 = [i + (bar_width / 2) + 0.2 for i in index]
    # plt.xticks(index5, [r'$\rm nDCG$', r'$\rm HD$'], fontsize=font_size)  # 注意不能使用ax.set_xticks()
    # plt.title('ML1M数据集', fontsize=font_size)  # change
    # ax.legend(loc='upper left', bbox_to_anchor=(0.12, 1.05), fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像
    #
    # # MSD
    # y1 = [0.310, 0.72]
    # y2 = [0.189, 0.807]
    # y3 = [0.156, 0.867]
    # index = [0.5, 3.5]
    # ax = plt.subplot(111)
    # font_size = 25
    # bar_width = 0.35
    # index2 = [i + bar_width for i in index]
    # index3 = [i + 2 * bar_width for i in index]
    # index4 = [i + 3 * bar_width for i in index]
    # rects1 = plt.bar(index, y1, bar_width, color='white', label='高活跃度用户群')
    # rects2 = plt.bar(index2, y2, bar_width, color='silver', label='普通活跃度用户群')
    # rects3 = plt.bar(index3, y3, bar_width, color='dimgray', label='低活跃度用户群')
    # # # 给图加text
    # # plt.text(index2[0] + 0.2, y2[0] + 0.01, '39%', ha='center', va='bottom')
    # # plt.text(index2[1] + 0.2, y2[1] + 0.01, '1%', ha='center', va='bottom')
    # # plt.text(index2[2] + 0.2, y2[2] + 0.01, '-3%', ha='center', va='bottom')
    # #
    # # plt.text(index3[0] + 0.2, y3[0] + 0.01, '70%', ha='center', va='bottom')
    # # plt.text(index3[1] + 0.2, y3[1] + 0.01, '2%', ha='center', va='bottom')
    # # plt.text(index3[2] + 0.2, y3[2] + 0.01, '3%', ha='center', va='bottom')
    # ax.set_xlabel(r'评价指标', fontsize=font_size)  # change
    # # ax.set_ylabel(r'值')  # change
    # ax.set_xlim([0, None])
    # ax.set_ylim([0, 1])
    # index5 = [i + (bar_width / 2) + 0.2 for i in index]
    # plt.xticks(index5, [r'$\rm nDCG$', r'$\rm HD$'], fontsize=font_size)  # 注意不能使用ax.set_xticks()
    # plt.title('MSD数据集', fontsize=font_size)  # change
    # ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1), fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像

    # 在HPH算法参数lam遍历过程中，复合指标ADH对其的评测能力
    # 双坐标轴
    # ML1M
    x = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1]
    font_size = 25
    y1 = [0.196,0.477,0.519,0.511,0.495,0.468,0.449,0.436]
    y2 = [0.896,0.885,0.783,0.696,0.629 ,0.549 ,0.506 ,0.479]
    y3 = [0.651,0.725,0.677,0.62,0.573,0.513 ,0.480 ,0.458]
    fig, ax = plt.subplots()  # 创建图像文件。最好写成ax = plt.subplot(111)
    ax.plot(x, y2, 'kv:', mfc='k', linewidth=2.0, markersize=9, label=r'HD')
    ax.plot(x, y3, 'ks-', mfc='k', linewidth=2.0, markersize=9, label=r'ADH')
    ax.plot(x, y1, 'k^:', mfc='none', linewidth=2.0, markersize=9, label=r'nDCG')
    ax.set_xlim([-0.1, 1.1])
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel(r'$\lambda$', fontsize=font_size)
    ax = plt.gca()
    ax.yaxis.grid(True)
    # ax1.set_ylabel(r'$\rm nDCG, Metric, HD$', fontsize=font_size)
    plt.title('ML1M数据集', fontsize=font_size)
    ax.legend(loc='upper right', bbox_to_anchor=(1.02, 1.02), fontsize=font_size)
    plt.tight_layout()  # 使布局更紧凑
    plt.show()  # 显示图像

    # MSD
    x = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    font_size = 25
    # y1 = [0.024,0.04,0.077,0.129,0.171,0.204,0.226,0.238,0.241,0.232,0.211]  # all dots
    # y2 = [0.985 ,0.990 ,0.993 ,0.993 ,0.986 ,0.975 ,0.957 ,0.931 ,0.895 ,0.849 ,0.800]  # all dots
    # y3 = [0.551,0.559,0.575,0.596,0.609,0.617,0.617 ,0.608,0.589 ,0.56,0.524]
    y1 = [0.024,0.077,0.171,0.204,0.226,0.238,0.241,0.232,0.211]
    y2 = [0.985 ,0.993 ,0.986 ,0.975 ,0.957 ,0.931 ,0.895 ,0.849 ,0.800]
    y3 = [0.551,0.575,0.609,0.617,0.617 ,0.608,0.589 ,0.56,0.524]
    fig, ax = plt.subplots()  # 创建图像文件。最好写成ax = plt.subplot(111)
    ax.plot(x, y2, 'kv:', mfc='k', linewidth=2.0, markersize=9, label=r'HD')
    ax.plot(x, y3, 'ks-', mfc='k', linewidth=2.0, markersize=9, label=r'ADH')
    ax.plot(x, y1, 'k^:', mfc='none', linewidth=2.0, markersize=9, label=r'nDCG')
    ax.set_xlim([-0.1, 1.1])
    ax.set_xticks([0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_ylim([0, 1.1])
    ax.set_xlabel(r'$\lambda$', fontsize=font_size)
    ax = plt.gca()
    ax.yaxis.grid(True)
    # ax1.set_ylabel(r'$\rm nDCG, Metric, HD$', fontsize=font_size)
    plt.title('MSD数据集', fontsize=font_size)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 0.1), fontsize=font_size)
    plt.tight_layout()  # 使布局更紧凑
    plt.show()  # 显示图像

    # # 推荐结果的平均物品流行度
    # # ML1M
    # y1 = [4.338495640436849, 4.429379357062413, 4.262192154749766]
    # y2 = [1.9482214067259314, 1.8269899869745247, 2.112108145530308]
    # index = [0.5, 2.5, 4.5]
    # ax = plt.subplot(111)
    # font_size = 25
    # bar_width = 0.35
    # index2 = [i + bar_width for i in index]
    # rects1 = plt.bar(index, y1, bar_width, color='white', label='UCFQ')
    # rects2 = plt.bar(index2, y2, bar_width, color='dimgray', label='ICFN')
    # # # 给图加text
    # # plt.text(index2[0] + 0.2, y2[0] + 0.01, '39%', ha='center', va='bottom')
    # # plt.text(index2[1] + 0.2, y2[1] + 0.01, '1%', ha='center', va='bottom')
    # # plt.text(index2[2] + 0.2, y2[2] + 0.01, '-3%', ha='center', va='bottom')
    #
    # # ax.set_xlabel(r'测试集用户', fontsize=font_size)  # change
    # # ax.set_ylabel(r'值')  # change
    # ax.set_xlim([0, None])
    # ax.set_ylim([0, 5])
    # index5 = [index[0] + (bar_width / 2) + 0.1, index[1] + (bar_width / 2) + 0.2, index[2] + (bar_width / 2) + 0.4]
    # plt.xticks(index5, [r'高活跃度用户群', r'普通活跃度用户群', '低活跃度用户群'], fontsize=18)  # 注意不能使用ax.set_xticks()
    # plt.ylabel('物品对数流行度', fontsize=font_size)
    # plt.title('MSD数据集', fontsize=font_size)  # change
    # ax.legend(loc='upper right', bbox_to_anchor=(0.77, 1.02), fontsize=font_size)
    # ax = plt.gca()
    # ax.yaxis.grid(True)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像

    # # 重排序方法
    # # UCF算法
    # # ML1M
    # index = [0.5, 2.5, 4.5]
    # y1 = [0.781, 0.815, 0.908]
    # y2 = [0.712, 0.722, 0.467]
    # ax = plt.subplot(111)
    # font_size = 25
    # bar_width = 0.35
    # index2 = [i + bar_width for i in index]
    # rects1 = plt.bar(index, y1, bar_width, color='white', label='HD指标')
    # rects2 = plt.bar(index2, y2, bar_width, color='dimgray', label='nDCG指标')
    # # # 给图加text
    # # plt.text(index2[0] + 0.2, y2[0] + 0.01, '39%', ha='center', va='bottom')
    # # plt.text(index2[1] + 0.2, y2[1] + 0.01, '1%', ha='center', va='bottom')
    # # plt.text(index2[2] + 0.2, y2[2] + 0.01, '-3%', ha='center', va='bottom')
    # # ax.set_xlabel(r'测试集用户', fontsize=font_size)  # change
    # # ax.set_ylabel(r'值')  # change
    # ax.set_xlim([0, None])
    # ax.set_ylim([0.2, None])
    # index5 = [i + (bar_width / 2) + 0.1 for i in index]
    # plt.xticks(index5, [r'UCF', r'K(UCF)', r'TS(UCF)'], fontsize=font_size)  # 注意不能使用ax.set_xticks()
    # # plt.ylabel('', fontsize=font_size)
    # plt.title('ML1M数据集', fontsize=font_size)  # change
    # ax.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.1), fontsize=font_size)  # bbox_to_anchor=(0.755, 1)
    # plt.grid(True)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像
    #
    # # ICF算法
    # # ML1M
    # index = [0.5, 2.5, 4.5]
    # y1 = [0.812, 0.836, 0.969]
    # y2 = [0.698, 0.681, 0.617]
    # ax = plt.subplot(111)
    # font_size = 25
    # bar_width = 0.35
    # index2 = [i + bar_width for i in index]
    # rects1 = plt.bar(index, y1, bar_width, color='white', label='HD指标')
    # rects2 = plt.bar(index2, y2, bar_width, color='dimgray', label='nDCG指标')
    # # # 给图加text
    # # plt.text(index2[0] + 0.2, y2[0] + 0.01, '39%', ha='center', va='bottom')
    # # plt.text(index2[1] + 0.2, y2[1] + 0.01, '1%', ha='center', va='bottom')
    # # plt.text(index2[2] + 0.2, y2[2] + 0.01, '-3%', ha='center', va='bottom')
    # # ax.set_xlabel(r'测试集用户', fontsize=font_size)  # change
    # # ax.set_ylabel(r'值')  # change
    # ax.set_xlim([0, None])
    # ax.set_ylim([0, None])
    # index5 = [i + (bar_width / 2) + 0.1 for i in index]
    # plt.xticks(index5, [r'ICF', r'K(ICF)', r'TS(ICF)'], fontsize=font_size)  # 注意不能使用ax.set_xticks()
    # # plt.ylabel('', fontsize=font_size)
    # plt.title('ML1M数据集', fontsize=font_size)  # change
    # ax.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.1), fontsize=font_size)  # bbox_to_anchor=(0.755, 1)
    # plt.grid(True)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像

    # # 反向推荐重排序静态权重对推荐效果的影响
    # # ML1M数据集
    # # 高活跃度用户群
    # x = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # y1 = [0.633,0.634,0.631,0.615,0.591,0.493]
    # y2 = [0.648,0.654,0.669,0.685,0.726,0.771]
    # font_size = 25
    # bar_width = 0.35
    # fig, ax1 = plt.subplots()  # 创建图像文件。最好写成ax = plt.subplot(111)
    # ax2 = ax1.twinx()
    # ax1.plot(x, y1, 'kv:', mfc='none', linewidth=2.0, markersize=9, label='nDCG')
    # ax2.plot(x, y2, 'ks-', mfc='k', linewidth=2.0, markersize=9, label='HD')
    # ax1.set_xlim([-0.2, 1.2])
    # ax1.set_ylim([0, 1])
    # ax2.set_ylim([0, 1])
    # ax1.set_xlabel(r'反向得分值的权重', fontsize=font_size)
    # ax1.set_ylabel('nDCG', fontsize=font_size)
    # ax2.set_ylabel('HD', fontsize=font_size)
    # ax1.legend(loc='lower left', fontsize=font_size)
    # ax2.legend(loc='lower right', fontsize=font_size)
    # ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # ax1 = plt.gca()
    # ax1.yaxis.grid(True)
    # plt.title('高活跃度用户群', fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像
    #
    # # 低活跃度用户群
    # x = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # y1 = [0.262,0.264,0.266,0.269,0.254,0.133]
    # font_size = 25
    # bar_width = 0.35
    # fig, ax = plt.subplots()  # 创建图像文件。最好写成ax = plt.subplot(111)
    # ax.plot(x, y1, 'kv:', mfc='none', linewidth=2.0, markersize=9, label='nDCG')
    # ax.set_xlim([-0.2, 1.2])
    # ax.set_ylim([0, 0.4])
    # ax.set_xlabel(r'反向得分值的权重', fontsize=font_size)
    # ax.set_ylabel('nDCG', fontsize=font_size)
    # ax.legend(loc='lower left', fontsize=font_size)
    # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # ax = plt.gca()
    # ax.yaxis.grid(True)
    # plt.title('低活跃度用户群', fontsize=font_size)
    # plt.tight_layout()  # 使布局更紧凑
    # plt.show()  # 显示图像
