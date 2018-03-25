"""
为每一个用户做推荐，one by one。结果保存到外存磁盘。
// 注意，用户与用户之间没有顺序，但推荐结果有前后顺序。
// 手动设置推荐列表长度为最大（各个用户的推荐列表长度均不一样）。
"""

import os
import pickle
import recommend.cf as cf
import recommend.graph as graph
import recommend.hybrid as hybrid
import sys
from collections import defaultdict
import numpy as np


def recommend_onebyone(method, topn_file, current_directory, dataset, type, args):
    """
    每个用户最大保存前MAX_TOPN个推荐结果，不足MAX_TOPN时则保存其全部即可。
    // 注意，不足MAX_TOPN时如果为None，则不保存，直接跳过该用户target_user。所以切记之后的评价指标计算需要考虑边界条件。
    """

    # boundary condition
    if os.path.exists(topn_file):
        print('推荐结果topn文件已存在，不用再重新生成')
        return

    MAX_TOPN = 100
    args.update({'MAX_TOPN': MAX_TOPN})  # 推荐列表最大长度规定为MAX_TOPN
    users_pair = args['users_pair']
    items_pair = args['items_pair']

    # 推荐前
    methods = {
               'User CF q': cf.ucf_q,
               'User CF q Neighbordegree': cf.ucf_q_neighbordegree,
               'User CF Userdegree': cf.ucf_userdegree,
               'User CF Userdegree Neighbordegree': cf.ucf_userdegree_neighbordegree,
               'User CF q Neighbordegree2': cf.ucfq_neighbordegree2,

               'Item CF kNN Norm': cf.icf_knn_norm,
               'Item CF kNN Norm Itemdegree': cf.icf_knn_norm_itemdegree,

               'ProbS': graph.probs,
               'HeatS': graph.heats,
               'HPH': graph.hph,
               'PD': graph.pd,
               'ProbS Randomwalk': graph.probs_randomwalk,
               'ProbS Step1': graph.probs_step1,
               'ProbS Step3': graph.probs_step3,
               'ProbS Step1+3': graph.probs_step1_step3,

               'UCF Reranking Itemdegree': hybrid.ucfq_reranking_itemdegree,
               'ICF Reranking Itemdegree': hybrid.icfknnnorm_reranking_itemdegree,
               'ProbS Reranking Itemdegree': hybrid.probs_reranking_itemdegree,
               'UCF+ICF Reranking': hybrid.ucf_icf_reranking,
               'UCF+ICF Reranking TOPSIS': hybrid.ucfq_icf_reranking_topsis,
               'ProbS+HeatS Reranking': hybrid.probs_heats_reranking,
               'ProbS+HeatS Reranking TOPSIS': hybrid.probs_heats_reranking_topsis,
               'UCF Reranking DI TOPSIS': hybrid.ucfq_reranking_di_topsis,
               'ICF Reranking DI TOPSIS': hybrid.icfknnnorm_reranking_di_topsis,
               'ProbS Reranking DI TOPSIS': hybrid.probs_reranking_di_topsis,
               'TS(UCF) Weight': hybrid.ucfq_reranking_di_topsis_weight,
               'TS(ProbS) Weight': hybrid.probs_reranking_di_topsis_weight,

               'UCFu Reranking Itemdegree': hybrid.ucfu_reranking_itemdegree,
               'UCFu Reranking DI TOPSIS': hybrid.ucfu_reranking_di_topsis,
               'ICFj Reranking Itemdegree': hybrid.icfj_reranking_itemdegree,
               'ICFj Reranking DI TOPSIS': hybrid.icfj_reranking_di_topsis,
               'ProbSb Reranking DI TOPSIS': hybrid.probsb_reranking_di_topsis,

               'UCFuv Reranking Itemdegree': hybrid.ucfuv_reranking_itemdegree,
               'UCFuv+ICF Reranking TOPSIS': hybrid.ucfuv_icf_reranking_topsis,
               'UCFuv Reranking DI TOPSIS': hybrid.ucfuv_reranking_di_topsis,
               'ProbSb+HeatS Reranking TOPSIS': hybrid.probsb_heats_reranking_topsis
    }

    # 为种子用户做推荐，生成反向推荐结果
    print('种子用户推荐...')

    method_seed = None
    if 'UCF' in method:
        method_seed = 'User CF q'
    elif 'ICF' in method:
        method_seed = 'Item CF kNN Norm'
    elif 'ProbS' in method:
        method_seed = 'ProbS'
    else:
        print('Warning. Seed users are not considered!')
        method_seed = 'User CF q'
    pickle_filepath_method = r'{0}\{1}\{1}_out\seed-0.01{2}-{3}'.format(current_directory, dataset, type, method_seed)

    items_users_score = get_seed_recommendation(pickle_filepath_method, method_seed, methods, args)
    args.update({'items_users_score': items_users_score})

    # 推荐中
    count_invalidtargetuser = 0
    with open(topn_file, 'w') as file_write:
        count = 0
        for target_user in users_pair.keys():  # 为train文件中的每一个用户做推荐，one by one
            # 进度条
            count += 1
            sys.stdout.write('\r为train文件中的每一个用户做推荐：{0} / {1}'.format(count, len(users_pair)))
            sys.stdout.flush()

            items_score = methods[method](target_user, args)

            # first_part按score值从大到小排序
            # 剩余的，second_part按照物品id编号从小到大排序。（这样就和Matlab程序的生成结果一样了）
            row = []

            # boundary condition
            if items_score is None or len(items_score) == 0:  # 顺序不能颠倒
                count_invalidtargetuser += 1
            else:
                first_part = [item for (item, score) in sorted(items_score.items(), key=lambda a: a[1], reverse=True)]
                if len(first_part) >= MAX_TOPN:
                    row = first_part[:MAX_TOPN]  # 截取前MAX_TOPN个推荐结果
                else:  # 注意到这里有可能会“出错”，如果用户几乎交互了全体物品，那么生成的推荐列表row长度绝对不会满足MAX_TOPN的
                    # 集合的交并操作很费时。（如果想改进，可以把它放到循环中慢慢判断。）
                    items_other = set(items_pair.keys()) - users_pair[target_user] - set(items_score.keys())
                    second_part = sorted(items_other)
                    # 截取前MAX_TOPN个推荐结果。注意到，总长度不足MAX_TOPN时，则自动保存其全部
                    row = (first_part + second_part)[:MAX_TOPN]

            # 写入
            row_string_format = [str(item) for item in row]
            file_write.write(str(target_user) + ',')
            file_write.write(' '.join(row_string_format))  # 分隔符使用空格' '
            file_write.write('\n')

    print('\n测试集testall中共{}用户，其中无效用户（使用的推荐算法产生的候选集为空）有{}个'.format(len(users_pair), count_invalidtargetuser))
    return


def get_seed_recommendation(pickle_filepath_method, method_seed, methods, args):
    # 为种子用户推荐。

    # boundary condition
    if os.path.exists(pickle_filepath_method):
        f = open(pickle_filepath_method, 'rb')
        items_users_score = pickle.load(f)
        f.close()
        return items_users_score

    # 反向推荐结果
    items_users_score = {}
    users_seed = args.get('users_seed', None)
    count = 0
    for user in users_seed:
        # 进度条
        if count % 100 == 0:
            print('已处理{0}/{1}'.format(count, len(users_seed)))
        count += 1


        items_score = methods[method_seed](user, args)
        # boundary condition
        if items_score is None or len(items_score) == 0:
            continue
        for (item, score) in items_score.items():
            items_users_score.setdefault(item, {})
            items_users_score[item][user] = score

    # # 异常值（outlier）的判别与剔除
    # items_users_score_valid = defaultdict(lambda: dict())
    # count_all = 0
    # count_valid = 0
    # for (item_new, users_score) in items_users_score.items():
    #     mean_seed = np.mean(list(items_users_score[item_new].values()))
    #     std_seed = np.std(list(items_users_score[item_new].values()))
    #
    #     for (user, score) in users_score.items():
    #         z_score = (score - mean_seed) / std_seed
    #         if -3 <= z_score <= 3:
    #             items_users_score_valid[item_new][user] = score
    #             count_valid += 1
    #         count_all += 1

    f = open(pickle_filepath_method, 'wb')
    pickle.dump(items_users_score, f)
    f.close()

    return items_users_score
