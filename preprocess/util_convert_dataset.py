"""
推荐前。
"""

import numpy as np
import sys
sys.path.append(r'D:\recommender')
sys.path.append(r'F:\recommender')
import preprocess.degree_distribution as myutil
import preprocess.util_split_traintest as myutil2
import random
from collections import defaultdict
import gc


# 去除无效字符、去除缺失值，使数据集变得规整和完整
def removeinvalid_bookcrossing(in_file, out_file):
    # 去除一些行。那些行可能包含有无效字符，也可能有缺失值。
    # // 注意不改变原有的分隔符格式。

    col_user = []  # 保存过滤后每行的第一列的值
    col_item = []
    col_rating = []

    in_read = open(in_file, 'r')
    whole = in_read.readlines()
    in_read.close()
    whole = whole[1:]  # 去除第一行

    # 开始进行过滤remove无效行
    count_invalid_line = 0
    count_all_line = 0
    for line in whole:
        count_all_line += 1
        line_split = line.strip().split(';')  # notice

        if len(line_split) != 3:
            count_invalid_line += 1
            print('列数不为3：', end='')
            print(line, end='')
            continue

        user = line_split[0]
        item = line_split[1]
        rating = line_split[2]

        if not (user.startswith('"') and user.endswith('"')):
            count_invalid_line += 1
            print('user列格式不规范：', end='')
            print(line, end='')
            continue
        user = user.lstrip('"').rstrip('"')
        if not user.isdigit():  # user列格式应该是纯数字
            count_invalid_line += 1
            print('user列格式不规范：', end='')
            print(line, end='')
            continue

        if not (item.startswith('"') and item.endswith('"')):
            count_invalid_line += 1
            print('item列格式不规范：', end='')
            print(line, end='')
            continue
        # 重要。ISBN无论旧10位还是新13位，都只有数字或者仅多含一个大写的X（表示10）
        item = item.lstrip('"').rstrip('"')
        if not item.isalnum():  # item列格式应该是只有数字和字母混合
            count_invalid_line += 1
            print('item列格式不规范：', end='')
            print(line, end='')
            continue

        if not (rating.startswith('"') and rating.endswith('"')):
            count_invalid_line += 1
            print('rating列格式不规范：', end='')
            print(line, end='')
            continue
        rating = rating.lstrip('"').rstrip('"')
        try:  # rating列格式应该是只是整数或小数
            x = float(rating)
        except ValueError:
            count_invalid_line += 1
            print('rating列格式不规范：', end='')
            print(line, end='')
            continue

        #
        col_user.append(user)
        col_item.append(item)
        col_rating.append(rating)
    print('原始数据集有{0}行，其中无效行有{1}'.format(count_all_line, count_invalid_line))

    # 把过滤后的文件，保存到外存磁盘
    data = zip(col_user, col_item, col_rating)
    count_all_line = 0
    count_zero_line = 0
    with open(out_file, 'w') as out_write:
        for (u, i, r) in data:
            #
            count_all_line += 1
            if int(r) == 0:
                count_zero_line += 1

            out_write.write(u + ';' + i + ';' + r)
            out_write.write('\n')

    m = len(np.unique(col_user))
    n = len(np.unique(col_item))
    rmin = np.min([int(r) for r in col_rating])
    rmax = np.max([int(r) for r in col_rating])
    print('{0}用户, {1}图书， 评分最小{2}, 评分最大{3}'.format(m, n, rmin, rmax))
    print('过滤后新的数据集有{0}行，其中rating等于0的行有{1}'.format(count_all_line, count_zero_line))
    return
def removeinvalid_bookcrossing_isbn(in_file, out_file):
    # 进一步过滤。主要针对旧ISBN号。

    col_item = []  # 保存过滤后每行的第二列（ISBN图书编号）的值

    in_read = open(in_file, 'r')
    whole = in_read.readlines()
    in_read.close()

    # 过滤，然后保存到外存磁盘
    with open(out_file, 'w') as out_write:
        count_line_remove = 0
        for line in whole:
            line_split = line.split(';')
            item_isbn = line_split[1]

            # 过滤remove
            if len(item_isbn) != 10 and len(item_isbn) != 13:
                count_line_remove += 1
                continue

            # 过滤remove。因为ISBN无论旧10位还是新13位，都只有数字或者仅末尾含有一个大写的X（表示10）
            if item_isbn.endswith('X'):
                part = item_isbn.rstrip('X')
                if not part.isdigit():
                    count_line_remove += 1
                    continue

            # 保存过滤后的内容，保存到外存磁盘
            out_write.write(line)
            col_item.append(item_isbn)
    print('共有{0}图书'.format(len(np.unique(col_item))))
    return
def removeinvalid_bookcrossing_itemdegree(in_file, out_file):
    # 因为发现经过以上2个函数的过滤，物品数仍有33万，而实际官方文档指出应该只有27万。所以我猜测是因为有些ISBN编号扭曲了，导致物品数变大。
    # 我考虑过滤掉这些可能ISBN编号扭曲的ISBN。（通过物品度来过滤。）
    # // 但是我最终没有使用该函数，因为经过过滤，物品数变为19万，比27万小了太多，慎重起见，我没有使用该函数。

    in_read = open(in_file, 'r')
    whole = in_read.readlines()
    in_read.close()

    items_pair = {}
    for line in whole:
        line_split = line.strip().split(';')
        user = int(line_split[0])
        item = line_split[1]
        items_pair.setdefault(item, set())
        items_pair[item].add(user)

    all_items = len(items_pair)
    invalid_items1 = [1 for (item, users) in items_pair.items() if len(users) == 0]
    invalid_items2 = [1 for (item, users) in items_pair.items() if len(users) == 1]
    print('共{}图书，其中度为0的有{}部，度为1的有{}部'.format(all_items, len(invalid_items1), len(invalid_items2)))
    return
def removeinvalid_netflix(in_file, out_file):
    # 去除一些行。那些行可能包含有无效字符，也可能有缺失值。
    # // 注意不改变原有的分隔符格式。
    pass


# 将id标准化成数字形式
def normalize_batch(df, out_file):
    # // 该函数已删除不使用。
    # 去除nan等无效值；
    # 去除重复行；
    # 统一id（把字符串映射成整数）。

    # 去除nan等无效值
    df.dropna(inplace=True)

    # 去除重复行
    #...

    # 统一id（把字符串id映射成整数id）
    users_id = {}
    users_name = []

    items_id = {}
    items_name = []

    print('用户id标准化')
    for row in df.values:
        user_name = str(row[0])
        # 编号（标准化）
        if user_name not in users_id:
            users_name.append(user_name)
            users_id[user_name] = len(users_name) - 1

    print('物品id标准化')
    for row in df.values:
        item_name = str(row[1])
        # 编号（标准化）
        if item_name not in items_id:
            items_name.append(item_name)
            items_id[item_name] = len(items_name) - 1

    # 保存到外存磁盘
    print('经过数据预处理后，把结果写出到外存')
    with open(out_file, 'w') as f:
        has_timestamp = True
        if df.values.shape[1] <= 3:
            has_timestamp = False
        else:
            has_timestamp = True

        for row in df.values:
            user_name = str(row[0])
            item_name = str(row[1])
            user_id = users_id[user_name] + 1
            item_id = items_id[item_name] + 1
            rating = float(row[2])

            line = None
            if has_timestamp:
                timestamp = str(row[3])
                line = (str(user_id), str(item_id), str(rating), str(timestamp))
            else:
                line = (str(user_id), str(item_id), str(rating))

            f.write('::'.join(line))  # 分隔符也统一使用'::'
            f.write('\n')
    return
def normalize_timestamp(in_file, sep, out_file):
    # 目标：id标准化。
    # // 注意不改变原有的分隔符格式。
    # // 模板函数。

    # id标准化（把字符串id映射成整数id）
    users_id = {}
    items_id = {}
    with open(in_file, 'r') as in_read:
        count = 0
        for line in in_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r进行数据预处理：已处理{0}万行'.format(count / 10000))
                sys.stdout.flush()
            count += 1

            row = line.strip().split(sep)

            # boundary condition
            if row[0].startswith('"') or row[0].endswith('"'):
                print('数据集没有经过最初的预处理，还保留有双引号""。id标准化程序无法运行！')
                exit(0)
            if row[1].startswith('"') or row[1].endswith('"'):
                print('数据集没有经过最初的预处理，还保留有双引号""。id标准化程序无法运行！')
                exit(0)
            if row[2].startswith('"') or row[2].endswith('"'):
                print('数据集没有经过最初的预处理，还保留有双引号""。id标准化程序无法运行！')
                exit(0)
            if row[3].startswith('"') or row[3].endswith('"'):
                print('数据集没有经过最初的预处理，还保留有双引号""。id标准化程序无法运行！')
                exit(0)

            user_name = row[0]
            item_name = row[1]

            # 编号（标准化）。从1开始，与标准数据集（例如Movielens）保持一致
            if user_name not in users_id:
                users_id[user_name] = len(users_id) + 1

            # 编号（标准化）。从1开始，与标准数据集（例如Movielens）保持一致
            if item_name not in items_id:
                items_id[item_name] = len(items_id) + 1

    # id标准化，保存到外存磁盘
    print('\n进行id标准化，把结果写出到外存...')
    with open(out_file, 'w') as out_write:
        with open(in_file, 'r') as in_read:

            count = 0
            for line in in_read:
                # 进度条
                if count % 10000 == 0:
                    sys.stdout.write('\r进行数据预处理：已处理{0}万行'.format(count / 10000))
                    sys.stdout.flush()
                count += 1

                row = line.strip().split(sep)
                user_name = row[0]
                user_id = users_id[user_name]
                item_name = row[1]
                item_id = items_id[item_name]
                rating = row[2]
                timestamp = row[3]

                r = (str(user_id), str(item_id), rating, timestamp)
                out_write.write(sep.join(r))
                out_write.write('\n')
    return
def normalize_notimestamp(in_file, sep, out_file):
    # 目标：id标准化。
    # // 注意不改变原有的分隔符格式。
    # // 模板函数。

    # id标准化（把字符串id映射成整数id）
    users_id = {}
    items_id = {}
    with open(in_file, 'r') as in_read:
        count = 0
        for line in in_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r进行数据预处理：已处理{0}万行'.format(count / 10000))
                sys.stdout.flush()
            count += 1

            row = line.strip().split(sep)

            # boundary condition
            if row[0].startswith('"') or row[0].endswith('"'):
                print('数据集没有经过最初的预处理，还保留有双引号""。id标准化程序无法运行！')
                exit(0)
            if row[1].startswith('"') or row[1].endswith('"'):
                print('数据集没有经过最初的预处理，还保留有双引号""。id标准化程序无法运行！')
                exit(0)
            if row[2].startswith('"') or row[2].endswith('"'):
                print('数据集没有经过最初的预处理，还保留有双引号""。id标准化程序无法运行！')
                exit(0)

            user_name = row[0]
            item_name = row[1]

            # 编号（标准化）。从1开始，与标准数据集（例如Movielens）保持一致
            if user_name not in users_id:
                users_id[user_name] = len(users_id) + 1

            # 编号（标准化）。从1开始，与标准数据集（例如Movielens）保持一致
            if item_name not in items_id:
                items_id[item_name] = len(items_id) + 1

    # id标准化，保存到外存磁盘
    print('\n进行id标准化，把结果写出到外存...')
    with open(out_file, 'w') as out_write:
        with open(in_file, 'r') as in_read:

            count = 0
            for line in in_read:
                # 进度条
                if count % 10000 == 0:
                    sys.stdout.write('\r进行数据预处理：已处理{0}万行'.format(count / 10000))
                    sys.stdout.flush()
                count += 1

                row = line.strip().split(sep)
                user_name = row[0]
                user_id = users_id[user_name]
                item_name = row[1]
                item_id = items_id[item_name]
                rating = row[2]

                r = (str(user_id), str(item_id), rating)
                out_write.write(sep.join(r))
                out_write.write('\n')
    return
def normalize(in_file, sep, out_file):
    # 目标：id标准化。
    # // 注意不改变原有的分隔符格式。

    # 先检查有几列
    in_read = open(in_file, 'r')
    first_line = in_read.readline()
    in_read.close()
    line_split = first_line.strip().split(sep)
    count_cols = len(line_split)

    #
    if count_cols != 3 and count_cols != 4:
        print('Error. 数据集有特殊的列，需要单独处理')
        exit(1)
    methods = {3: normalize_notimestamp,
               4: normalize_timestamp}
    methods[count_cols](in_file, sep, out_file)
    return


# 去除重复行
def unique_simple_timestamp(in_file, sep, out_file):
    # 难点。主键决定是否是重复的行。此函数认为u-i是主键，而不是认为u-i-t（如果有timestamp的话）是主键。
    # 目标：(1)检查并去除重复行；(2)把分隔符sep统一转换成','。
    # // 模板函数。

    # 去除重复行，保存到外存磁盘
    lens_duplicate = 0
    unique_lines = set()  # 可能内存存不下来，需要优化
    print('去除重复行，把结果写出到外存...')
    with open(out_file, 'w') as out_write:

        in_read = open(in_file, 'r')
        count = 0
        for line in in_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r进行数据预处理：已处理{0}万行'.format(count / 10000))
                sys.stdout.flush()
            count += 1

            row = line.strip().split(sep)

            user = int(row[0])
            item = int(row[1])
            rating = row[2]
            timestamp = row[3]

            if (user, item) not in unique_lines:
                unique_lines.add((user, item))
                r = (str(user), str(item), rating, timestamp)
                out_write.write(','.join(r))  # 分隔符统一使用','
                out_write.write('\n')
            else:
                lens_duplicate += 1
                continue
        in_read.close()

    if lens_duplicate == 0:  # boundary condition
        print('\n原始数据集没有重复行')
    else:
        print('\n原始数据集有重复行，共{0}行'.format(lens_duplicate))
    return
def unique_simple_notimestamp(in_file, sep, out_file):
    # 目标：(1)检查并去除重复行；(2)把分隔符sep统一转换成','。
    # // 模板函数。

    # 去除重复行，保存到外存磁盘
    lens_duplicate = 0
    unique_lines = set()  # 可能内存存不下来，需要优化
    print('去除重复行，把结果写出到外存...')
    with open(out_file, 'w') as out_write:

        in_read = open(in_file, 'r')
        count = 0
        for line in in_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r进行数据预处理：已处理{0}万行'.format(count / 10000))
                sys.stdout.flush()
            count += 1

            row = line.strip().split(sep)

            user = int(row[0])
            item = int(row[1])
            rating = row[2]

            if (user, item) not in unique_lines:
                unique_lines.add((user, item))
                r = (str(user), str(item), rating)
                out_write.write(','.join(r))  # 分隔符统一使用','
                out_write.write('\n')
            else:
                lens_duplicate += 1
                continue
        in_read.close()

    if lens_duplicate == 0:  # boundary condition
        print('\n原始数据集没有重复行')
    else:
        print('\n原始数据集有重复行，共{0}行'.format(lens_duplicate))
    return
def unique_simple(in_file, sep, out_file):
    # 目标：(1)检查并去除重复行；(2)把分隔符sep统一转换成','。

    # 先检查有几列
    in_read = open(in_file, 'r')
    first_line = in_read.readline()
    in_read.close()
    line_split = first_line.strip().split(sep)
    count_cols = len(line_split)

    #
    if count_cols != 3 and count_cols != 4:
        print('Error. 数据集有特殊的列，需要单独处理')
        exit(1)
    methods = {3: unique_simple_notimestamp,
               4: unique_simple_timestamp}
    methods[count_cols](in_file, sep, out_file)
    return
def unique_complete(in_file, sep, out_file):
    # 无法将简单的u-i当做主键时，编写此函数。
    # // 是作为上面unique_simple()函数的复杂化改写。
    pass


def print_data(in_file):
    # 统计数据集的基本信息。

    count = 0
    with open(in_file, 'r') as file_read:
        users_degree = defaultdict(lambda: 0)
        items_degree = defaultdict(lambda: 0)
        for line in file_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r统计用户度和物品度：已处理{0}万行'.format(count/10000))
                sys.stdout.flush()
            count += 1

            row = line.split(',')
            user = int(row[0])  # 报错说明数据集不规范
            item = int(row[1])
            users_degree[user] += 1
            items_degree[item] += 1

    # 统计数据集的基本信息
    print("\n统计信息：\n\t用户数：{0}\n\t物品数：{1}\n\t评分数：{2}\n\t稠密度：{3}%".format(
        len(users_degree),
        len(items_degree),
        sum(users_degree.values()),
        round(100.0*sum(users_degree.values())/len(users_degree)/len(items_degree), 3)
        )
    )

    # 用户度，物品度
    userdegrees = list(users_degree.values())  # 主要为了统计用户度
    itemdegrees = list(items_degree.values())  # 主要为了统计用户度
    print("度分布信息：\n\t用户度最小：{0}\t用户度最大{1}\n\t物品度最小：{2}\t物品度最大：{3}".format(
        min(userdegrees), max(userdegrees), min(itemdegrees), max(itemdegrees)
        )
    )
    print()
    return
def print_data_row(in_file, sep):
    # sep不再是统一的','。
    # 统计数据集的基本信息。

    count = 0
    with open(in_file, 'r') as file_read:
        users_degree = defaultdict(lambda: 0)
        items_degree = defaultdict(lambda: 0)
        for line in file_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r统计用户度和物品度：已处理{0}万行'.format(count / 10000))
                sys.stdout.flush()
            count += 1

            row = line.strip().split(sep)
            user = row[0]
            item = row[1]
            users_degree[user] += 1
            items_degree[item] += 1

        # 统计数据集的基本信息
        print("\n统计信息：\n\t用户数：{0}\n\t物品数：{1}\n\t评分数：{2}\n\t稠密度：{3}%".format(
            len(users_degree),
            len(items_degree),
            sum(users_degree.values()),
            round(100.0 * sum(users_degree.values()) / len(users_degree) / len(items_degree), 3)
            )
        )

        # 用户度，物品度
        userdegrees = list(users_degree.values())  # 主要为了统计用户度
        itemdegrees = list(items_degree.values())  # 主要为了统计用户度
        print("度分布信息：\n\t用户度最小：{0}\t用户度最大{1}\n\t物品度最小：{2}\t物品度最大：{3}".format(min(userdegrees), max(userdegrees), min(itemdegrees), max(itemdegrees)))
        print()
        return


def explicit_to_implicit_fivescale(in_file, out_file, threshold=2.0):
    # 按照threshold来隐式化。
    # // 不改变csv文件里列名、列的数量等格式。

    with open(out_file, 'w') as out_write:
        with open(in_file, 'r') as in_read:
            count = 0
            for line in in_read:
                # 进度条
                if count % 10000 == 0:
                    sys.stdout.write('\r隐式化：已处理{0}万行'.format(count / 10000))
                    sys.stdout.flush()
                count += 1

                row = line.strip().split(',')
                rating = float(row[2])
                if rating > threshold:
                    out_write.write(line)
    print()
    return
def explicit_to_implicit_eumr(in_file, out_file):
    # 按照Each User's Mid-Rating来隐式化。
    # // 不改变csv文件里列名、列的数量等格式。

    # 确定每个用户的评分尺度rating scale
    users_rating_scale = {}
    users_explicit_rating = {}  # {user1: [max_rating, min_rating]}
    with open(in_file, 'r') as in_read:
        count = 0
        for line in in_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r确定每个用户的评分尺度rating scale：已处理{0}万行'.format(count / 10000))
                sys.stdout.flush()
            count += 1

            row = line.strip().split(',')
            user = int(row[0])
            rating = float(row[2])

            users_explicit_rating.setdefault(user, [-sys.maxsize, sys.maxsize])
            if users_explicit_rating[user][0] < rating:
                users_explicit_rating[user][0] = rating
            if users_explicit_rating[user][1] > rating:
                users_explicit_rating[user][1] = rating
    users_rating_scale = {user: (maxmin[0] + maxmin[1]) * 1.0 / 2 for (user, maxmin) in users_explicit_rating.items()}

    # 隐式化
    print()
    with open(out_file, 'w') as out_write:
        in_read = open(in_file, 'r')
        count = 0
        for line in in_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r隐式化：已处理{0}万行'.format(count / 10000))
                sys.stdout.flush()
            count += 1

            row = line.strip().split(',')
            user = int(row[0])
            rating = float(row[2])

            # 注意是>=mid都可以
            if rating < users_rating_scale[user]:
                continue
            else:
                out_write.write(line)
        in_read.close()
    print()
    return
def explicit_to_implicit_bookcrossing(in_file, out_file):
    # 按照Each User's Mid-Rating来隐式化。
    # // 不改变csv文件里列名、列的数量等格式。
    #
    # 注意到Book-Crossing数据集有一部分隐反馈数据（rating==0的行），这些行不用过滤。（不过滤的另一个原因是，数据集共100万条评分，而上述的隐反馈数据就有70万条占了70%。）

    # 确定每个用户的评分尺度rating scale
    users_explicit_rating = {}  # {user1: [max_rating, min_rating]}
    users_rating_scale = {}  # {user1: mid}
    with open(in_file, 'r') as in_read:
        count = 0
        for line in in_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r确定每个用户的评分尺度rating scale：已处理{0}万行'.format(count / 10000))
                sys.stdout.flush()
            count += 1

            row = line.strip().split(',')
            user = int(row[0])
            rating = float(row[2])

            users_explicit_rating.setdefault(user, [-sys.maxsize, sys.maxsize])
            if users_explicit_rating[user][0] < rating:
                users_explicit_rating[user][0] = rating
            if users_explicit_rating[user][1] > rating:
                users_explicit_rating[user][1] = rating
    users_rating_scale = {user: (maxmin[0] + maxmin[1]) * 1.0 / 2 for (user, maxmin) in users_explicit_rating.items()}

    # 隐式化
    print()
    with open(out_file, 'w') as out_write:
        in_read = open(in_file, 'r')
        count = 0
        for line in in_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r隐式化：已处理{0}万行'.format(count / 10000))
                sys.stdout.flush()
            count += 1

            row = line.strip().split(',')
            user = int(row[0])
            rating = float(row[2])

            # 注意是>=mid都可以
            if rating != float(0) and rating < users_rating_scale[user]:
                continue
            else:
                out_write.write(line)
        in_read.close()
    print()
    return
def remove_toobiguser(in_file, out_file, topvalue=0.005):
    # 保留用户度<trd的物品，删除度>=trd的用户。
    # 去除前topvalue比例的（过于活跃的）用户。

    users_degree = myutil.get_userdegree(in_file)
    (X, Y) = myutil.degree_pdf_and_cdf(users_degree,
        dataset=None, degreetype=None, get_pdf=False, get_cdf=True, savecdf=False)
    toobig_threshold = myutil2.find_percent(X, Y, topvalue)

    # 保存到外存磁盘
    with open(out_file, 'w') as out_write:
        with open(in_file, 'r') as in_read:
            count = 0
            for line in in_read:
                # 进度条
                if count % 10000 == 0:
                    sys.stdout.write('\r进行数据预处理：已处理{0}万行'.format(count / 10000))
                    sys.stdout.flush()
                count += 1

                r = line.split(',')
                user = int(r[0])
                item = int(r[1])
                # 注意是>=trd的记录都剔除掉
                if users_degree[user] < toobig_threshold:
                    out_write.write(str(user) + ',' + str(item) + '\n')
    print()
    print()
    return


def remove_smalluser(in_file, out_file, smalldegree_threshold):
    # 保留用户度>trd的物品，删除度<=trd的用户。
    # 注意到有顺序关系（两个方法调用有顺序关系）。一般先过滤小度用户，再过滤小度物品（尽量不过滤物品）。

    # 用户度
    users_degree = defaultdict(lambda: 0)
    with open(in_file, 'r') as in_read:
        count = 0
        for line in in_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r过滤小度用户。统计用户度的分布：已处理{0}万行'.format(count / 10000))
                sys.stdout.flush()
            count += 1

            r = line.strip().split(',')
            user = int(r[0])  # 报错说明数据集不规范
            users_degree[user] += 1

    # 保存到外存磁盘
    with open(out_file, 'w') as out_write:
        with open(in_file, 'r') as in_read:
            count = 0
            for line in in_read:
                # 进度条
                if count % 10000 == 0:
                    sys.stdout.write('\r过滤小度用户。保存到外存磁盘：已处理{0}万行'.format(count / 10000))
                    sys.stdout.flush()
                count += 1

                r = line.split(',')
                user = int(r[0])
                item = int(r[1])
                if users_degree[user] > smalldegree_threshold:  # 即删除度<smalldegree_threshold的用户
                    out_write.write(str(user) + ',' + str(item) + '\n')
    print()
    return
def remove_smallitem(in_file, out_file, smalldegree_threshold):
    # 保留物品度>trd的物品，删除度<=trd的物品。
    # // 尽量不过滤物品。但是过滤度为1的物品的原因是该(u,i)样本最后只能被划分到训练集train，且对于CF推荐算法来说，没有起到贡献。

    # 物品度
    items_degree = defaultdict(lambda: 0)
    with open(in_file, 'r') as in_read:
        count = 0
        for line in in_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r统计上一个数据集中的物品度分布，为过滤小度物品做准备：已处理{0}万行'.format(count/10000))
                sys.stdout.flush()
            count += 1

            r = line.strip().split(',')
            item = int(r[1])  # 报错说明数据集不规范
            items_degree[item] += 1
    print()

    # 保存到外存磁盘
    with open(out_file, 'w') as out_write:
        with open(in_file, 'r') as in_read:
            count = 0
            for line in in_read:
                # 进度条
                if count % 10000 == 0:
                    sys.stdout.write('\r过滤小度物品，保存到外存磁盘：已处理{0}万行'.format(count/10000))
                    sys.stdout.flush()
                count += 1

                r = line.strip().split(',')
                user = int(r[0])
                item = int(r[1])
                if items_degree[item] > smalldegree_threshold:
                    out_write.write(str(user) + ',' + str(item) + '\n')
    print()
    print()
    return


def filter_data(in_file, out_file, ku_threshold, ki_threshold):
    # 以Book-Crossing数据集为原型写成。

    # 读入数据集
    users_pair = defaultdict(set)
    with open(in_file, 'r') as in_read:
        count = 0
        for line in in_read:
            # 进度条
            if count % 10000 == 0:
                sys.stdout.write('\r读入数据：已处理{0}万行'.format(count / 10000))
                sys.stdout.flush()
            count += 1

            r = line.strip().split(',')
            user = int(r[0])  # 报错说明数据集不规范
            item = int(r[1])
            users_pair[user].add(item)
    print()

    # # 递归式过滤（压缩数据集）
    # users_pair_new = flt_old1(users_pair, ku_threshold, ki_threshold)
    # print('把压缩后的数据集保存到外存磁盘...')
    # with open(out_file, 'w') as out_write:
    #     for (user, items) in users_pair_new.items():  # 注意变量名是否带_new
    #         for item in items:
    #             line = str(user) + ',' + str(item) + '\n'  # 只有两列。
    #             out_write.write(line)
    # return

    # # 递归式过滤（压缩数据集）
    # flt_old2(users_pair, ku_threshold, ki_threshold)
    # print('把压缩后的数据集保存到外存磁盘...')
    # with open(out_file, 'w') as out_write:
    #     for (user, items) in users_pair.items():  # 注意变量名是否带_new
    #         for item in items:
    #             line = str(user) + ',' + str(item) + '\n'  # 只有两列。
    #             out_write.write(line)
    # return

    # 迭代式过滤（压缩数据集）
    flt(users_pair, ku_threshold, ki_threshold)
    # 保存到外存磁盘
    print('正在把压缩后的数据集保存到外存磁盘...')
    with open(out_file, 'w') as out_write:
        for (user, items) in users_pair.items():  # 注意变量名是否带_new
            for item in items:
                line = str(user) + ',' + str(item) + '\n'  # 只有两列。
                out_write.write(line)
    return
def flt_old1(users_pair_old, ku_threshold, ki_threshold, is_end=False, itecount=0):
    # 递归式地删除小度用户结点和小度物品结点。（注意，主要通过删除小度用户结点来压缩数据集。）
    # // 已淘汰。

    # 递归出口
    if is_end:
        return users_pair_old

    # 递归主体
    is_end = True

    print('\n第{0}次迭代，正在删除小度用户...'.format(itecount))
    users_pair_new = {user: items for (user, items) in users_pair_old.items() if len(items) >= ku_threshold}
    if len(users_pair_new) < len(users_pair_old):
        is_end = False
    else:
        pass
    items_pair_new = ui_transe(users_pair_new)

    print('第{0}次迭代，正在删除小度物品...'.format(itecount))
    items_pair_new2 = {item: users for (item, users) in items_pair_new.items() if len(users) >= ki_threshold}
    if len(items_pair_new2) < len(items_pair_new):
        is_end = False
    else:
        pass
    users_pair_new2 = ui_transe(items_pair_new2)

    # 打印数据集基本信息
    m = len(users_pair_new2)
    n = len(items_pair_new2)
    count_ratings = sum([len(items) for items in users_pair_new2.values()])
    print('用户数：{0}\t物品数： {1}\t评分数：{2}'.format(m, n, count_ratings))

    return flt_old1(users_pair_new2, ku_threshold, ki_threshold, is_end, itecount+1)  # 切记itecount++
def flt_old2(users_pair, ku_threshold, ki_threshold, is_end=False, itecount=0):
    # 递归式地删除小度用户结点和小度物品结点。（注意，主要通过删除小度用户结点来压缩数据集。）
    # // 节省内存。在字典dict上进行删除操作。

    # 递归出口
    if is_end:
        print('Debug: {0}'.format(len(users_pair)))
        return

    # 递归主体
    is_end = True

    print('\n第{0}次迭代，正在删除小度用户...'.format(itecount))
    users = list(users_pair.keys())
    for user in users:  # 注意使用list()来避免iterate，使得del操作得以成功
        if len(users_pair[user]) < ku_threshold:  # 删除小度用户
            del users_pair[user]
    if len(users_pair) < len(users):
        is_end = False
    else:
        pass
    # 转置
    items_pair = defaultdict(set)
    for (k, v) in list(users_pair.items()):
        for value in v:
            items_pair[value].add(k)
        del users_pair[k]
    # 重点，难点。为了保证“相对全局变量”users_pair不丢失，不要给它赋新值。此时它实际上为{}
    items_pair = dict(items_pair)
    gc.collect()

    print('第{0}次迭代，正在删除小度物品...'.format(itecount))
    items = list(items_pair.keys())
    for item in items:  # 注意使用list()来避免iterate，使得del操作得以成功
        if len(items_pair[item]) < ki_threshold:  # 删除小度物品
            del items_pair[item]
    if len(items_pair) < len(items):
        is_end = False
    else:
        pass
    n = len(items_pair)  # 临时统计一下，为下面的print()操作做准备
    # 转置
    for (k, v) in list(items_pair.items()):
        for value in v:
            users_pair.setdefault(value, set())
            users_pair[value].add(k)
        del items_pair[k]
    del items_pair  # 彻底删除items_pair。虽然它此时已经为{}
    gc.collect()

    # 打印数据集基本信息
    m = len(users_pair)
    count_ratings = sum([len(items) for items in users_pair.values()])
    print('用户数：{0}\t物品数： {1}\t评分数：{2}'.format(m, n, count_ratings))

    flt_old2(users_pair, ku_threshold, ki_threshold, is_end, itecount+1)  # 切记itecount++
    print('Debug: {0}'.format(len(users_pair)))
    return
def flt(users_pair, ku_threshold, ki_threshold, is_end=False, itecount=0):
    # 迭代式。

    while not is_end:
        is_end = True

        print('\n第{0}次迭代。正在删除小度用户...'.format(itecount))
        users = list(users_pair.keys())
        for user in users:  # 注意使用list()来避免iterate，使得del操作得以成功
            if len(users_pair[user]) < ku_threshold:  # 删除小度用户
                del users_pair[user]
        if len(users_pair) < len(users):
            is_end = False
        else:
            pass
        # 转置
        items_pair_temp = defaultdict(set)
        for (k, v) in list(users_pair.items()):
            for value in v:
                items_pair_temp[value].add(k)
            del users_pair[k]
        gc.collect()  # 重点，难点。为了保证“相对全局变量”users_pair不丢失，不要给它赋新值。此时它实际上为{}

        print('第{0}次迭代.正在删除小度物品...'.format(itecount))
        items = list(items_pair_temp.keys())
        for item in items:  # 注意使用list()来避免iterate，使得del操作得以成功
            if len(items_pair_temp[item]) < ki_threshold:  # 删除小度物品
                del items_pair_temp[item]
        if len(items_pair_temp) < len(items):
            is_end = False
        else:
            pass
        n = len(items_pair_temp)  # 统计一下，为下面的print()操作做准备
        # 转置
        for (k, v) in list(items_pair_temp.items()):
            for value in v:
                users_pair.setdefault(value, set())
                users_pair[value].add(k)
            del items_pair_temp[k]
        del items_pair_temp  # 彻底删除items_pair_temp。虽然它此时已经为{}
        gc.collect()

        # 打印数据集基本信息
        m = len(users_pair)
        rows = sum([len(items) for items in users_pair.values()])
        print('用户数：{0}\t物品数： {1}\t评分数：{2}'.format(m, n, rows))

        itecount += 1
    return
def ui_transe_old(kv):
    # 形如{u1: {i1, i2, i3}, ...}。
    # // 已淘汰。ui_transe()方法比它快一倍。

    vk = {}
    for (k, v) in kv.items():
        for value in v:
            vk.setdefault(value, set())
            vk[value].add(k)
    return vk
def ui_transe(kv):
    # 形如{u1: {i1, i2, i3}, ...}。

    vk = defaultdict(set)
    for (k, v) in kv.items():
        for value in v:
            vk[value].add(k)
    return dict(vk)


def sample_random(in_file, out_file, users_threshold, seed=47):
    # 从原始数据集中随机抽样出users_threshold个用户，构成一个小子集。

    # 统计原始数据集中的用户度分布
    users_degree = myutil.get_userdegree(in_file)

    # 随机抽样
    random.seed(seed)
    sub_users = random.sample(users_degree.keys(), users_threshold)

    # 保存到外存磁盘
    sub_users_set = set(sub_users)
    with open(out_file, 'w') as out_write:
        with open(in_file, 'r') as in_read:
            count = 0
            for line in in_read:
                # 进度条
                if count % 10000 == 0:
                    sys.stdout.write('\r对原始数据集进行抽样，生成新的数据集：已处理{0}万行'.format(count / 10000))
                    sys.stdout.flush()
                count += 1

                r = line.strip().split(',')
                user = int(r[0])  # 报错说明数据集不规范

                if user in sub_users_set:
                    out_write.write(line)
    print('\n抽样完成')
    return


def convert_jester():
    # Jester数据集。

#    # 从矩阵形式转换成稀疏矩阵的形式
#    import pandas as pd
#    from collections import defaultdict
#    df = pd.read_excel(r'D:\recommender_data\jester\原始存档\jester-data-1.xls', header=None)
#    array = df.values
#    array = array[:, 1:array.shape[1]]  # 去除第一列
#    users = [i + 1 for i in range(array.shape[0])]
#    jokes = [i + 1 for i in range(array.shape[1])]
#    
#    file_explicit = r'D:\recommender_data\jester\jester-explicit'
#    with open(file_explicit, 'w') as out_write:
#        for index_user in range(array.shape[0]):
#            user = users[index_user]
#            for index_joke in range(array.shape[1]):
#                joke = jokes[index_joke]
#                rating = array[index_user, index_joke]
#
#                if rating == float(99):  # boundary condition
#                    continue
#                out_write.write(str(user) + ',' + str(joke) + ',' + str(rating) + '\n')

#    # 隐式化（>=3、>=EUMR、和all）
#    file_explicit = r'D:\recommender_data\jester\jester-explicit'
#    file_implicit = r'D:\recommender_data\jester\implicit'  # 首先要求>0，其次>= (each users' max rating - 2)
#    
#    # 确定每个用户的评分尺度rating scale
#    users_maxrating = {}  # {user1: [max_rating, min_rating]}
#    with open(file_explicit, 'r') as in_read:
#        count = 0
#        for line in in_read:
#            # 进度条
#            if count % 10000 == 0:
#                sys.stdout.write('\r确定每个用户的评分尺度rating scale：已处理{0}万行'.format(count / 10000))
#                sys.stdout.flush()
#            count += 1
#
#            row = line.strip().split(',')
#            user = int(row[0])
#            rating = float(row[2])
#
#            users_maxrating.setdefault(user, -sys.maxsize)
#            if users_maxrating[user] < rating:
#                users_maxrating[user] = rating
#
#    # 隐式化
#    print()
#    with open(file_implicit, 'w') as out_write:
#        in_read = open(file_explicit, 'r')
#        count = 0
#        for line in in_read:
#            # 进度条
#            if count % 10000 == 0:
#                sys.stdout.write('\r隐式化：已处理{0}万行'.format(count / 10000))
#                sys.stdout.flush()
#            count += 1
#
#            row = line.strip().split(',')
#            user = int(row[0])
#            rating = float(row[2])
#
#            # 
#            if rating > 0.0 and rating >= (users_maxrating[user] - 2.0):
#                out_write.write(line)
#        in_read.close()
#
    # 压缩
    file_dataset0 = r'D:\recommender_data\jester\dataset-v0'
    #remove_toobiguser(file_implicit, file_dataset0)  # 过滤前千分之五的过于活跃的用户
    # 压缩
    file_dataset1 = r'D:\recommender_data\jester\dataset-v1'
    remove_smalluser(file_dataset0, file_dataset1, 4)
    
    print_data(file_dataset0)
    print_data(file_dataset1)

    return
def convert_movielens100k():
    # MovieLens-100k。

    # # 隐式化（>=3、>=EUMR、和all）
    # file_explicit = r'D:\recommender_data\movielens100k\explicit'
    # file_implicit = r'D:\recommender_data\movielens100k\implicit'
    # explicit_to_implicit_fivescale(file_explicit, file_implicit)  # >=3
    #
    # # 压缩
    # file_dataset0 = r'D:\recommender_data\movielens100k\dataset-v0'
    # remove_toobiguser(file_implicit, file_dataset0)  # 过滤前千分之五的过于活跃的用户
    # # 压缩
    # file_dataset1 = r'D:\recommender_data\movielens100k\dataset-v1'
    # remove_smalluser(file_dataset0, file_dataset1, 9)

    print_data(r'D:\recommender_data\movielens100k\dataset-v0')
    print_data(r'D:\recommender_data\movielens100k\dataset-v1')
    return
def convert_movielens1m():
    # MovieLens-1m。

    # 隐式化（>=3、>=EUMR、和all）
    file_explicit = r'D:\recommender_data\movielens1m\movielens1m-explicit'
    file_implicit = r'D:\recommender_data\movielens1m\implicit'
    explicit_to_implicit_fivescale(file_explicit, file_implicit)  >=3

    # 压缩
    file_dataset0 = r'D:\recommender_data\movielens1m\dataset-v0'
    remove_toobiguser(file_implicit, file_dataset0)
    # 压缩
    file_dataset1 = r'D:\recommender_data\movielens1m\dataset-v1'
    remove_smalluser(file_dataset0, file_dataset1, 9)
    return
def convert_netflix():
    # Netflix。

    # 隐式化（>=3、>=EUMR、和all）
    file_explicit = r'D:\recommender_data\netflix\netflix-explicit'
    file_implicit = r'D:\recommender_data\netflix\implicit'
    explicit_to_implicit_fivescale(file_explicit, file_implicit)  # >=3

    # 压缩
    file_dataset0 = r'D:\recommender_data\netflix\dataset-v0'
    remove_toobiguser(file_implicit, file_dataset0)
    # 压缩
    file_sampletemp = r'D:\recommender_data\netflix\dataset-sampletemp'
    sample_random(file_dataset0, file_sampletemp, 10000)
    file_dataset1 = r'D:\recommender_data\netflix\dataset-v1'
    remove_smalluser(file_sampletemp, file_dataset1, 9)
    return
def convert_movielens20m():
    # MovieLens-20m。

    # 隐式化（>=3、>=EUMR、和all）
    file_explicit = r'D:\recommender_data\movielens20m\movielens20m-explicit'
    file_implicit = r'D:\recommender_data\movielens20m\implicit'
    explicit_to_implicit_fivescale(file_explicit, file_implicit)  # >=3

    # 压缩
    file_dataset0 = r'D:\recommender_data\movielens20m\dataset-v0'
    remove_toobiguser(file_implicit, file_dataset0)
    # 压缩
    file_sampletemp = r'D:\recommender_data\movielens20m\dataset-sampletemp'
    sample_random(file_dataset0, file_sampletemp, 10000)
    file_dataset1 = r'D:\recommender_data\movielens20m\dataset-v1'
    remove_smalluser(file_sampletemp, file_dataset1, 9)
    return


def convert_bookcrossing():
    # BookCrossing。

    # 隐式化（>=3、>=EUMR、和all）
    file_explicit = r'D:\recommender_data\bookcrossing\bookcrossing-explicit'
    file_implicit = r'D:\recommender_data\bookcrossing\implicit'
    explicit_to_implicit_bookcrossing(file_explicit, file_implicit)

    # 压缩
    file_dataset0 = r'D:\recommender_data\bookcrossing\dataset-v0'
    remove_toobiguser(file_implicit, file_dataset0)

    # 压缩
    file_dataset1 = r'D:\recommender_data\bookcrossing\dataset-v1'
    remove_smalluser(file_dataset0, file_dataset1, 9)
    return
def convert_lastfm():
    # Last.fm。

    #in_file = r'D:\recommender_data\lastfm\usersha1-artmbid-artname-plays-v2'
    #out_file_explicit = r'D:\recommender_data\lastfm\explicit'
    ## 为了简单起见，最后我们只选择有item_MBID的行作为有效数据，也就是说删除掉了item_noMBID对应的部分。这样做是合理的，因为99%评分其实都是item_MBID类型而不是item_noMBID类型的
    ## id标准化（把字符串id映射成整数id）
    #sep = '\t'
    #users_id = {}
    #items_id = {}
    #
    #with open(out_file_explicit, 'w') as out_write:
    #    import codecs
    #    with codecs.open(in_file, "r",encoding='utf-8', errors='ignore') as in_read:
    #            count_valid = 0
    #            count = 0
    #            for line in in_read:
    #                # 进度条
    #                if count % 10000 == 0:
    #                     sys.stdout.write('\r读入原始数据集：已处理{0}万行'.format(count / 10000))
    #                     sys.stdout.flush()
    #                count += 1
    #
    #                row = line.strip().split(sep)
    #
    #                if len(row) == 4:
    #                    user_name = row[0]
    #                    item_MBID = row[1]
    #                    item_noMBID = row[2]
    #                    playcounts = row[3]
    #                    if not playcounts.isdigit():  # 很少
    #                        continue
    #                    if len(user_name) != 40:  # 很少
    #                        continue
    #                    if len(item_MBID) != 36:  # 大部分是此情况。即Artist是without MBID类型的
    #                        continue
    #
    #                    # 编号（标准化）。从1开始，与标准数据集（例如Movielens）保持一致
    #                    if user_name not in users_id:
    #                        users_id[user_name] = len(users_id) + 1
    #
    #                    # 编号（标准化）。从1开始，与标准数据集（例如Movielens）保持一致
    #                    if item_MBID not in items_id:
    #                        items_id[item_MBID] = len(items_id) + 1
    #
    #                    user_id = users_id[user_name]
    #                    item_id = items_id[item_MBID]
    #
    #                    r = (str(user_id), str(item_id), playcounts)
    #                    out_write.write(','.join(r))
    #                    out_write.write('\n')
    #
    #                    count_valid += 1
    #
    #print('\n无效行共', count-count_valid)

    # # 隐式化
    # file_explicit = r'D:\recommender_data\lastfm\explicit'
    # file_implicit = r'D:\recommender_data\lastfm\implicit'
    # explicit_to_implicit_fivescale(file_explicit, file_implicit, threshold=0)  # 借用movielens的方法。规定playcounts>=2的记录为positive feedback。但是因为2太小可能会有噪声数据
    #
    # # 压缩
    # #...。不过滤过于活跃的用户。因为Last.fm数据集中用户度分布的很均匀而且很接近，其实不存在过于活跃的用户
    # file_dataset00 = r'D:\recommender_data\lastfm\dataset-v00'
    #
    # # 压缩
    # file_sampletemp = r'D:\recommender_data\lastfm\dataset-sampletemp'
    # file_dataset2 = r'D:\recommender_data\lastfm\dataset-v2'
    # sample_random(file_dataset00, file_sampletemp, users_threshold=10000)
    # remove_smalluser(file_sampletemp, file_dataset2, 9)

    # Debug
    print_data(r'D:\recommender_data\lastfm\dataset-sampletemp')
    print_data(r'D:\recommender_data\lastfm\dataset-v2')

    return
def convert_msd():
    # Million Song Dataset的子集。Taste Profile Subset。
    
#    in_file = r'D:\recommender_data\msd\train_triplets'
#    out_file = r'D:\recommender_data\msd\explicit'
#    
#    # id标准化（把字符串id映射成整数id）
#    sep = '\t'
#    users_id = {}
#    items_id = {}
#    
#    with open(out_file, 'w') as out_write:
#        import codecs
#        with codecs.open(in_file, "r",encoding='utf-8', errors='ignore') as in_read:
#                count_valid = 0
#                count = 0
#                for line in in_read:
#                    # 进度条
#                    if count % 10000 == 0:
#                         sys.stdout.write('\r读入原始数据集：已处理{0}万行'.format(count / 10000))
#                         sys.stdout.flush()
#                    count += 1
#    
#                    row = line.strip().split(sep)
#    
#                    if len(row) == 3:
#                        user_name = row[0]
#                        song_name = row[1]
#                        playcounts = row[2]
#                        # 编号（标准化）。从1开始，与标准数据集（例如Movielens）保持一致
#                        if user_name not in users_id:
#                            users_id[user_name] = len(users_id) + 1
#    
#                        # 编号（标准化）。从1开始，与标准数据集（例如Movielens）保持一致
#                        if song_name not in items_id:
#                            items_id[song_name] = len(items_id) + 1
#    
#                        user_id = users_id[user_name]
#                        item_id = items_id[song_name]
#    
#                        r = (str(user_id), str(item_id), playcounts)
#                        out_write.write(','.join(r))
#                        out_write.write('\n')
#    
#                        count_valid += 1
#    
#    print('\n无效行共', count-count_valid)
#    print_data(out_file)
    
    # 隐式化。all
    file_explicit = r'D:\recommender_data\msd\explicit'
    file_implicit = r'D:\recommender_data\msd\implicit'
    # explicit_to_implicit_fivescale(file_explicit, file_implicit, threshold=0)  # 借用movielens的方法。规定所有记录都为positive feedback。
    
    # 压缩
    file_dataset0 = r'D:\recommender_data\msd\dataset-v0'
    remove_toobiguser(file_implicit, file_dataset0, topvalue=0.0005)  # 因为用户度分布的比较均匀，所以我们只剔除前0.05%流行度的过于活跃的用户
    
    # 压缩
    file_sampletemp = r'D:\recommender_data\msd\dataset-sampletemp'
    file_dataset1 = r'D:\recommender_data\msd\dataset-v1'
    sample_random(file_dataset0, file_sampletemp, users_threshold=10000)
    remove_smalluser(file_sampletemp, file_dataset1, 9)
    return
def convert_amazonbook():
    # Amazon-Book数据集。

    # id标准化，保存explicit file
#    in_file = r'D:\recommender_data\amazonbook\ratings_Books_afterPreprocessing.csv'
#    file_explicit = r'D:\recommender_data\amazonbook\explicit'
#    sep = '::'
#    users_id = {}
#    items_id = {}
#    with open(file_explicit, 'w') as out_write:
#        import codecs
#        with codecs.open(in_file, 'r', encoding='utf-8', errors='ignore') as in_read:
#            count_valid = 0
#            count = 0
#            for line in in_read:
#                # 进度条
#                if count % 10000 == 0:
#                     sys.stdout.write('\r读入原始数据集：已处理{0}万行'.format(count / 10000))
#                     sys.stdout.flush()
#                count += 1
#
#                row = line.strip().split(sep)
#
#                user_name = row[0]
#                song_name = row[1]
#                rating = row[2]
#                
#                # 编号（标准化）。从1开始，与标准数据集（例如Movielens）保持一致
#                if user_name not in users_id:
#                    users_id[user_name] = len(users_id) + 1
#
#                # 编号（标准化）。从1开始，与标准数据集（例如Movielens）保持一致
#                if song_name not in items_id:
#                    items_id[song_name] = len(items_id) + 1
#
#                user_id = users_id[user_name]
#                item_id = items_id[song_name]
#
#                r = (str(user_id), str(item_id), rating)
#                out_write.write(','.join(r))
#                out_write.write('\n')
#
#                count_valid += 1
#    
#    print('\n无效行共', count-count_valid)
#    print_data(file_explicit)
#    return

    # 隐式化。
    file_explicit = r'D:\recommender_data\amazonbook\explicit'
    file_implicit = r'D:\recommender_data\amazonbook\implicit'
    explicit_to_implicit_fivescale(file_explicit, file_implicit, threshold=3.0)

    # 压缩
    file_dataset00 = r'D:\recommender_data\amazonbook\dataset-v00'
    remove_toobiguser(file_implicit, file_dataset00, topvalue=0.0005)  # 我们只剔除前0.01%流行度(而不是0.5%)的过于活跃的用户
    # 进一步压缩
    file_dataset0 = r'D:\recommender_data\amazonbook\dataset-v0'
    remove_smalluser(file_dataset00, file_dataset0, 4)

    # 压缩
    file_dataset1 = r'D:\recommender_data\amazonbook\dataset-v1'
    sample_random(file_dataset0, file_dataset1, users_threshold=30000)  # 抽样3万用户
    return
    

if __name__ == '__main__':
    print_data(r'D:\recommender_data\msd\msd_out\testsmall-v1')
