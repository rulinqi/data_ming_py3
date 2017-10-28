# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np

def run_main():
    """

    :return:
    """
    print('====== pandas examples ======')
    pandas_data_structures()
    pandas_data_process()
    pandas_stats()
    pandas_multi_indx()
    pandas_groupby()
    pandas_grouped_apply_transform()

def pandas_data_structures():
    """
        Pandas数据结构
    :return:
    """
    # 1. Series
    # 通过list构建Series
    ser_obj = pd.Series(range(10,20))
    print(type(ser_obj))
    # 获取数据
    print(ser_obj.values)
    # 获取索引
    print(ser_obj.index)
    # 预览数据
    print(ser_obj.head(3))
    # 通过索引获取数据
    print(ser_obj[3])
    # 索引与数据的对应关系仍保持在数组运算的结果中
    print(ser_obj * 3)
    print(ser_obj > 13)
    # 通过dict构建Series
    year_data = {2000:23.1,2001:34,2003:78}
    ser_obj2 = pd.Series(year_data)
    print(ser_obj2.head())
    print(ser_obj2.index)
    # name属性
    ser_obj2.name = 'temp'
    ser_obj2.index.name = 'year'
    print(ser_obj2.head())

    # 2.DataFrame
    # 通过ndarray构建DataFrame
    array = np.random.randn(5,4)
    print(array)
    df_obj = pd.DataFrame(array)
    print(df_obj)
    # 通过dict构建DataFrame
    dict_data = {'A': 1.,
                 'B': pd.Timestamp('20161217'),
                 'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                 'D': np.array([3] * 4, dtype='int32'),
                 'E': pd.Categorical(["Python", "Java", "C++", "C#"]),
                 'F': 'ChinaHadoop'}
    df_obj2 = pd.DataFrame(dict_data)
    print(df_obj2.head())
    # 通过列索引获取列数据
    print(df_obj2['A'])
    print(type(df_obj2['A']))
    print(df_obj2.A)
    # 增加列
    df_obj2['G'] = df_obj2['D'] + 4
    print(df_obj2.head())
    # 删除列
    del df_obj2['G']
    print(df_obj2.head())

    # 3.索引对象 Index
    print(type(df_obj.index))
    print(type(df_obj2.index))
    print(df_obj2.index)
    # 索引对象不可变
    df_obj2.index[0] = 2

def pandas_data_process():
    """
    Pandas数据操作
    :return:
    """
    # 1.Series索引
    ser_obj = pd.Series(range(5), index=['a', 'b', 'c', 'd', 'e'])
    print(ser_obj.head())
    # 行索引
    print(ser_obj['a'])
    print(ser_obj[0])
    # 切片索引
    print(ser_obj[1:3]) #位置切片不包末尾
    print(ser_obj['b':'d'])
    # 不连续索引
    print(ser_obj[[0,2,4]])
    print(ser_obj[['a', 'e']])
    # 布尔索引
    ser_bool = ser_obj > 2
    print(ser_bool)
    print(ser_obj[ser_bool])
    print(ser_obj[ser_obj > 2])

    # 2.DataFrame索引
    df_obj = pd.DataFrame(np.random.randn(5, 4), columns=['a', 'b', 'c', 'd'])
    print(df_obj.head())
    # 列索引
    print('列索引')
    print(df_obj['a'])  # 返回Series类型
    # print(df_obj[[0,1]])   # python 中不可用
    # print(type(df_obj[[0]])) # 返回DataFrame类型
    # 不连续索引
    print('不连续索引')
    print(df_obj[['a', 'c']])
    # print(df_obj[[1, 3]])

    # 3.三种索引方式
    # 标签索引 loc
    # Series
    print(ser_obj['b':'d'])
    print(ser_obj.loc['b':'d'])
    # DataFrame
    print(df_obj['a'])
    print(df_obj.loc[0:2, 'a'])

    # 整型位置索引 iloc
    print(ser_obj[1:3])
    print(ser_obj.iloc[1:3])
    # DataFrame
    print(df_obj.iloc[0:2, 0])  # 注意和df_obj.loc[0:2, 'a']的区别  == 0:2 显示0,1行，而标签索引显示0,1,2行

    # 混合索引 ix
    print(ser_obj.ix[1:3])
    print(ser_obj.ix['b':'c'])
    # DataFrame
    print(df_obj)
    print(df_obj.ix[0:2, 1])  # 先按标签索引尝试操作，然后再按位置索引尝试操作
    print(df_obj.iloc[0:2, 1])
    print(df_obj.loc[0:2, 'b'])

    # 4.运算与对齐
    s1 = pd.Series(range(10, 20), index=range(10))
    s2 = pd.Series(range(20, 25), index=range(5))
    print('s1: ')
    print(s1)
    print('')
    print('s2: ')
    print(s2)

    # Series 对齐运算
    s1 + s2  #少的行用nan 补，结果同样

    df1 = pd.DataFrame(np.ones((2, 2)), columns=['a', 'b'])
    df2 = pd.DataFrame(np.ones((3, 3)), columns=['a', 'b', 'c'])
    test = pd.DataFrame(np.zeros((3,2)))
    print(test)
    print('df1: ')
    print(df1)
    print('')
    print('df2: ')
    print(df2)
    # DataFrame对齐操作
    df1 + df2 #少的用nan 补

    # 填充未对齐的数据进行运算
    print(s1)
    print(s2)
    s1.add(s2, fill_value=-1)
    print(df1)
    print(df2)
    print(df1.sub(df2, fill_value=2.))
    print(df2.sub(df1, fill_value=2.))

    # 填充NaN
    s3 = s1 + s2
    print(s3)
    s3_filled = s3.fillna(-1)
    print(s3_filled)
    df3 = df1 + df2
    print(df3)
    df3.fillna(100, inplace=True)
    print(df3)

    # 5.函数应用
    # Numpy ufunc 函数
    df = pd.DataFrame(np.random.randn(5, 4) - 1)
    print(df)
    print(np.abs(df)) # 绝对值

    # 使用apply应用行或列数据
    # f = lambda x : x.max()
    print(df.append(lambda x : x.max()))
    # 指定轴方向 --行
    print(df.apply(lambda x: x.max(), axis=1))

    # 使用applymap应用到每个数据
    f2 = lambda x: '%.2f' % x
    print(df.applymap(f2))

    # 6.排序
    s4 = pd.Series(range(10, 15), index=np.random.randint(5, size=5))
    print(s4)
    # 索引排序
    s4.sort_index()
    df4 = pd.DataFrame(np.random.randn(3, 4),
                       index=np.array([3,1,2]),
                       columns=np.array([9,5,2]))
    print(df4)
    df4.sort_index(axis=1)
    # 按值排序
    df4.sort_values(by=1, axis=1)

    # 7.处理缺失数据
    df_data = pd.DataFrame([np.random.randn(3), [1., np.nan, np.nan],
                            [4., np.nan, np.nan], [1., np.nan, 2.]])
    df_data.head()
    # isnull
    df_data.isnull()
    # dropna
    df_data.dropna()
    # df_data.dropna(axis=1)
    # fillna
    df_data.fillna(-100.)

def pandas_stats():
    """
    Pandas统计计算和描述
    :return:
    """
    # 1.常用的统计计算
    df_obj = pd.DataFrame(np.random.randn(5, 4), columns=['a', 'b', 'c', 'd'])
    print(df_obj)
    df_obj.sum()
    df_obj.max()
    df_obj.min(axis=1)

    # 2.统计描述
    df_obj.describe()

def pandas_multi_indx():
    """
    Pandas层级索引
    :return:
    """
    ser_obj = pd.Series(np.random.randn(12),
                        index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd'],
                               [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]])
    print(ser_obj)

    # 1.MultiIndex索引对象
    print(type(ser_obj.index))
    print(ser_obj.index)

    # 2.选取子集
    # 外层选取
    print(ser_obj['c'])
    # 内层选取
    print(ser_obj[:, 2])

    # 3.交换分层顺序
    print(ser_obj.swaplevel())

    # 4.交换并排序分层
    print(ser_obj.swaplevel().sortlevel())

def pandas_groupby():
    """
    分组与聚合
    :return:
    """
    # 1.GroupBy对象
    dict_obj = {'key1': ['a', 'b', 'a', 'b',
                         'a', 'b', 'a', 'a'],
                'key2': ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                'data1': np.random.randn(8),
                'data2': np.random.randn(8)}
    df_obj = pd.DataFrame(dict_obj)
    print(df_obj)
    # dataframe根据key1进行分组
    print(type(df_obj.groupby('key1')))
    # data1列根据key1进行分组
    print(type(df_obj['data1'].groupby(df_obj['key1'])))

    # 分组运算
    grouped1 = df_obj.groupby('key1')
    print(grouped1.mean())
    grouped2 = df_obj['data1'].groupby(df_obj['key1'])
    print(grouped2.mean())
    # size
    print(grouped1.size())
    print(grouped2.size())
    # 按列名分组
    df_obj.groupby('key1')
    # 按自定义key分组，列表
    self_def_key = [1, 1, 2, 2, 2, 1, 1, 1]
    df_obj.groupby(self_def_key).size()
    # 按自定义key分组，多层列表
    df_obj.groupby([df_obj['key1'], df_obj['key2']]).size()
    # 按多个列多层分组
    grouped2 = df_obj.groupby(['key1', 'key2'])
    print(grouped2.size())
    # 多层分组按key的顺序进行,对分组结果进行展开
    grouped3 = df_obj.groupby(['key2', 'key1'])
    print(grouped3.mean())
    print()
    print(grouped3.mean().unstack())

    # 2.GroupBy对象分组迭代
    # 单层分组
    for group_name, group_data in grouped1:
        print(group_name)
        print(group_data)
    # 多层分组
    for group_name, group_data in grouped2:
        print(group_name)
        print(group_data)
    # GroupBy对象转换list
    listg = list(grouped1)
    print(listg)
    print(listg[0][1])
    # GroupBy对象转换dict
    dict(list(grouped1))
    # 按数据类型分组
    print(df_obj)
    print(df_obj.dtypes)
    print(df_obj.groupby(df_obj.dtypes, axis=1).size())
    print(df_obj.groupby(df_obj.dtypes, axis=1).sum())

    # 3.其他分组方法
    df_obj2 = pd.DataFrame(np.random.randint(1, 10, (5, 5)),
                           columns=['a', 'b', 'c', 'd', 'e'],
                           index=['A', 'B', 'C', 'D', 'E'])
    df_obj2.ix[1, 1:4] = np.NaN
    print(df_obj2)
    # 通过字典分组
    mapping_dict = {'a': 'python', 'b': 'python', 'c': 'java', 'd': 'C', 'e': 'java'}
    print(df_obj2)
    print(df_obj2.groupby(mapping_dict, axis=1).size())
    print(df_obj2.groupby(mapping_dict, axis=1).count())  # 非NaN的个数
    print(df_obj2.groupby(mapping_dict, axis=1).sum())

    # 通过函数分组
    df_obj3 = pd.DataFrame(np.random.randint(1, 10, (5, 5)),
                           columns=['a', 'b', 'c', 'd', 'e'],
                           index=['AA', 'BBB', 'CC', 'D', 'EE'])

    # df_obj3

    def group_key(idx):
        """
            idx 为列索引或行索引
        """
        # return idx
        return len(idx)

    df_obj3.groupby(group_key).size()
    # 以上自定义函数等价于
    # df_obj3.groupby(len).size()

    # 通过索引级别分组
    columns = pd.MultiIndex.from_arrays([['Python', 'Java', 'Python', 'Java', 'Python'],
                                         ['A', 'A', 'B', 'C', 'B']], names=['language', 'index'])
    df_obj4 = pd.DataFrame(np.random.randint(1, 10, (5, 5)), columns=columns)
    print(df_obj4)
    # 根据language进行分组
    print(df_obj4.groupby(level='language', axis=1).sum())
    print(df_obj4.groupby(level='index', axis=1).sum())

    # 4.聚合
    dict_obj = {'key1': ['a', 'b', 'a', 'b',
                         'a', 'b', 'a', 'a'],
                'key2': ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                'data1': np.random.randint(1, 10, 8),
                'data2': np.random.randint(1, 10, 8)}
    df_obj5 = pd.DataFrame(dict_obj)
    print(df_obj5)
    # 内置的聚合函数
    print(df_obj5.groupby('key1').sum())
    print(df_obj5.groupby('key1').max())
    print(df_obj5.groupby('key1').min())
    print(df_obj5.groupby('key1').mean())
    print(df_obj5.groupby('key1').size())
    print(df_obj5.groupby('key1').count())
    print(df_obj5.groupby('key1').describe())

    # 自定义聚合函数
    def peak_range(df):
        """
            返回数值范围
        """
        # print type(df) #参数为索引所对应的记录
        return df.max() - df.min()

    print(df_obj5.groupby('key1').agg(peak_range))
    print(df_obj5.groupby('key1').agg(lambda df: df.max() - df.min()))

    # 应用多个聚合函数
    # 同时应用多个聚合函数
    print(df_obj.groupby('key1').agg(['mean', 'std', 'count', peak_range]))  # 默认列名为函数名
    print(df_obj.groupby('key1').agg(['mean', 'std', 'count', ('range', peak_range)]))  # 通过元组提供新的列名
    # 每列作用不同的聚合函数
    dict_mapping = {'data1': 'mean',
                    'data2': 'sum'}
    print(df_obj.groupby('key1').agg(dict_mapping))
    dict_mapping = {'data1': ['mean', 'max'],
                    'data2': 'sum'}
    print(df_obj.groupby('key1').agg(dict_mapping))

def pandas_grouped_apply_transform():
    """
    数据分组运算
    :return:
    """
    # 分组运算后保持shape
    dict_obj = {'key1': ['a', 'b', 'a', 'b',
                         'a', 'b', 'a', 'a'],
                'key2': ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                'data1': np.random.randint(1, 10, 8),
                'data2': np.random.randint(1, 10, 8)}
    df_obj = pd.DataFrame(dict_obj)
    print(df_obj)
    # 按key1分组后，计算data1，data2的统计信息并附加到原始表格中
    k1_sum = df_obj.groupby('key1').sum().add_prefix('sum_') #添加前缀
    print(k1_sum)
    # 方法1，使用merge
    pd.merge(df_obj, k1_sum, left_on='key1', right_index=True)
    # 方法2，使用transform
    k1_sum_tf = df_obj.groupby('key1').transform(np.sum).add_prefix('sum_')
    print(k1_sum_tf)
    df_obj[k1_sum_tf.columns] = k1_sum_tf
    print(df_obj)

    # 自定义函数传入transform
    def diff_mean(s):
        """
            返回数据与均值的差值
        """
        return s - s.mean()

    df_obj.groupby('key1').transform(diff_mean)

    dataset_path = './starcraft.csv'
    df_data = pd.read_csv(dataset_path, usecols=['LeagueIndex', 'Age', 'HoursPerWeek',
                                                 'TotalHours', 'APM'])
    print(df_data.head())
    # applay
    def top_n(df, n=3, column='APM'):
        """
            返回每个分组按 column 的 top n 数据
        """
        return df.sort_values(by=column, ascending=False)[:n]

    df_data.groupby('LeagueIndex').apply(top_n)
    # apply函数接收的参数会传入自定义的函数中
    df_data.groupby('LeagueIndex').apply(top_n, n=2, column='Age')

    # 禁止分组 group_keys=False  结果的呈现将不按分组进行，直接排列
    df_data.groupby('LeagueIndex', group_keys=False).apply(top_n)

def data_merge():
    """
    数据连接 merge
    :return:
    """
    df_obj1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                            'data1': np.random.randint(0, 10, 7)})
    df_obj2 = pd.DataFrame({'key': ['a', 'b', 'd'],
                            'data2': np.random.randint(0, 10, 3)})

    print(df_obj1)
    print(df_obj2)
    # 默认将重叠列的列名作为“外键”进行连接
    pd.merge(df_obj1, df_obj2)
    # on显示指定“外键”
    pd.merge(df_obj1, df_obj2, on='key')

    # left_on，right_on分别指定左侧数据和右侧数据的“外键”
    # 更改列名
    df_obj1 = df_obj1.rename(columns={'key': 'key1'})
    df_obj2 = df_obj2.rename(columns={'key': 'key2'})
    print(df_obj1)
    print(df_obj2)
    pd.merge(df_obj1, df_obj2, left_on='key1', right_on='key2')
    # “外连接”
    pd.merge(df_obj1, df_obj2, left_on='key1', right_on='key2', how='outer')
    # 左连接
    pd.merge(df_obj1, df_obj2, left_on='key1', right_on='key2', how='left')
    # 右连接
    pd.merge(df_obj1, df_obj2, left_on='key1', right_on='key2', how='right')

    # 处理重复列名
    df_obj1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                            'data': np.random.randint(0, 10, 7)})
    df_obj2 = pd.DataFrame({'key': ['a', 'b', 'd'],
                            'data': np.random.randint(0, 10, 3)})
    print(df_obj1)
    print(df_obj2)
    pd.merge(df_obj1, df_obj2, on='key', suffixes=('_left', '_right'))

    # 按索引连接
    df_obj1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                            'data1': np.random.randint(0, 10, 7)})
    df_obj2 = pd.DataFrame({'data2': np.random.randint(0, 10, 3)}, index=['a', 'b', 'd'])
    print(df_obj1)
    print(df_obj2)
    pd.merge(df_obj1, df_obj2, left_on='key', right_index=True)

def data_concat():
    """
    数据合并 concat
    :return:
    """
    # 1.NumPy的concat
    arr1 = np.random.randint(0, 10, (3, 4))
    arr2 = np.random.randint(0, 10, (3, 4))
    print(arr1)
    print(arr2)
    np.concatenate([arr1, arr2])
    np.concatenate([arr1, arr2], axis=1)

    # 2.Series上的concat
    # index 没有重复的情况
    ser_obj1 = pd.Series(np.random.randint(0, 10, 5), index=range(0, 5))
    ser_obj2 = pd.Series(np.random.randint(0, 10, 4), index=range(5, 9))
    ser_obj3 = pd.Series(np.random.randint(0, 10, 3), index=range(9, 12))
    print(ser_obj1)
    print(ser_obj2)
    print(ser_obj3)
    pd.concat([ser_obj1, ser_obj2, ser_obj3])
    pd.concat([ser_obj1, ser_obj2, ser_obj3],axis=1)

    # index 有重复的情况
    ser_obj1 = pd.Series(np.random.randint(0, 10, 5), index=range(5))
    ser_obj2 = pd.Series(np.random.randint(0, 10, 4), index=range(4))
    ser_obj3 = pd.Series(np.random.randint(0, 10, 3), index=range(3))
    print(ser_obj1)
    print(ser_obj2)
    print(ser_obj3)
    pd.concat([ser_obj1, ser_obj2, ser_obj3])
    pd.concat([ser_obj1, ser_obj2, ser_obj3], axis=1, join='inner')

    # DataFrame上的concat
    df_obj1 = pd.DataFrame(np.random.randint(0, 10, (3, 2)), index=['a', 'b', 'c'],
                           columns=['A', 'B'])
    df_obj2 = pd.DataFrame(np.random.randint(0, 10, (2, 2)), index=['a', 'b'],
                           columns=['C', 'D'])
    print(df_obj1)
    print(df_obj2)
    pd.concat([df_obj1, df_obj2])
    pd.concat([df_obj1, df_obj2], axis=1)

def data_reshape():
    """
    数据重构
    :return:
    """
    # stack
    df_obj = pd.DataFrame(np.random.randint(0, 10, (5, 2)), columns=['data1', 'data2'])
    print(df_obj)
    stacked = df_obj.stack()
    print(stacked)
    print(type(stacked))
    print(type(stacked.index))
    # 默认操作内层索引
    stacked.unstack()
    # 通过level指定操作索引的级别
    stacked.unstack(level=0)

def data_replace():
    """
    数据转换
    :return:
    """
    df_obj = pd.DataFrame({'data1': ['a'] * 4 + ['b'] * 4,
                           'data2': np.random.randint(0, 4, 8)})
    print(df_obj)
    df_obj.duplicated()
    df_obj.drop_duplicates()
    df_obj.drop_duplicates('data2')

    # map函数
    ser_obj = pd.Series(np.random.randint(0, 10, 10))
    print(ser_obj)
    ser_obj.map(lambda x: x ** 2)

    # 数据替换replace
    # 替换单个值
    ser_obj.replace(0, -100)
    # 替换多个值
    ser_obj.replace([0, 2], -100)
    # 替换多个值
    ser_obj.replace([0, 2], [-100, -200])

if __name__ == '__main__':
    run_main()