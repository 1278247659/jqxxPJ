import pandas as pd
import numpy as np

# Series(序列)
p1 = pd.Series(range(1, 20, 3))  # 生成带索引的10个数据 可通过下标访问数值
print(p1)
print(p1[3])

p2_2 = {'a': 1, 'a2': 10, 'a3': 100}  # 字典的key做索引
p2 = pd.Series(p2_2)
print(p2)
print(p2['a2'])

# DataFrame 通过直接读取文件即可
p3_3 = {'a': [1, 2, 3, 4, 5], 'a2': [10, 11, 12, 13, 14], 'a3': [100, 101, 103, 105, 104]}  # 生成一个二维矩阵图
p3 = pd.DataFrame(p3_3)
print(p3)

# 文件操作
p_f = pd.read_csv('./data/data/2_apple.csv')
print('输出前5行：', p_f.head())
print('输出后5行：', p_f.tail())
print('索引：', p_f.index)
print('查看列：', p_f.columns)
print('统计分析：\n', p_f.describe())

# 数据的排序
print('按苹果价格排序：', p_f.sort_values(by='apple'))

# 转置
print(p_f.T)

# 选择数据
print('根据列名选择:', p_f['year'])  # 根据列名选择
print('根据索引选择:', p_f[0:2])  # 根据索引选择
print('根据行号选择:', p_f.iloc[0:2, 1:3])  # 根据行号选择

# 缺失值处理
p3['b'] = pd.Series([1, 2, 3, 4])
print(p3)
print(p3.dropna(how='any')) #会剔除第5行缺失数据的那行 参数'any':只要存在就去掉=
