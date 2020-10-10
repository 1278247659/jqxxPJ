import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix


file = pd.read_csv(r'D:\aa罗\data\data\3_film.csv')
content = pd.DataFrame(file)
# print(content.head())
# 画直方图
# for i in content:
# plt.hist(content[i])
# plt.show()

print(sns.boxplot(data=content))
# 画密度图
file.plot(kind='density', subplots=True, layout=(2, 2), sharex=False, fontsize=8, figsize=(12, 7))
# subplots 制作多个子图 layout 子图数量
# sharex 共享x轴
plt.show()

# 画箱线图
file.boxplot(figsize=(12, 7))
# subplots 制作多个子图 layout 子图数量
# sharex 共享x轴
plt.show()

# 画热力图
name = ['filmnum', 'filmsize', 'ratio', 'quality']  # 设置变量名
corr = file.corr()  # 计算变量间的相关矩阵

# 绘制相关热力图
fig = plt.figure()  # 创建一个绘图对象
ax = fig.add_subplot(111)  # 调用画板绘制第一个子图
cax = ax.matshow(corr, vmin=0.3, vmax=1)  # 绘制热力图 颜色深浅由0.3-1
fig.colorbar(cax)  # 设置为颜色渐变
ticks = np.arange(0, 4, 1)
ax.set_xticks(ticks)  # 生成刻度
ax.set_yticks(ticks)
ax.set_xticklabels(name)  # 生成标签
ax.set_yticklabels(name)
plt.title('hot pic')
plt.show()

# 绘制散点图矩阵
scatter_matrix(file,figsize=(8,8),c='b')
plt.show()