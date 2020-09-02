import numpy as np
import scipy.stats as ss  # 统计包
import scipy.linalg  as sl  # 线性代数包
import scipy.optimize as so
from matplotlib import pyplot as plt

# 统计
s1 = ss.uniform.rvs(size=10)  # （0,1）分布的正态分布数
print(s1)
s2 = ss.norm.rvs(1, 1, size=10)
s3 = ss.poisson.rvs(2, size=10)
print('1,1分布的正态函数：', s2)  # (1,1)分布的正态分布
print('k值为2的泊松分布', s3)  # 泊松分布
print('t检验的t值和p值分别为：', ss.ttest_ind(s2, s3))  # T检验 两个正态分布数

# 线代

# 求行列式
a = np.mat('1,0,3;3,6,7;2,3,2')
b = np.mat('4; 16; 7')
l = sl.det(a)
print(f'行列式{a}的值为{l}')

# 求矩阵的逆
l2 = sl.inv(a)
print(f'行列式{a}的逆矩阵为{l2}')
print('检验正确性：', np.dot(a, l2))

# 解矩阵的解
l2 = sl.solve(a, b)
print(f'求解线性方程\n{a}\n={b}\n的解:x1={l2[0]},x2={l2[1]},x3={np.round(l2[2])}')

# 求矩阵的特征值和特征向量
l3 = sl.eig(a)
print(f'特征值和特征向量：{l3}')
l4 = sl.eigvals(a)
print(f'特征值：{l4}')


# 优化optimize(提供求函数的最小值、求函数的根、曲线拟合)
def f(x):
    return x ** 2 + 10 * np.sin(x)


x = np.arange(-5, 5, 0.2)
plt.plot(x, f(x), 'g-')
plt.show()

x1_y1 = so.minimize(f, 0)
x2_y2 = so.minimize(f, 4)
print(so.fmin(f, 0))  # 求全局极小值点

print('minimize:', x1_y1)  # 求全局极小值点
print('minimize2:', x2_y2)  # 求局部极小值点

print(f'x1:{x1_y1["x"]},y1:{x1_y1["fun"]}')  # 求全局极小值及极小值点
print(f'x2:{x2_y2["x"]},y2:{x2_y2["fun"]}')  # 求局部极小值及极小值点
