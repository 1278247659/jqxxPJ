# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np

x = np.arange(-2 * np.pi, 2 * np.pi, 0.25)
y = np.sin(x)

# 折线图
# plt.plot(x, y, '-.')

# 散点图
# plt.scatter(x, y, color='r', marker='.')
# plt.title('SCATTER PIC')

# 饼图
a = np.random.randint(0, 10, 4)
# labers = ['1', '2', '3', '4']
# plt.pie(a, labels=labers)
# plt.axis('equal') #将x和y重合 （这样才能得到饼图）
# plt.legend()
# plt.title('PIE PIC')

# 柱状图
plt.xlim(-1, 4)
plt.ylim(0, 10)
plt.xlabel('goods')
plt.ylabel('price')
plt.title('goods & price')
plt.bar(range(4), a, color='g')
plt.show()
