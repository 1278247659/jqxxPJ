import numpy as np

# 数组
a1 = np.array([1, 2, 3, 4])
print(a1)  # 一维数组

a2 = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [5, 6, 7, 8]])
print(a2)  # 二维数组

# 查看数组属性
print(a1.shape)
print(a1.size)
print(a1.dtype)

# 创建数组的形式
print(np.array(10))  # 创建一个10位数的一维数组
print(np.linspace(1, 10, 5))  # 在1-10之间生成5个等距离的一维数组
print(np.zeros((3, 3)))  # 生成3*3的0元素组成的数组
print(np.eye(3))  # 生成一个三阶单位矩阵
print(np.random.random(10))  # 随机生成十个数组成的一维数组

# 基础数组运算
print(a1 * 2)
a1_1 = np.array(range(4))
print(a1 + a1_1)

# 一维数组切片/排序
print(a1[0:2])
print(a1[-1])

#多维数组切片 从（0,0）开始
print(a2[-1])
print(a2[0:2])
print(a2[1, 3])

#数组统计函数
print('排序：',np.sort(a1))
print('均值:',np.mean(a1))
print('方差:',np.std(a1))
print('求和:',np.sum(a1))

#创建矩阵
a2_2=np.mat("1,3,4,5;12,3,5,5;1,3,5,7;12,46,78,9")
print('构造的4*4矩阵是:',a2_2)
a3 = np.dot(a1, a2_2)  # 矩阵的乘积
print('1*4矩阵和4*4矩阵想乘的结果：',a3)
