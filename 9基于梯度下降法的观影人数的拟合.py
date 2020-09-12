from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

file = pd.read_csv('D:/aa罗/data/data/3_film.csv')

file.insert(1, 'ones', 1)
# 选取特征变量与响应变量
# print(file.head())

cols = file.shape[1]
x = file.iloc[:, 1:cols]
y = file.iloc[:, 0:1]
x = np.array(x.values)
y = np.array(y.values)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)


# 定义成本函数
def cost(x, y, thera):
    inner = np.power(((x * thera.T) - y), 2)
    return np.sum(inner) / (2 * len(x))


# 构造梯度下降算法函数 核心算法
def gradDescent(x, y, theta, alpha, iters):  # thera是参数 alpha是学习率 iters是迭代次数
    temp = np.matrix(np.zeros(theta.shape))  # 构造零值矩阵
    param = int(theta.ravel().shape[1])  # 计算求解参数
    P_cost = np.zeros(iters)  # 构建迭代次数等同长度的个数
    for i in range(iters):
        error = (x * theta.T) - y  # 计算h0(x)-y
        for j in range(param):
            term = np.multiply(error, x[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(x)) * np.sum(term))

        theta = temp
        P_cost[i] = cost(x, y, theta)
    return theta, P_cost


# 设定相关参数的初始值
alpha = 0.000001
iters = 100
theta = np.matrix(np.array([0, 0, 0, 0]))
g, b_cost = gradDescent(x, y, theta, alpha, iters)
y_hat = x_test * g.T
# print(y_hat)
# print(y_test)

# 作图观察拟合图像
t = np.arange(len(x_test))
plt.plot(t, y_test, 'r', label='y_test')
plt.plot(t, y_hat, 'b', label='y_hat')
plt.legend()
plt.show()


# 对预测结果评估
print('R方(拟合度):',r2_score(y_test,y_hat))
print('平均绝对差MAE:',metrics.mean_absolute_error(y_test,y_hat))
print('均方差MSE:',metrics.mean_squared_error(y_test,y_hat))
print('均方根差RMSE:',(metrics.mean_squared_error(y_test,y_hat))**0.5)

