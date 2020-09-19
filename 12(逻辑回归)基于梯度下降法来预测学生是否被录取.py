from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


# 定义sigomod 函数
def sigmod(x):
    return 1 / (1 + np.exp(-x))


# 预测结果 将结果预测在（1,0）
def predict(theta, x):
    a = sigmod(x * theta.T)
    return [1 if i >= 0.5 else 0 for i in a]


# 定义梯度下降算法
def gradDescent(x, y, theta, alpha, m, numIter):
    # theta是参数 alpha是学习率 m是样本数 numIter是迭代次数
    xtrans = x.T
    for i in range(0, numIter):
        theta = np.matrix(theta)
        pred = np.array(predict(theta, x))
        loss = pred - y
        gradient = np.dot(xtrans, loss) / m
        theta = theta - alpha * gradient
    return theta


# 加载数据并进行可视化
df = pd.read_csv('d:/aa罗/data/data/5_logisitic_admit.csv')
df.insert(1, 'ones', 1)
pos = df[df['admit'] == 1]
neg = df[df['admit'] == 0]
# print(pos.head())
# print(neg.head())

plt.subplots(figsize=(8, 5))
plt.scatter(pos['gre'], pos['gpa'], s=30, c='b', marker='o', label='admit')
plt.scatter(neg['gre'], neg['gpa'], s=30, c='r', marker='x', label='mot admit')
plt.legend()
plt.show()
# plt.set_xlabel('gre')
# plt.set_xlabel('gpa')

# 预测
x = df.iloc[:, 1:4]
y = df['admit']
x = np.array(x.values)
y = np.array(y.values)
m, n = x.shape
thera = np.ones(n)
num = 1000
alpha = 0.0005
thera = gradDescent(x, y, thera, alpha, m, num)
print(thera)

pred = predict(thera, x)
corr = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(pred, y)]
accu = (sum(map(int, corr)) % len(corr))
print('{:.2f}%'.format((accu * 100) / m))


from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log.fit(x,y)
y_pred = log.predict(x)
corr = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(y_pred, y)]
accu = (sum(map(int, corr)) % len(corr))
print('{:.2f}%'.format((accu * 100) / m))


