from matplotlib import pyplot as plt
from sklearn.datasets import load_boston
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

boston = load_boston()
data = boston.data
target = boston.target

a = pd.DataFrame(data)
b = pd.DataFrame(target)
a = a.iloc[:, 5:6]
b = target
# data = np.array(data.values)
# print(type(target))
# print(boston.feature_names)
# print(a[5])

# 做散点图 查看相关性

# plt.scatter(a[5], target)
# plt.xlabel('house number')
# plt.ylabel('house price')
# plt.title('house scatter picture')
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(a, b, train_size=0.50)

# 线性模型预测
from sklearn.linear_model import LinearRegression  # 线性逻辑回归模型

lr = LinearRegression()  # 实例化模型
lr.fit(x_train, y_train)  # 对训练集进行训练
print('截距：', lr.intercept_)
print('系数：', lr.coef_)
y_pred = lr.predict(x_test)
a = metrics.mean_squared_error(y_test, y_pred)
b = metrics.mean_absolute_error(y_test, y_pred)
c = a ** 0.5
print('均方差：', a)
print('绝对方差：', b)
print('均方根差：', c)


#引入R方 确定拟合度好坏
from sklearn.metrics import r2_score

print('拟合度R^2:',r2_score(y_test, y_pred))

# 作图观察结果
t = np.arange(len(x_test))
plt.plot(t, y_test, 'g', label='y_test')
plt.plot(t, y_pred, 'r', label='y_train')
plt.legend()
plt.show()
