from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

# 岭回归是为了解决过拟合的问题 适当添加参数
file = pd.read_csv('d:/aa罗/data/data/3_film.csv')
y = file.iloc[:, 0:1]
x = file.iloc[:, 1:]

x = np.array(x.values)
y = np.array(y.values)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# 导入线性模块
from sklearn import linear_model

# 从线性模块中导入岭回归算法
ridge = linear_model.Ridge(alpha=0.01)
ridge.fit(x_train, y_train)
y_pred = ridge.predict(x_test)
print('截距：', ridge.intercept_)
print('系数：', ridge.coef_)

# 可视化
x = np.arange(len(x_test))
plt.plot(x, y_test, 'r', linewidth=2, label='y_test')
plt.plot(x, y_pred, 'g', linewidth=2, label='y_pred')
plt.legend()
plt.show()

# 查看拟合度和误差
print('R方拟合度：', r2_score(y_test, y_pred))
mse = metrics.mean_squared_error(y_test, y_pred)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('均方差:', mse)
print('均方根差:', mse ** 0.5)
