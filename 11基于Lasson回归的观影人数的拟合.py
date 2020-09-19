from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

file = pd.read_csv(r'D:/aa罗/data/data/3_film.csv')
x = file.iloc[:, 1:]
# print(x.head())
y = file.iloc[:, 0:1]
# print(y.head())

x = np.array(x.values)
y = np.array(y.values)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(x_train, y_train)
print('截距:', lasso.intercept_)
print('系数:', lasso.coef_)
y_pred = lasso.predict(x_test)

# 可视化
t = np.arange(len(x_test))

plt.plot(t, y_test, 'g', label='y_test')
plt.plot(t, y_pred, 'r', label='y_pred')
plt.legend()
plt.show()

# 查看拟合度和误差
print('R方:', r2_score(y_test, y_pred))
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
mse = metrics.mean_squared_error(y_test, y_pred)
print('均方差:', mse)
print('均方根差:', mse ** 0.5)
