from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

file = pd.read_csv(r'D:\aa罗\data\data\3_film.csv')
content = pd.DataFrame(file)

x = file.iloc[:, 1:4]
# print(x.head)
y = file.filmsize
x = np.array(x.values)
y = np.array(y.values)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)#ransom_state表示固定随机种子
# print(x_train.shape, y_train.shape)
lr = LinearRegression()
lr.fit(x_train, y_train)
print('系数：', lr.coef_)
print('截距：', lr.intercept_)
y_pred = lr.predict(x_test)
print('拟合度：', r2_score(y_test, y_pred))
print('均方差：', metrics.mean_squared_error(y_test, y_pred))
print('均方根差：', (metrics.mean_squared_error(y_test, y_pred)) ** 0.5)

# 做图观看拟合
t = np.arange(len(x_test))
plt.plot(t, y_test, 'g',label='y_test')
plt.plot(t, y_pred, 'r',label='y_pred')
plt.legend()
plt.show()