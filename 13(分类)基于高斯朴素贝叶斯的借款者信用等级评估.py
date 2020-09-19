import numpy as np, pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('d:/aa罗/data/data/6_credit.csv')
# print(df['credit']) 信用等级
print(df.head())
# print(df.head())
a = df[df['credit'] == 1]
b = df[df['credit'] == 2]
c = df[df['credit'] == 3]

# 可视化
str = 'income'
str1 = 'numbers'
plt.scatter(a[str], a[str1], c='r', marker='o', label='level-1')
plt.scatter(b[str], b[str1], c='b', marker='+', label='level-2')
plt.scatter(c[str], c[str1], c='g', marker='^', label='level-3')
plt.legend()
plt.show()

# 选取特征变量和响应变量
x = df.iloc[:, 1:6]
y = df['credit']
# print(x.info())
x = np.array(x.values)
y = np.array(y.values)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1)
from sklearn.naive_bayes import GaussianNB  # 高斯朴素贝叶斯包

gauss = GaussianNB()
gauss.fit(x_train, y_train)
print('训练样本集：', gauss.class_count_)
print('各个先验概论：', gauss.class_prior_)
print('特征值的均值：', gauss.theta_)
print('特征值的方差：', gauss.sigma_)
y_pred = gauss.predict(x_test)

# 混淆矩阵查看预测结果进行评价
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print('精度：', accuracy_score(y_test, y_pred))
print('混淆矩阵：', confusion_matrix(y_test, y_pred))

# 用逻辑回归预测结果进行评价
from sklearn.linear_model import LogisticRegression

log = LogisticRegression(solver='liblinear')
log.fit(x_train, y_train)
y_pred2 = log.predict(x_test)

print('逻辑回归精度：', accuracy_score(y_test, y_pred2))
print('逻辑回归混淆矩阵：', confusion_matrix(y_test, y_pred2))
print(df.shape[0])
