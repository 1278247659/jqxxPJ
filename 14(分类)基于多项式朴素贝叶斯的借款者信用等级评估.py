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
# str = 'income'
# str1 = 'numbers'
# plt.scatter(a[str], a[str1], c='r', marker='o', label='level-1')
# plt.scatter(b[str], b[str1], c='b', marker='+', label='level-2')
# plt.scatter(c[str], c[str1], c='g', marker='^', label='level-3')
# plt.legend()
# plt.show()

# 选取特征变量和响应变量
x = df.iloc[:, 1:6]
y = df['credit']
# print(x.info())
x = np.array(x.values)
y = np.array(y.values)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

from sklearn.naive_bayes import MultinomialNB

mul = MultinomialNB()
mul.fit(x_train, y_train)
y_pred = mul.predict(x_test)
# print(y_pred)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print('精度：',accuracy_score(y_test,y_pred))
print('混淆矩阵：',confusion_matrix(y_test,y_pred))