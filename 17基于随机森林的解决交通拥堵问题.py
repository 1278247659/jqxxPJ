from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('d:/aa罗/data/data/7_traffic.csv')
print(df.head())
x = df.iloc[:, :6]
y = df['traffic']
print(y.values)
x.hist(figsize=(7, 10))
plt.show()
y.hist()
plt.show()

x = np.array(x.values)
y = np.array(y.values)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2)
# 决策树id3分类
id3 = DecisionTreeClassifier(criterion='entropy')
id3.fit(x_train, y_train)
y_pred = id3.predict(x_test)
print('ID3精度：', accuracy_score(y_test, y_pred))
print('ID3混淆矩阵：', confusion_matrix(y_test, y_pred))

# 随机森林分类
rf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, oob_score=0,random_state=0)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
print('随机森林的精度：', accuracy_score(y_test, y_pred))
print('随机森林的混淆矩阵：', confusion_matrix(y_test, y_pred))

#极端随机森林
erf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
erf.fit(x_train, y_train)
y_pred = erf.predict(x_test)
print('极端随机森林的精度：', accuracy_score(y_test, y_pred))
print('极端随机森林的混淆矩阵：', confusion_matrix(y_test, y_pred))