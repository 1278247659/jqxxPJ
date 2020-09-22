import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('d:/aa罗/data/data/7_traffic.csv')
x = df.iloc[:, 0:6]
y = df['traffic']
x = np.array(x.values)
y = np.array(y.values)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2)

knn = KNeighborsClassifier(n_neighbors=5,weights='distance')
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print('精度1：', accuracy_score(y_test, y_pred))
print('混淆矩阵1：', confusion_matrix(y_test, y_pred))

knn = KNeighborsClassifier(n_neighbors=15,weights='distance')
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
# c = 0
#
# for i, a in enumerate(y_pred):
#     for j, b in enumerate(y_test):
#         if y_pred[a] == y_test[b] and i == j:
#             c += 1
#
# print(c / len(y_pred))

print('精度2：', accuracy_score(y_test, y_pred))
print('混淆矩阵2：', confusion_matrix(y_test, y_pred))


knn = KNeighborsClassifier(n_neighbors=15,weights='distance',metric='cosine')
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print('精度3：', accuracy_score(y_test, y_pred))
print('混淆矩阵3：', confusion_matrix(y_test, y_pred))


