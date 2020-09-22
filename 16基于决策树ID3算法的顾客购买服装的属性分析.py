import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from io import StringIO
df = pd.read_csv('d:/aa罗/data/data/7_buy.csv')
# print(df.info())
x = df.iloc[:, 0:4]
y = df['buy']
x = np.array(x.values)
y = np.array(y.values)
ID3 = DecisionTreeClassifier(criterion='entropy')
ID3.fit(x, y)
y_pred = ID3.predict(x)
print(y_pred, y)

print('精度：', accuracy_score(y_pred, y))
print('混淆矩阵：', confusion_matrix(y_pred, y))

# 生成决策树
fn = list(df.columns[:-1])
tn = ['0', '1']
# import pydotplus  # 写dot语言的接口
# from IPython.display import Image  # Image图形输出库
# from sklearn import tree
#
# dd = StringIO()
# tree.export_graphviz(ID3, out_file=dd, feature_names=fn, class_names=tn, filled=True, rounded=True,
#                      special_characters=True)
# graph = pydotplus.graph_from_dot_data(dd.getvalue())
# Image(graph.create_png())