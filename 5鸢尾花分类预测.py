from sklearn import datasets  # 导入自带的鸢尾花分类数据
from sklearn.model_selection import train_test_split  # 分离模板（大量数据只选择其中一部分进行预测）
import numpy as np

data_target = datasets.load_iris()

# 数据预处理
data = data_target.data  # 导入数据中的训练集
target = data_target.target  # 导入数据中的结果集

x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.3)

# 不同分类算法模型的比较

# 逻辑回归算法
from sklearn.linear_model import LogisticRegression  # 线性逻辑回归模型

lr = LogisticRegression()  # 实例化模型
lr.fit(x_train, y_train)  # 对训练集进行训练
print(lr.coef_)  # 查看训练模型的参数

# 评价算法优度好坏
y_pred = lr.predict(x_test)  # 对测试集数据用模型得出的参数进行预测
from sklearn import metrics  # 导入性能指标库

# 差值越小表示模型效果越好
print('逻辑回归的均方差：', metrics.mean_squared_error(y_test, y_pred))
print('均方根差：', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# 决策树分类
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类算法

dt = DecisionTreeClassifier(max_depth=2)  # 实例化算法
dt.fit(x_train, y_train)  # 对训练集进行训练
y_pred = dt.predict(x_test)
# 差值越小表示模型效果越好
print('DecisionTreeClassifier预测的均方差：', metrics.mean_squared_error(y_test, y_pred))
print('均方根差：', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# K邻近算法
from sklearn.neighbors import KNeighborsClassifier  # 导入K邻近分类算法

kn = KNeighborsClassifier()
kn.fit(x_train, y_train)
y_pred = kn.predict(x_test)
# 差值越小表示模型效果越好
print('KNeighborsClassifier的均方差：', metrics.mean_squared_error(y_test, y_pred))
print('均方根差：', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# SVM支持向量机算法
from sklearn import svm

svc = svm.SVC()
svc.fit(x_train, y_train)
y_pred =svc.predict(x_test)
# 差值越小表示模型效果越好
print('SVM的均方差：', metrics.mean_squared_error(y_test, y_pred))
print('均方根差：', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#综上的误差分析 即可确定最优的逻辑回归算法
