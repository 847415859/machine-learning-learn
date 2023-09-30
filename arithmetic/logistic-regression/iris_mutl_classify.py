'''
    逻辑回归实战——鸢尾花多样本分类（超过2个分类的目标值）
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 1.数据获取
data = load_iris(as_frame=True)
# print(data)
# 获取某一个特征
# "petal length (cm)",
X = data["data"][["petal length (cm)","petal width (cm)"]].values
y = data["target"]

# 2.模型训练
lr = LogisticRegression(solver="sag",max_iter=10000,multi_class="ovr",C=1000)
lr.fit(X,y)

y_test = 4 * np.random.rand(5, 2)
print("预测为那种类型的概率 : ",lr.predict_proba(y_test))
print("预测为哪个类别：",lr.predict(y_test[:1]))

# 绘制 3-class 样子与之对应的决策边界 decision boundary
_, ax = plt.subplots(figsize=(4, 3))
DecisionBoundaryDisplay.from_estimator(
    lr,
    X,
    cmap=plt.cm.Paired,
    ax=ax,
    response_method="predict",
    xlabel="Sepal length",
    ylabel="Sepal width",
    eps=0.5,
)

plt.scatter(X[:,0],X[:,1],c=y,edgecolors="k",cmap=plt.cm.Paired)
plt.show()