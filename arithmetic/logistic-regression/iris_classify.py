'''
    逻辑回归实战——鸢尾花分类
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# 1.数据获取
data = load_iris(as_frame=True)
# 获取某一个特征
X = data["data"]["petal width (cm)"].values.reshape(-1,1)
# 因为目标值有3个，逻辑回归只能处理二分类问题，需要对目标值做处理
y = np.array(data["target"] / 2,dtype=int)

# 2.模型训练
lr = LogisticRegression(solver="sag",max_iter=10000)
lr.fit(X,y)

y_test = 4 * np.random.rand(5, 1)
print("预测为那种类型的概率 : ",lr.predict_proba(y_test))
print("预测为哪个类别：",lr.predict(y_test[:1]))
