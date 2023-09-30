'''
    逻辑回归损失函数
'''
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import  matplotlib.pyplot as plt
from sklearn.preprocessing import scale

# 1.加载breast_cancer数据
data = load_breast_cancer()
# 获取特征值，只获取前两个特征值，为了方便计算和画图
X = data["data"][:,:2]
# 获取目标值
y = data["target"]
# 2.求出两个维度对应的数据在逻辑回归算法下的最优解
lr = LogisticRegression(fit_intercept=False)
lr.fit(X,y)

# 获取最优解
theta1 = lr.coef_[0,0]
theta2 = lr.coef_[0,1]


# 在已知w1和w2的情况下，根据传进来数据X，返回数据的 y_predict
def p_theta_function(features,w1,w2):
    # sigmoid 函数
    z = w1 * features[0] + w2 * features[1]
    return 1 / (1 + np.exp(-z))


# 传入一份已知数据的X,y,如果在已知w1和w2的情况下，计算对应这份数据的Loss函数
def loss_function(samples_features, samples_labels, w1,w2):
    result = 0
    # 便利数据集中的每一条样本，并且计算每条样本的损失，进行累加求和
    for features,label in zip(samples_features,samples_labels):
        # 计算一条样本的 y_predict
        p_result = p_theta_function(features,w1,w2)
        loss_result =  -(1 * label * np.log(p_result) + (1-label) * np.log(1-p_result))
        result += loss_result
    return result

print(loss_function(X,y,theta1,theta2))


theta1_linspace = np.linspace(theta1 - 0.6, theta1 + 0.6,50)
theta2_linspace = np.linspace(theta2 - 0.6, theta2 + 0.6,50)

# theta_2不变，修改theta1
result_1 = np.array([loss_function(X,y,i,theta2) for i in theta1_linspace])
# theta_1不变，修改theta2
result_2 = np.array([loss_function(X,y,theta1,i) for i in theta2_linspace])
print(result_1)

plt.figure(figsize=(20,8))
plt.subplot(2,2,1)
plt.plot(theta1_linspace,result_1)
plt.ylabel("Loss")
plt.xlabel("theta1")

plt.subplot(2,2,2)
plt.plot(theta2_linspace,result_2)
plt.ylabel("Loss")
plt.xlabel("theta2")
#
# # 画出等高线
theta1_grid,theta2_grid = np.meshgrid(theta1_linspace, theta2_linspace)
#
loss_grid = loss_function(X,y, theta1_grid,theta2_grid)
plt.subplot(2,2,3)
plt.contour(theta1_grid,theta2_grid,loss_grid)
plt.ylabel("theta2")
plt.xlabel("theta1")

loss_grid = loss_function(X,y, theta1_grid,theta2_grid)
plt.subplot(2,2,4)
plt.contour(theta1_grid,theta2_grid,loss_grid,30)
plt.ylabel("theta2")
plt.xlabel("theta1")

# 画出3D图像
fig2 = plt.figure()
ax = Axes3D(fig2)
ax.plot_surface(theta1_grid,theta2_grid,loss_grid)
plt.show()