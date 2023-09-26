# 代码实战解析解求解模型的方法
import numpy as np
import matplotlib.pyplot as plt

# 线性回归，有监督的机器学习 X,y

# 1、 求解出权重 θ
# 生成100行1列的二维矩阵
X = 2 * np.random.rand(100,1)
# print(X)
# 模拟出来数据y是代表真实的数据，所有就是 y_hat + error
y = 5 + 4 * X + np.random.randn(100,1)    # np.random.randn 返回一个符合正态分布的样本
# print(y)
# 为了求解 w_0 横距: 需要借 X矩阵加上一列全为1的 x_0
X_b = np.c_[np.ones((100,1)),X]
# print(X_b)
# 计算出 θ   计算公式  [X^TX]^-1 X^T y
θ = np.linalg.inv(np.dot(X_b.T,X_b)).dot(X_b.T).dot(y)
print("通过求解公式求的 θ 权重的求解为：",θ)

# 2、根据求解出的θ进行预测
# 生产新的X
X_new = np.array([[0],
                  [2]])
X_new_b = np.c_[np.ones((2,1)),X_new]
print(X_new_b)
# 预测计算公式 y_hat = θX
y_predict = X_new_b.dot(θ)
print("预测值为：\n",y_predict)


# 3.绘图查看效果
plt.plot(X_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.plot(X_new, y_predict, 'g.')
plt.axis([0, 2, 0, 15])
plt.show()
