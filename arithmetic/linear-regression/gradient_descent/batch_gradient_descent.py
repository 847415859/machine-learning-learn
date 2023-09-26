'''
    手动实现全量梯度下降法
'''
import numpy as np

# 1.创建数据集
X = np.random.rand(100,1)
# 加上横截距的列向量
X_b = np.c_[np.ones((100,1)),X]
# 生成目标值
y = 4 + 3 * X + np.random.randn(100,1)

# 2. 创建超参数
# 学习率
n_learning_rate = 0.001
# 最大迭代次数
n_iter = 10000

# 3.实现全量梯度下降的程序
# 3.1 初始化权重
theta = np.random.randn(2,1)
# 3.4 迭代计算，判断是否收敛

for _ in range(n_iter):
    # 3.2 计算梯度  公式： gradient_j = (y_hat - y) * X_j  = (WX - y) * X_j
    # 计算维度0（横截距）的梯度下降值
    # gradient_0 = X_b.dot(theta).T.dot(np.array(X_b[:,0]).reshape(-1,1))
    # (m,2)^T 转化为 (2,m) 点乘上 (m,1)的值
    gradients = X_b.T.dot(X_b.dot(theta) - y)
    # 3.3 重新计算权重  公式：θ = θ - η * gradient
    theta = theta - n_learning_rate * gradients

print("经过全量梯度下降算出的 θ",theta)