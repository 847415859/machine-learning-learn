'''
随机梯度下降（只有一个样本进行求梯度）
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
# 最大迭代次数（轮次）
n_iter = 10000
# 样本数(批次)
m = len(X)

# 3.实现随机梯度下降的程序
# 3.1 初始化权重
theta = np.random.randn(2,1)
# 3.4 迭代计算，判断是否收敛
for iter in range(n_iter):
    for _ in range(m):
        # 获取随机到的特征和目标值
        random_index =  np.random.randint(m)
        x_i = X_b[random_index:random_index+1]
        y_i = y[random_index:random_index+1]
        # 3.2 计算梯度  公式： gradient_j = (y_hat - y) * X_j  = (WX - y) * x_j
        gradient_i =  x_i.T.dot(x_i.dot(theta) - y_i)
        # 3.3 重新计i权重  公式：θ = θ - η * gradient
        theta = theta - n_learning_rate * gradient_i

print("经过随机梯度下降算出的 θ: \n",theta)
