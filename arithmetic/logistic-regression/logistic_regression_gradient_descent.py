'''
    逻辑回归——基于梯度下降实现
'''
import numpy as np
# 创建数据集
X = 6 * np.random.rand(100,1)
# 生成 0,1目标值
y = np.random.randint(0,2,100).reshape(-1,1)


def predict(x):
    return theta * x


def sigmoid(x):
    return 1 / (1+ np.exp(-predict(x)))


def compute_gradient():
    '''
        计算梯度
        逻辑回归的梯度函数 h_\theta()
    :return:
    '''
    # print(sigmoid(X) - y)
    return (sigmoid(X)  - y).T.dot(X)


# 设置超参数
# 初始化 \theta
theta = np.random.randn(1,1)
# 最大迭代次数
max_iter = 1000
# 学习率
alpha = 0.002

for i in range(max_iter):
    theta = theta - alpha * compute_gradient()

print("梯度下降法计算出的 theta为：",theta)
