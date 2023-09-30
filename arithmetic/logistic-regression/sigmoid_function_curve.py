import math
import numpy as np
import matplotlib.pyplot as plt

'''
    根据sigmoid画出 S 形曲线
'''
# sigmoid 函数
#  sigmoid函数公式: 1/(1+e^(-x))
def sigmoid(x):
    y = []
    for i in x:
        y.append(1.0 / (1.0 + math.exp(-i)))
    return y

# 随机 -10,10 连续性数值间隔为0.1
x = np.arange(-10,10,0.1)
y = sigmoid(x=x)

# 画出连续的线性画图
plt.plot(x,y)
plt.show()