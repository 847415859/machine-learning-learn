import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor

X = 2 * np.random.rand(100,1)
y = 4 + 3*X + np.random.randn(100,1)

# 岭回归 + L2正则项
# 公式： min_w || X_w - y||_2^2 + ||w_||_2^2
elasticnet_reg = ElasticNet(alpha=0.04,l1_ratio=0.5, max_iter=10000)
elasticnet_reg.fit(X, y)
print("预测值: ",elasticnet_reg.predict([[1.5]]))
print("横截距: ",elasticnet_reg.intercept_)
print("系数: ",elasticnet_reg.coef_)

sgd_reg = SGDRegressor(penalty="elasticnet",max_iter=10000)
sgd_reg.fit(X,y.ravel())
print("预测值: ",sgd_reg.predict([[1.5]]))
print("横截距: ",sgd_reg.intercept_)
print("系数: ",sgd_reg.coef_)
print("l2的权重为：",sgd_reg.alpha)



