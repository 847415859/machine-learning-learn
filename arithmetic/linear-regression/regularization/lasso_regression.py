import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor

X = 2 * np.random.rand(100,1)
y = 4 + 3*X + np.random.randn(100,1)

# 岭回归 + L2正则项
# 公式： min_w || X_w - y||_2^2 + ||w_||_2^2
lasso_reg = Lasso(alpha=0.04, max_iter=10000)
lasso_reg.fit(X, y)
print("预测值: ",lasso_reg.predict([[1.5]]))
print("横截距: ",lasso_reg.intercept_)
print("系数: ",lasso_reg.coef_)


sgd_reg = SGDRegressor(penalty="l1",max_iter=10000)
sgd_reg.fit(X,y.ravel())
print("预测值: ",sgd_reg.predict([[1.5]]))
print("横截距: ",sgd_reg.intercept_)
print("系数: ",sgd_reg.coef_)
print("l2的权重为：",sgd_reg.alpha)


