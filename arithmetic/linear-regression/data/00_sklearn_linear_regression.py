from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 生成特征和目标值
X =  6 *  np.random.rand(100,2)
y = 5 + 3 * X[:,0].reshape(-1,1) + 4 * X[:,1].reshape(-1,1) + np.random.randn(100,1)

# 模型训练
'''
    fit_intercept 是否计算截距项
    
'''
estimator = LinearRegression(fit_intercept=True)
estimator.fit(X,y)

print("Linear Regression 截距项为：",estimator.intercept_)
print("Linear Regression 参数为：",estimator.coef_)
# 预测
X_predict = np.array([[1,2],
                      [2,3],
                      [4,5],
                      [5,6]])
y_predict = estimator.predict(X_predict)
print(y_predict)


plt.plot(X[:,0],y,"b.")
plt.plot(X_predict[:,0],y_predict,"r-")
plt.show()