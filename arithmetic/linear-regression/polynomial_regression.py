'''
    特征升维 增加多项式 polynomail
'''
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# 求误差
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1.生成数据
np.random.seed(20)
m = 100
X = 6 * np.random.rand(m,1) - 3     # [-3,3)
y = 0.5 * X**2 + X + 2  + np.random.randn(m,1)

plt.scatter(X,y,c='b',marker='.')

# 2.切片操作把数据分为训练集和测试集
X_train = X[:80]
y_train = y[:80]
X_test = X[80:]
y_test = y[80:]

styles =  {1:('g','.'),2:('r','+'), 20: ('y','*')}

for n_degree in styles:
    # 多项式升维  include_bias =True 保留截距项
    poly_features = PolynomialFeatures(degree=n_degree,include_bias=True)
    X_poly_train = poly_features.fit_transform(X_train)
    X_poly_test = poly_features.fit_transform(X_test)
    print(X_train.shape,X_train[0])
    print(X_poly_train.shape,X_poly_train[0])

    reg =  LinearRegression(fit_intercept=False)
    reg.fit(X_poly_train,y_train)
    print(reg.intercept_,reg.coef_)
    # 看看是否随着 degree 的维度增加，是否过拟合了
    y_train_predict = reg.predict(X_poly_train)
    y_test_predict = reg.predict(X_poly_test)
    # 画出不同维度对应的图像
    plt.scatter(X_poly_train[:,1],y_train_predict,c = styles[n_degree][0], marker=styles[n_degree][1])

    # 使用均方误差进行测试
    print(f"{n_degree}项 训练集MSE :",mean_squared_error(y_train,y_train_predict))
    print(f"{n_degree}项 测试集MSE :",mean_squared_error(y_test,y_test_predict))

plt.show()