from sklearn.linear_model import LinearRegression

# 构造数据集
x = [[80, 86],
[82, 80],
[85, 78],
[90, 90],
[86, 82],
[82, 90],
[78, 80],
[92, 94]]
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

# 机器学习
estimator = LinearRegression()
# 使用fit方法进行训练
estimator.fit(X=x,y=y)

print("训练出来的回归系数为:",estimator.coef_)
print("训练出来的偏置为:",estimator.intercept_)
print("预测出来的目标值",estimator.predict([[100,80]]))