from sklearn.neighbors import KNeighborsClassifier

# 构造数据集一
x = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

# 构造数据集二
ds_x = [[39,0,31],[3,2,65],[2,3,55],[9,38,2],[8,34,17],[5,2,57],[21,17,5],[45,2,9]]
ds_y = [0,1,2,2,2,2,1,1]

# 模型训练
# 实例化API
estimator = KNeighborsClassifier(n_neighbors=1)
# 使用fit方法进行训练
estimator.fit(x, y)
# 预测100所对于的目标值
ret = estimator.predict([[100]])
print(f"数据集一训练出来的结果为：{ret}")

# 数据集格式二对应的测试数据
estimator.fit(ds_x,ds_y)
ret = estimator.predict([[9,39,2]])
print(f"数据集二训练出来的结果为：{ret}")