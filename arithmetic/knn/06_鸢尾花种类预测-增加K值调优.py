from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# 1.数据提取
iris = load_iris()
# 2.数据的预处理
# x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,train_size=0.2,random_state=22)
x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,train_size=0.2)
# 3.特征工程 ; 标准化
transfer =  StandardScaler()
x_train = transfer.fit_transform(x_train)
# x_test = transfer.fit_transform(x_test)
# 为什么测试集可以用训练集里面的方差？
# 不管数据怎么划分，对于数据的分布都是差不多的, 测试集取数据的分布状态的与训练集一致的（整体的是正态分布，取出来的部分也属于正态分布）
x_test = transfer.transform(x_test)
# 4.模型训练（机器学习）
# 4.1 实例化预估器类
estimator = KNeighborsClassifier()
# 4.2 模型选择与调优——网络搜索和交叉验证
param_dict = {"n_neighbors": [1, 3, 5]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
# 4.3 fit数据进行训练
estimator.fit(x_train,y_train)
# 5.模型评估
# 方法一： 对比真实值和预测值
predicts = estimator.predict(x_test)
print("预测的结果为\n",predicts)
print("对比真实值和预测值\n", predicts == y_test)
# 方法二：直接计算准确率
score = estimator.score(x_test,y_test)
print("准确率为\n",score)

print("在交叉验证中验证的最好结果：\n", estimator.best_score_)
print("最好的参数模型：\n", estimator.best_estimator_)
print("每次交叉验证后的准确率结果：\n", estimator.cv_results_)