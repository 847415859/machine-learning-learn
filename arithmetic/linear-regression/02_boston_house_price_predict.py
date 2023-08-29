'''
    波士顿房价预测
'''
# load_boston` has been removed from scikit-learn since version 1.2.
# from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.metrics import mean_squared_error

'''
    load_boston` has been removed from scikit-learn since version 1.2.
    加载波士顿房价数据
'''
def load_boston_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    # feature data
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    # target data
    target = raw_df.values[1::2, 2]
    return data,target

'''
    线性回归：正规方程
'''
def linear_model1():
    # 1.获取数据
    data,target = load_boston_data()
    # 2.数据预处理——数据集划分
    feature_train,feature_test,target_train,target_test = train_test_split(data,target,test_size=0.2,random_state=22)
    # 3.特征工程——标准化
    transfer = StandardScaler()
    feature_train = transfer.fit_transform(feature_train)
    feature_test = transfer.transform(feature_test)
    # 4.模型训练
    estimator = LinearRegression()
    estimator.fit(feature_train, target_train)
    # 5.模型评估
    # 5.1 获取系数等值
    target_predict = estimator.predict(feature_test)
    print("预测值为",target_predict)
    print("模型中的系数为",estimator.coef_)
    print("模型中的偏置为",estimator.intercept_)

    # 5.2 评价 均方误差
    error = mean_squared_error(target_predict,target_test)
    print("误差为",error)
    return None


'''
    线性回归：梯度下降法
'''
def linear_model2():
    # 1.获取数据
    data,target = load_boston_data()
    # 2.数据预处理——数据集划分
    feature_train,feature_test,target_train,target_test = train_test_split(data,target,test_size=0.2,random_state=22)
    # 3.特征工程——标准化
    transfer = StandardScaler()
    feature_train = transfer.fit_transform(feature_train)
    feature_test = transfer.transform(feature_test)
    # 4.模型训练
    # estimator = SGDRegressor(max_iter=1000)
    estimator = SGDRegressor(max_iter=1000,learning_rate="constant",eta0=0.0001)
    estimator.fit(feature_train, target_train)
    # 5.模型评估
    # 5.1 获取系数等值
    target_predict = estimator.predict(feature_test)
    print("预测值为",target_predict)
    print("模型中的系数为",estimator.coef_)
    print("模型中的偏置为",estimator.intercept_)

    # 5.2 评价 均方误差
    error = mean_squared_error(target_predict,target_test)
    print("误差为",error)
    return None

print("线性回归：正规方程")
linear_model1()
print("线性回归：梯度下降法")
linear_model2()