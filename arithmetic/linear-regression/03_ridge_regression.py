
'''
    波士顿房屋预测——岭回归
'''


# load_boston` has been removed from scikit-learn since version 1.2.
# from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,RidgeCV
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
    利用岭回归预测波士顿房价
'''
def ridge_regression():
    # 1.提取数据
    feature,target = load_boston_data()
    # 2.数据处理——训练、测试数据集的切割
    feature_train,feature_test,target_train,target_test = train_test_split(feature,target,test_size=0.2)
    # 3.特征工程——标准化
    transfer = StandardScaler()
    feature_train = transfer.fit_transform(feature_train)
    feature_test = transfer.fit_transform(feature_test)
    # 4.机器学习/模型训练——线性回归（Ridge Regression）
    # estimator = Ridge(alpha=1.0)
    # 交叉验证
    estimator = RidgeCV(alphas=(0.1,1,10))
    estimator.fit(feature_train,target_train)
    # 5.模型评估
    feature_predict = estimator.predict(feature_test)
    # print("预测值为",feature_predict)
    print("模型中的系数为",estimator.coef_)
    print("模型中的偏置为",estimator.intercept_)
    print("模型最好分数为",estimator.best_score_)

    # 5.2 评价——均方误差
    mse =  mean_squared_error(feature_predict,target_test)
    print("MSE:",mse)

if __name__ == '__main__':
    ridge_regression()