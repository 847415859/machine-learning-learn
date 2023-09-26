import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


def variance_demo():
    '''
        删除低方差特征
    :return:
    '''
    data = pd.read_csv('./data/factor_returns.csv')
    print(data)
    # 实例化 转换器类
    # threshold 方差的阈值，低于这个阈值的特征的方差被删除
    transfer = VarianceThreshold(threshold=10)
    decomposition_data = transfer.fit_transform(data.iloc[:,1:10])
    print("删除低方差特征的结果：\n", decomposition_data)
    print("删除地方插特征前的形状：\n", data.iloc[:,1:10].shape)
    print("形状：\n", decomposition_data.shape)

def pearsonrCorrelationCoefficent():
    '''
         皮尔逊相关系数（Pearson Correlation Coefficient）
    :return:
    '''
    x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
    x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]
    result =  pearsonr(x1,x2)
    print(result)

def spearmanr_demo():
    '''
    斯皮尔曼相关系数
    :return:
    '''
    x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, -60.9]
    x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]
    print(spearmanr(a=x1, b=x2))

def pac():
    '''
    PCA
    :return:
    '''
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
    # 1、实例化PCA, 小数——保留多少信息
    transfer = PCA(n_components=0.9)
    transfer_data = transfer.fit_transform(data)
    print("保留90%的信息，降维结果为：\n", transfer_data)

    # 1、实例化PCA, 整数——指定降维到的维数
    transfer2 = PCA(n_components=3)
    # 2、调用fit_transform
    transfer_data = transfer2.fit_transform(data)
    print("降维到3维的结果：\n", transfer_data)

if __name__ == '__main__':
    # variance_demo()
    # pearsonrCorrelationCoefficent()
    spearmanr_demo()
    # pac()
    pass