
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd


if __name__ == '__main__':
    # 准备类比不平衡数据
    # 使用make_classification生成样本数据
    X, y = make_classification(n_samples=5000,
                               n_features=2,  # 特征个数= n_informative（） + n_redundant + n_repeated
                               n_informative=2,  # 多信息特征的个数
                               n_redundant=0,  # 冗余信息，informative特征的随机线性组合
                               n_repeated=0,  # 重复信息，随机提取n_informative和n_redundant 特征
                               n_classes=3,  # 分类类别
                               n_clusters_per_class=1,  # 某一个类别是由几个cluster构成的
                               weights=[0.01, 0.05, 0.94],  # 列表类型，权重比
                               random_state=0)
    data = pd.DataFrame(X)
    # 查看各个样本的样本
    counter = Counter(y)
    print(counter)  # Counter({2: 4674, 1: 262, 0: 64})
    # 数据的可视化(分类散点图)
    plt.scatter(data.iloc[:,0],data.iloc[:,1],c=y)
    plt.show()