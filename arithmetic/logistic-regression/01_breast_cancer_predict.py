'''
    癌症分类预测-良／恶性乳腺癌肿瘤预测
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    # 1.获取数据
    names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
             'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
             'Normal Nucleoli', 'Mitoses', 'Class']

    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        names=names)
    # 2.数据基本处理
    # 文本中存在16个 ? 未知的数据
    # 2.1 去除空值的数据
    data = data.replace(to_replace="?",value=np.NAN)
    data = data.dropna()
    # 2.2 区分特征值和目标值数据
    features = data.iloc[:,:10]
    targets = data["Class"]
    # 2.3 区分训练集和测试集
    feature_train,feature_test,target_train,target_test = train_test_split(features,targets,test_size=0.2)
    # 3.特征工程（数据标准化）
    transfor = StandardScaler()
    feature_train = transfor.fit_transform(feature_train)
    feature_test = transfor.fit_transform(feature_test)
    # 4.机器学习（逻辑回归）
    estimator = LogisticRegression()
    estimator.fit(feature_train,target_train)
    # 5.模型评估
    predicts = estimator.predict(feature_test)
    print(predicts)
    score = estimator.score(feature_test,target_test)
    print("准确率",score)
    # 分类评估报告
    ret = classification_report(predicts,target_test)
    print(ret)
    '''
                      precision    recall  f1-score   support

                   2       0.99      0.95      0.97        95
                   4       0.89      0.98      0.93        42
        
            accuracy                           0.96       137
           macro avg       0.94      0.96      0.95       137
        weighted avg       0.96      0.96      0.96       137
    '''
    ret = classification_report(predicts,target_test,labels=(2,4), target_names=("良性","恶性"))
    '''
                       precision    recall  f1-score    support
    
                  良性       0.99      0.97      0.98        89
                  恶性       0.94      0.98      0.96        48
        
            accuracy                             0.97       137
           macro avg         0.96      0.97      0.97       137
        weighted avg         0.97      0.97      0.97       137
    '''
    print(ret)

    # AUC计算
    # 0.5~1之间，越接近于1约好
    print(np.where(predicts > 2.5, 1, 0))
    rocAucScore = roc_auc_score(predicts,target_test)
    print("AUC指标",rocAucScore)
