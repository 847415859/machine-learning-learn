import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz

'''
    泰坦尼克号乘客生存预测
'''
if __name__ == '__main__':
    # 1.获取数据
    titanic = pd.read_csv("https://biostat.app.vumc.org/wiki/pub/Main/DataSets/titanic.txt")
    # print(titanic)
    # 2.数据基本处理
    # 2.1 确定特征值和目标值
    features = titanic[["pclass","age","sex"]]
    targets = titanic["survived"]
    # print(features)
    # print(targets)
    # 2.2 缺失数据处理
    features["age"].fillna(features["age"].mean(),inplace=True)
    # 2.3 数据集划分
    feature_train,feature_test,target_train,target_test = train_test_split(features,targets,test_size=0.2,random_state=11)
    # 3.特征工程（字典特征抽取）
    transfer = DictVectorizer(sparse=False)
    # 特征中出现类别符号，需要进行one - hot编码处理(DictVectorizer)
    # x.to_dict(orient="records") 需要将数组特征转换成字典数据
    # [{"pclass": "1st", "age": 29.00, "sex": "female"}, {}]
    print(feature_train.to_dict(orient="records"))
    feature_train = transfer.fit_transform(feature_train.to_dict(orient="records"))
    feature_test = transfer.fit_transform(feature_test.to_dict(orient="records"))
    print("特征值表头\n",transfer.get_feature_names_out())
    print(feature_train)
    # 4.机器学习（决策树）
    # 决策树API当中，如果没有指定max_depth那么会根据信息熵的条件直到最终结束。这里我们可以指定树的深度来进行限制树的大小
    estimator = DecisionTreeClassifier(criterion="entropy",max_depth=5)
    estimator.fit(feature_train,target_train)
    export_graphviz(estimator, out_file="./data/tree.dot", feature_names=['年龄', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])
    # 5.模型评估
    score = estimator.score(feature_test,target_test)
    print("模型评估 score: ",score)
    predict = estimator.predict(feature_test)
    print("对比真实值：",predict == target_test)
