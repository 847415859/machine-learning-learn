'''
    随机森林预测demo
    利用决策树《泰坦尼克号》案例基础上修改
'''
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

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
    # 4.机器学习
    # 机器学习—— 随机森林进行预测
    rf = RandomForestClassifier()
    # 超参数调优
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(feature_train, target_train)
    print("best_score_\n",gc.best_score_)
    print("best_estimator_\n",gc.best_estimator_)
    score = gc.score(feature_test,target_test)
    print("随机森林预测的准确率为：", score)
