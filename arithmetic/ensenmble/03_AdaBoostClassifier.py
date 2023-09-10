from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier

if __name__ == '__main__':
    # 生成1个特征，2维 100个样例
    X, y = make_classification(n_samples=100, n_features=2,
            n_informative = 2, n_redundant = 0,
            random_state = 0, shuffle = False)
    print(X[1:10])
    print(y)

    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)

    print("分类目标值: ",clf.classes_)
    print("分类预测概率: ",clf.predict_proba([[1, 0]]))
    print("预测为是哪个分类目标值: ",clf.predict([[0, 0]]))
