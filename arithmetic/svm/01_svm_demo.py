from sklearn import  svm


if __name__ == '__main__':
    X = [[0, 0], [1, 1]]
    y = [0,1]
    clf = svm.SVC()
    clf.fit(X,y)
    print(clf)

    y_predict = clf.predict([[2,2]])
    print(y_predict)
