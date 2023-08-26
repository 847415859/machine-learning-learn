import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

'''
    数据来源： https://www.kaggle.com/c/facebook-v-predicting-check-ins
'''

## 具体步骤：
# 1.获取数据集
facebook_data =  pd.read_csv('./data/FBlocation/train.csv')
print(facebook_data.shape)
# 2.基本数据处理
# 2.1 缩小数据范围
facebook_data = facebook_data.query("x > 2.0 & x <2.5 & y> 2.0 & y<2.5")
# facebook_data = facebook_data.query("x <2.5 &  y<2.5")
# 2.2 选择时间特征 单位指定秒 默认毫秒ms
time =  pd.to_datetime(facebook_data["time"],unit="s")   # 返回值类型 <class 'pandas.core.series.Series'>
time = pd.DatetimeIndex(time)
facebook_data["day"] = time.day
facebook_data["hour"] = time.hour
facebook_data["weekday"] = time.weekday
print("缩小数据范围后的数据",facebook_data.shape)
# 2.3 去掉签到较少的地方
facebook_data_count = facebook_data.groupby("place_id")["row_id"].count()
facebook_data_filtered = facebook_data_count[facebook_data.groupby("place_id")["row_id"].count() > 3]
facebook_data = facebook_data[facebook_data["place_id"].isin(facebook_data_filtered.index)]
print("去掉签到较少的地方之后",facebook_data)
# 2.4 确定特征值和目标值
features = facebook_data[["x","y","accuracy","day","hour","weekday"]]
targets =facebook_data["place_id"]
# 2.5 分割数据集
feature_train, feature_test, target_train, target_test = train_test_split(features,targets,test_size=0.2,random_state=22)

# 3.特征工程 -- 特征预处理(标准化)
transfer = StandardScaler()
feature_train = transfer.fit_transform(feature_train)
feature_test = transfer.transform(feature_test)

# 4.机器学习 -- knn+cv
# 实例化估计其
estimator = KNeighborsClassifier()
# 交叉验证 + 网络搜索
# n_jobs: 使用几个CPU跑任务 -1,使用所有
estimator = GridSearchCV(estimator,param_grid={"n_neighbors":[1,3,5,7,9]},cv=5,n_jobs=-1)
# 模型训练
estimator.fit(feature_train,target_train)

# 5.模型评估
# 5.1 基本评估方式
score = estimator.score(feature_test,target_test)
print("最后预测的准确率为\n", score)
target_predict = estimator.predict(feature_test)
print("最后的预测值为:\n", target_predict)
print("预测值和真实值的对比情况:\n", target_predict == target_test)
# 5.2 使用交叉验证后的评估方式
print("在交叉验证中验证的最好结果:\n", estimator.best_score_)
print("最好的参数模型:\n", estimator.best_estimator_)
print("每次交叉验证后的验证集准确率结果和训练集准确率结果:\n",estimator.cv_results_)