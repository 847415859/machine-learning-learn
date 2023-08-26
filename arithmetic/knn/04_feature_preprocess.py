'''
    特征预处理
'''
from  sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd


'''
    归一化处理 MinMaxScaler
'''
def minMaxScaler():
    df = pd.read_csv("./data/dating.txt")
    print(df)
    # 实例化转换器类  默认[0,1]
    transfer = MinMaxScaler(feature_range=(1,10))
    # 调用fit_transform
    data = transfer.fit_transform(df[["milage","Liters","Consumtime"]])
    print("最小值最大值归一化处理的结果：\n", data)


'''
    标准化处理 StandardScaler
'''
def standardScaler():
    df = pd.read_csv("./data/dating.txt")
    print(df)
    # 实例化转换器类
    transfer = StandardScaler()
    # 调用fit_transform
    data = transfer.fit_transform(df[["milage", "Liters", "Consumtime"]])
    print("标准化的结果：\n", data)
    print("每一列特征的平均值\n",transfer.mean_)
    print("每一列特征的方差\n",transfer.var_)

# minMaxScaler()
standardScaler()