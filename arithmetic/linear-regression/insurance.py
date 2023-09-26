import numpy as np
import  pandas as pd



# 读取数据
data = pd.read_csv('./data/insurance.csv')
print(data.isnull().describe())