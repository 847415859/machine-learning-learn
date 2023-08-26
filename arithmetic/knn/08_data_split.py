'''
    数据分割
'''
from sklearn.model_selection import  LeaveOneOut,KFold,StratifiedKFold
import numpy as np



# 留一法 Leave-One-Out
data = [1, 2, 3, 4]
loo = LeaveOneOut()
for train,test in loo.split(data):
    print("%s %s" % (train,test))

''' 结果：
[1 2 3] [0]
[0 2 3] [1]
[0 1 3] [2]
[0 1 2] [3]
'''


# KFold 和 StratifiedFold
X = np.array([
    [1,2,3,4],
    [11,12,13,14],
    [21,22,23,24],
    [31,32,33,34],
    [41,42,43,44],
    [51,52,53,54],
    [61,62,63,64],
    [71,72,73,74]
])

y = np.array([1,1,0,0,1,1,0,0])
# folder = KFold(n_splits = 4, random_state=1, shuffle = True)
folder = KFold(n_splits = 4, shuffle = False)
# sfolder = StratifiedKFold(n_splits = 4, random_state = 1, shuffle = True)
sfolder = StratifiedKFold(n_splits = 4,  shuffle = False)

print("KFold:")
for train, test in folder.split(X, y):
    print('train:%s | test:%s' %(train, test))

print("StratifiedFold:")
for train, test in sfolder.split(X, y):
    print('train:%s | test:%s'%(train, test))

''' 结果
KFold:
train:[2 3 4 5 6 7] | test:[0 1]
train:[0 1 4 5 6 7] | test:[2 3]
train:[0 1 2 3 6 7] | test:[4 5]
train:[0 1 2 3 4 5] | test:[6 7]
StratifiedFold:
train:[1 3 4 5 6 7] | test:[0 2]
train:[0 2 4 5 6 7] | test:[1 3]
train:[0 1 2 3 5 7] | test:[4 6]
train:[0 1 2 3 4 6] | test:[5 7]
'''