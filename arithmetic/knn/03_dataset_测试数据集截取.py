from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载数据（鸢尾花数据集）
iris = load_iris()

print(iris)
print("特征值:\n",iris["data"])
print("目标值：\n",(iris.target))

print(iris.data)
# 对鸢尾花数据集进行分割，训练集的特征值x_train 测试集的特征值x_test 训练集的目标值y_train 测试集的目标值y_test
# test_size 测试集所占有的比例
# random_state 随机数种子,相同的随机数种子划分出来的数据集相同
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=11)
print("训练集的特征值：\n" , x_train)
print("训练集的目标值：\n" , y_train)
print("测试集的特征值：\n" , x_test)
print("测试集的特征值：\n" , y_test)
#  [2 2 2 1 2 0 1 0 0 1 2 1 1 2 2 0 2 1 2 2 1 0 0 1 0 0 2 1 0 1]
x_train1,x_test1,y_train1,y_test1 = train_test_split(iris.data,iris.target,test_size=0.2,random_state=11)
print("测试集的特征值1(相同随机种子)：\n" , y_test1)
#  [2 2 2 1 2 0 1 0 0 1 2 1 1 2 2 0 2 1 2 2 1 0 0 1 0 0 2 1 0 1]
x_train2,x_test2,y_train2,y_test2 = train_test_split(iris.data,iris.target,test_size=0.2,random_state=12)
print("测试集的特征值（不同随机种子）：\n" , y_test2)
# [0 2 0 1 2 2 2 0 2 0 1 0 0 0 1 2 2 1 0 1 0 1 2 1 0 2 1 1 0 0]