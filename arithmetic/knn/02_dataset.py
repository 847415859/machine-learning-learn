from sklearn.datasets import load_iris,fetch_20newsgroups
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 1.获取鸢尾花数据集
iris = load_iris()
print("鸢尾花数据集的返回值：\n", iris)
# 返回值是一个继承自字典的Bench
print("鸢尾花的特征值:\n", iris["data"])
print("鸢尾花的目标值：\n", iris.target)
print("鸢尾花特征的名字：\n", iris.feature_names)
print("鸢尾花目标值的名字：\n", iris.target_names)
print("鸢尾花的描述：\n", iris.DESCR)


# print("="*100)
# 2. 拉取大数据集
# news = fetch_20newsgroups()
# print(news)

# 3.数据集的可视化
# 把数据转换成dataframe的格式
iris_d = pd.DataFrame(iris['data'], columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
iris_d['Species'] = iris.target

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体

def plot_iris(iris, col1, col2):
    sns.lmplot(x = col1, y = col2, data = iris, hue = "Species", fit_reg = False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('鸢尾花种类分布图')
    plt.show()

plot_iris(iris_d, 'Petal_Width', 'Sepal_Length')

# 4.数据集的划分



