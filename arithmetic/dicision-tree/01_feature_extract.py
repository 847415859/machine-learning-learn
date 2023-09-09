'''
    特征工程——特征提取
'''

import jieba
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

'''
    字典提取
'''
def dict_extract():
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]
    # 实例化转换器类
    transfer = DictVectorizer() # 默认 sparse=False
    data = transfer.fit_transform(data)
    print("离散 返回的结果：\n",data)
    print("离散 特征名称：\n",transfer.feature_names_)

    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]
    transfer = DictVectorizer(sparse=False)
    data = transfer.fit_transform(data)
    print("one-hot编码 返回的结果：\n", data)
    print("one-hot编码 特征名称：\n", transfer.feature_names_)
    pass


'''
    文本特征提取-英文
'''
def text_extraction_English():
    data = ["life is short,i like like python", "life is too long,i dislike python"]
    # 1、实例化一个转换器类
    transfer = CountVectorizer()
    # 2、调用fit_transform
    data = transfer.fit_transform(data)
    print("文本特征抽取的结果：\n", data.toarray())
    print("返回特征名字：\n", transfer.get_feature_names_out())
    pass

'''
    文本特征提取-中文
'''
def text_extraction_Chinese():
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    token_list = list()
    for text in data:
        token_list.append(cut_word(text))
    print(token_list)
    # 1、实例化一个转换器类
    transfer = CountVectorizer()
    # 2、调用fit_transform
    data = transfer.fit_transform(token_list)
    print("文本特征抽取的结果：\n", data.toarray())
    print("返回特征名字：\n", transfer.get_feature_names_out())
    pass


'''
    对中文进行分词
'''
def cut_word(text):
    return " ".join(list(jieba.cut(text)))

'''
    文本特征提取-中文
'''
def tfidf_extraction_Chinese():
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    token_list = list()
    for text in data:
        token_list.append(cut_word(text))
    print(token_list)
    # 1、实例化一个转换器类
    transfer = TfidfVectorizer()
    # 2、调用fit_transform
    data = transfer.fit_transform(token_list)
    print("文本特征抽取的结果：\n", data.toarray())
    print("返回特征名字：\n", transfer.get_feature_names_out())
    pass


if __name__ == '__main__':
    # print(cut_word("将分词结果变成字符串当作fit_transform的输入值"))
    tfidf_extraction_Chinese()
    pass