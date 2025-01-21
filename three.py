from sklearn.feature_extraction import DictVectorizer

def dict_demo():
    """
    字典特征抽取
    :return:
    """
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]
    #1.实例化转换器类
    transfer = DictVectorizer(sparse=False)
    #2. 调用fit-transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("data_name\n", transfer.feature_names_)
    return None
if __name__ == "__main__":
    dict_demo()