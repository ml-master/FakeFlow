import numpy as np
from sklearn.utils import shuffle

# 设置随机种子，确保结果的可重复性
np.random.seed(0)

# 定义一个函数，用于分割数据集
def split(data, data_features, return_features, dev_ratio=0.176):
    # 从数据集中分离出训练集
    train = data[data.type == 'training']
    # 如果需要返回特征，则根据训练集的索引提取对应的特征
    train_features = data_features[train.index, :, :] if return_features else []
    # 打乱训练集的顺序
    train = shuffle(train)
    # 重置索引，并删除原始的id列
    train = train.reset_index(drop=True).reset_index()
    del train['id']
    # 重命名列名，将新的索引列命名为'id'
    train = train.rename(columns={'index': 'id'})

    # 从数据集中分离出测试集
    test = data[data.type == 'test']
    # 如果需要返回特征，则根据测试集的索引提取对应的特征
    test_features = data_features[test.index, :, :] if return_features else []
    # 打乱测试集的顺序
    test = shuffle(test)
    # 重置索引，并删除原始的id列
    test = test.reset_index(drop=True).reset_index()
    del test['id']
    # 重命名列名，将新的索引列命名为'id'
    test = test.rename(columns={'index': 'id'})

    # 初始化三个字典，分别用于存储训练集、验证集和测试集
    self_train = {}
    self_dev = {}
    self_test = {}

    # 根据dev_ratio生成一个布尔掩码，用于从训练集中分离出验证集
    msk_dev = np.random.rand(len(train)) < dev_ratio
    # 根据掩码提取验证集的文本、特征和标签
    self_dev['text'] = train['content'][msk_dev]
    self_dev['features'] = train_features[msk_dev, :, :] if return_features else []
    self_dev['label'] = train['label'][msk_dev]
    # 从训练集中移除验证集部分，剩余的作为新的训练集
    train = train[~msk_dev]
    # 提取新的训练集的文本、特征和标签
    self_train['text'] = train['content']
    self_train['features'] = train_features[~msk_dev, :, :] if return_features else []
    self_train['label'] = train['label']

    # 提取测试集的文本、特征和标签
    self_test['text'] = test['content']
    self_test['features'] = test_features if return_features else []
    self_test['label'] = test['label']
    # 返回分割后的训练集、验证集和测试集
    return self_train, self_dev, self_test
